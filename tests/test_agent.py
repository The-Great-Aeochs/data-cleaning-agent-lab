"""Exercise the real LangChain graph using scripted model responses."""

import json
from pathlib import Path
from typing import Any

from langchain.messages import AIMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatGeneration, ChatResult

from runner import run_agent
from tools import TOOLS
from tracing import load_trace, redact

DATA = Path(__file__).resolve().parents[1] / "data"


class ScriptedModel(BaseChatModel):
    """Only the model is replaced: graph, tools, streaming, and exports are real."""

    responses: list[AIMessage]
    position: int = 0

    @property
    def _llm_type(self):
        return "scripted-test-model"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs: Any):
        response = self.responses[min(self.position, len(self.responses) - 1)]
        self.position += 1
        response = response.model_copy(deep=True)
        for index, tool_call in enumerate(response.tool_calls):
            tool_call["id"] = f"call_{self.position}_{index}"
        return ChatResult(generations=[ChatGeneration(message=response)])


def call(name, **arguments):
    return AIMessage(content="", tool_calls=[{"name": name, "args": arguments,
                                              "id": f"call_{name}", "type": "tool_call"}])


def finish():
    return call("CleaningSummary", summary="Finished inspection and validation.",
                changes=[], uncertainties=[])


def execute(tmp_path, responses, dataset="clean", **kwargs):
    model = ScriptedModel(responses=responses)
    snapshots = list(run_agent(DATA / f"{dataset}.csv", model=model,
                              runs_root=tmp_path, **kwargs))
    return snapshots[-1], snapshots


def test_exactly_three_runtime_hidden_tools():
    assert {t.name for t in TOOLS} == {"inspect_data", "transform_data", "check_quality"}
    for tool in TOOLS:
        assert "runtime" not in tool.tool_call_schema.model_json_schema()["properties"]


def test_clean_graph_streams_and_saves_replay(tmp_path):
    result, snapshots = execute(tmp_path, [call("inspect_data", reason="Inspect data."),
                                          call("check_quality", reason="Validate."), finish()])
    assert result["report"]["status"] == "completed", result["report"]
    assert len(snapshots) > 3
    assert result["report"]["tool_sequence"] == ["inspect_data", "check_quality"]
    trace = load_trace(Path(result["directory"]) / "run.json")
    assert trace["status"] == "completed"
    assert sum(e["kind"] == "action" for e in trace["events"]) == 2
    assert (Path(result["directory"]) / "messages.json").exists()


def test_graph_recovers_from_stale_version(tmp_path):
    responses = [call("inspect_data", reason="Inspect."),
                 call("transform_data", operation="trim_text", expected_version=9, reason="Trim."),
                 call("transform_data", operation="trim_text", expected_version=0, reason="Retry."),
                 call("transform_data", operation="normalize_currency", expected_version=1, reason="Codes."),
                 call("transform_data", operation="normalize_amounts", expected_version=2, reason="Amounts."),
                 call("transform_data", operation="normalize_dates", expected_version=3, reason="Dates."),
                 call("transform_data", operation="remove_exact_duplicates", expected_version=4, reason="Copies."),
                 call("check_quality", reason="Validate final copy."), finish()]
    result, _ = execute(tmp_path, responses, "formatting")
    assert result["report"]["status"] == "completed", result["report"]
    observations = [e["body"] for e in result["events"] if e["kind"] == "observation"]
    assert any(isinstance(o, dict) and "error" in o for o in observations)
    assert result["report"]["quality"]["removed_duplicates"] == 1


def test_model_cannot_claim_success_without_check(tmp_path):
    result, _ = execute(tmp_path, [finish()])
    assert result["report"]["status"] == "unverified"
    assert "clean" not in result["report"]["artifacts"]


def test_edit_without_inspection_is_rejected(tmp_path):
    result, _ = execute(tmp_path, [call("transform_data", operation="trim_text",
                                      expected_version=0, reason="Edit prematurely."), finish()])
    assert not result["report"]["changes"]
    assert any("Inspect the dataset" in json.dumps(e) for e in result["events"])


def test_model_limit_keeps_partial_trace(tmp_path):
    result, _ = execute(tmp_path, [call("inspect_data", reason="Again.")], model_limit=2)
    assert result["report"]["status"] == "limit_reached", result["report"]
    assert result["report"]["metrics"]["model_turns"] == 2
    assert (Path(result["directory"]) / "run.json").exists()


def test_redaction(monkeypatch):
    key = "sk-" + "a" * 40
    monkeypatch.setenv("OPENAI_API_KEY", key)
    assert key not in json.dumps(redact({"error": key, "nested": [key]}))


def test_tool_budget_is_enforced(tmp_path):
    result, _ = execute(tmp_path, [call("inspect_data", reason="Inspect."),
                                  call("check_quality", reason="Validate."), finish()], tool_limit=1)
    assert result["report"]["status"] == "limit_reached"
    assert "clean" not in result["report"]["artifacts"]


def test_invalid_final_data_is_never_published_as_clean(tmp_path):
    result, _ = execute(tmp_path, [call("inspect_data", reason="Inspect."),
                                  call("check_quality", reason="Check unchanged data."), finish()], "formatting")
    assert result["report"]["status"] == "unverified"
    assert not result["report"]["quality"]["passed"]
    assert "clean" not in result["report"]["artifacts"]


def test_check_before_final_edit_cannot_verify_completion(tmp_path):
    result, _ = execute(tmp_path, [call("inspect_data", reason="Inspect."),
                                  call("check_quality", reason="Check."),
                                  call("transform_data", operation="trim_text", expected_version=0,
                                       reason="Edit after check."), finish()], "formatting")
    assert not result["report"]["agent_checked_final_version"]
    assert result["report"]["status"] == "unverified"


def test_partial_conversation_contains_observations_after_limit(tmp_path):
    result, _ = execute(tmp_path, [call("inspect_data", reason="Inspect."),
                                  call("check_quality", reason="Check.")], model_limit=2)
    history = json.loads((Path(result["directory"]) / "messages.json").read_text())
    assert any(message["type"] == "human" for message in history)
    assert any(message["type"] == "tool" for message in history)


def test_all_flagged_graph_reports_blocked_without_clean_artifact(tmp_path):
    result, _ = execute(tmp_path, [call("inspect_data", reason="Inspect."),
                                  call("transform_data", operation="flag_unresolved", expected_version=0,
                                       reason="Preserve uncertain rows."),
                                  call("check_quality", reason="Validate."), finish()], "all_flagged")
    assert result["report"]["status"] == "blocked"
    assert "clean" not in result["report"]["artifacts"]
    assert not (Path(result["directory"]) / "clean.csv").exists()
