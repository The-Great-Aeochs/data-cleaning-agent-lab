"""Rubric isolation, binary verdicts, saved artifacts, and suite/UI wiring."""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from judges import RUBRICS, Verdict, build_payload, evaluate_trace, judge_answer


@pytest.fixture
def trace():
    return {"run_id": "sample", "goal": "Prepare orders for revenue analysis.",
            "events": [{"index": 1, "kind": "observation", "title": "Quality",
                        "body": {"eligible_rows": 24}},
                       {"index": 2, "kind": "assistant", "title": "Explanation",
                        "body": "I claim I fixed everything."}],
            "report": {"summary": {"summary": "24 orders are ready.", "changes": [],
                                   "uncertainties": []},
                       "quality": {"eligible_rows": 24}, "changes": [],
                       "status": "completed", "agent_checked_final_version": True,
                       "artifacts": {"clean": "/private/path/clean.csv"}}}


class FakeJudge:
    def __init__(self, verdict=None, error=None, parsing_error=None):
        self.verdict = verdict
        self.error = error
        self.parsing_error = parsing_error
        self.messages = []

    def with_structured_output(self, schema, **options):
        assert schema is Verdict
        assert options == {"method": "json_schema", "strict": True, "include_raw": True}
        return self

    def invoke(self, messages):
        self.messages = messages
        if self.error:
            raise self.error
        return {"parsed": self.verdict, "parsing_error": self.parsing_error,
                "raw": SimpleNamespace(usage_metadata={"input_tokens": 12},
                                       response_metadata={"model_name": "fake"})}


@pytest.mark.parametrize("score", [True, False, "1", "0", 0.5, 1.0, 2, -1, None])
def test_scores_reject_non_binary_integers(score):
    with pytest.raises(ValidationError):
        Verdict(score=score, explanation="Evidence", evidence_ids=["answer"])


def test_judges_receive_different_inputs(trace):
    relevant = build_payload(trace, "answer_relevancy")
    grounded = build_payload(trace, "groundedness")
    assert set(relevant) == {"task", "answer"}
    assert set(grounded) == {"answer", "evidence"}
    assert "event:1" in grounded["evidence"]
    assert "event:2" not in grounded["evidence"]  # agent prose is not corroboration
    assert grounded["evidence"]["host:artifacts"] == ["clean"]
    assert "judges" not in json.dumps(relevant)


def test_verdict_logs_prompt_input_usage_and_reference(trace):
    model = FakeJudge(Verdict(score=1, explanation="Ready outcome addresses the task.",
                              evidence_ids=["task", "answer"]))
    result = judge_answer(trace, "answer_relevancy", model)
    assert result["score"] == 1
    assert result["status"] == "scored"
    assert result["response_model"] == "fake"
    assert result["usage"]["input_tokens"] == 12
    assert len(result["prompt_sha256"]) == 64
    assert model.messages[0][0] == "system"
    assert json.loads(model.messages[1][1]) == result["input"]


@pytest.mark.parametrize("verdict", [
    Verdict(score=1, explanation="Supported", evidence_ids=["invented:999"]),
    Verdict(score=1, explanation="", evidence_ids=["answer"]),
    Verdict(score=1, explanation="Supported", evidence_ids=[]),
    None,
])
def test_invalid_judge_response_is_unscored(trace, verdict):
    result = judge_answer(trace, "groundedness", FakeJudge(verdict))
    assert result["status"] == "error"
    assert result["score"] is None


def test_transport_error_redacts_secret_and_does_not_score_zero(trace):
    model = FakeJudge(error=RuntimeError("failed sk-thisisafakecredential1234567890"))
    result = judge_answer(trace, "groundedness", model)
    assert result["score"] is None
    assert "[REDACTED]" in result["error"]["message"]


def test_missing_answer_and_oversize_do_not_call_model(trace):
    model = FakeJudge(error=AssertionError("should never call"))
    trace["report"]["summary"] = None
    assert "No final" in judge_answer(trace, "groundedness", model)["error"]["message"]
    trace["report"]["summary"] = {"summary": "x" * 60_001}
    result = judge_answer(trace, "groundedness", model)
    assert "no evidence was truncated" in result["error"]["message"]
    assert model.messages == []


def test_judge_models_are_configured_independently(monkeypatch, trace):
    monkeypatch.setenv("JUDGE_ANSWER_RELEVANCY_MODEL", "relevance-model")
    monkeypatch.setenv("JUDGE_GROUNDEDNESS_MODEL", "ground-model")
    for criterion, name in zip(RUBRICS, ("relevance-model", "ground-model")):
        result = judge_answer(trace, criterion, FakeJudge())
        assert result["model"] == name


def test_attempts_are_saved_independently_and_score_disagreement_is_visible(trace, tmp_path):
    calls = []

    def judge(value, criterion):
        calls.append(criterion)
        return {"status": "scored", "score": int(criterion == "answer_relevancy")}

    one = evaluate_trace(trace, tmp_path, judge)
    two = evaluate_trace(trace, tmp_path, judge)
    assert calls == list(RUBRICS) * 2
    assert one["passed"] is False
    assert one["artifact"] != two["artifact"]
    assert len(list(tmp_path.glob("evaluation-*.json"))) == 2
    assert json.loads(open(one["artifact"]).read())["judges"]["groundedness"]["score"] == 0


def test_ui_loads_suite_without_calls_and_preserves_unscored_cells(monkeypatch, tmp_path):
    import app
    monkeypatch.setattr(app, "ROOT", tmp_path)
    (tmp_path / "runs").mkdir()
    summary = {"passed": 1, "total": 1, "results": [
        {"case": "clean", "passed": True, "status": "completed", "run_id": "x",
         "judges": {"answer_relevancy": {"score": 1}, "groundedness": {"score": None}}}]}
    (tmp_path / "runs" / "live-evaluation.json").write_text(json.dumps(summary))
    rows, note, details, file = app.load_suite()
    assert rows[0][2:5] == [1, 1, None]
    assert details == summary
    assert "1/1" in note
    assert file is None


def test_saved_evaluation_cannot_read_arbitrary_path():
    from app import evaluate_saved_run, load_evaluation
    assert load_evaluation("../../etc") == ([], {}, None)
    with pytest.raises(Exception, match="Select a saved"):
        evaluate_saved_run("../../etc")


def test_suite_shares_backend_and_saves_checks_and_judges(monkeypatch, tmp_path):
    import eval_live
    from runner import ROOT
    monkeypatch.setattr(eval_live, "ROOT", tmp_path)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "clean.csv").write_bytes((ROOT / "data" / "clean.csv").read_bytes())
    expected = eval_live.CASES["clean"]
    monkeypatch.setattr(eval_live, "CASES", {"clean": expected})
    report = {"status": "completed", "agent_checked_final_version": True,
              "changes": [], "tool_sequence": ["inspect_data", "check_quality"],
              "quality": {"eligible_rows": 24, "flagged_rows": 0, "removed_duplicates": 0,
                          "order_value_by_currency": {"INR": "17512.50"}},
              "error": None, "metrics": {}}
    final = {"report": report, "events": [], "trace": {}, "run_id": "fake",
             "directory": str(tmp_path)}
    monkeypatch.setattr(eval_live, "run_agent", lambda path: iter([final]))
    monkeypatch.setattr(eval_live, "evaluate_trace", lambda *args: {
        "passed": False, "judges": {"groundedness": {"score": 0, "status": "scored"}}})
    updates = list(eval_live.evaluate_steps())
    assert updates[0]["complete"] is False
    assert updates[-1]["complete"] is True
    assert updates[-1]["passed"] == 1  # Python passed; groundedness did not
    assert updates[-1]["overall_passed"] == 0
    saved = json.loads((tmp_path / "runs" / "live-evaluation.json").read_text())
    assert saved["results"][0]["judges"]["groundedness"]["score"] == 0


@pytest.mark.parametrize("repeats", [0, -1, 11])
def test_suite_rejects_invalid_repeat_budget(repeats):
    from eval_live import evaluate
    with pytest.raises(ValueError):
        evaluate(repeats=repeats)
