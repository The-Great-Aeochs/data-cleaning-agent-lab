"""Drive event streaming, verify the outcome, and persist classroom artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv
from langchain_core.messages import messages_to_dict

from agent import build_agent
from tools import TOOLS, RunContext
from tracing import TraceStore, redact
from workspace import Workspace

ROOT = Path(__file__).resolve().parent
DEFAULT_GOAL = "Prepare this orders CSV for revenue analysis. Fix safe issues, flag uncertain records, and validate the result."
TOOL_NAMES = {tool.name for tool in TOOLS}


def unpack_output(output):
    """Tool lifecycle projections return a ToolMessage or a plain value."""
    content = getattr(output, "content", output)
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return content
    return content


def run_agent(path: str | Path, goal: str = DEFAULT_GOAL, *, model=None,
              model_limit: int = 16, tool_limit: int = 14,
              runs_root: Path | None = None):
    """Yield the growing live trace; the last snapshot contains verified artifacts.

    The graph selects actions. This runner observes them and independently
    verifies completion; it never scripts a transformation sequence.
    """
    load_dotenv()
    workspace = Workspace(path)
    context = RunContext(workspace)
    store = TraceStore(runs_root or ROOT / "runs", goal,
                       os.getenv("OPENAI_MODEL", "gpt-4.1-mini") if model is None else "test-model")
    workspace.original.to_csv(store.directory / "original.csv", index=False)
    turns = calls = input_tokens = output_tokens = 0
    tool_sequence = []
    messages = []
    summary = None
    error = None
    store.add("start", "Run started", {"goal": goal, "rows": len(workspace.data)})
    yield snapshot(store, workspace, turns, calls, input_tokens, output_tokens)
    try:
        graph = build_agent(model=model, model_limit=model_limit, tool_limit=tool_limit)
        stream = graph.stream_events(
            {"messages": [{"role": "user", "content": goal}]},
            context=context, config={"recursion_limit": 64}, version="v3",
        )
        for kind, item in stream.interleave("messages", "tool_calls", "values"):
            if kind == "values":
                messages = messages_to_dict(item.get("messages", []))
                continue
            if kind == "messages":
                turns += 1
                store.add("model", f"Model turn {turns}", "Selecting the next action from the conversation.")
                yield snapshot(store, workspace, turns, calls, input_tokens, output_tokens)
                # Draining the message finalizes text, tool calls, and token usage.
                message = item.output
                text = getattr(message, "text", "")
                usage = message.usage_metadata or {}
                input_tokens += usage.get("input_tokens", 0)
                output_tokens += usage.get("output_tokens", 0)
                if text:
                    store.add("assistant", "Assistant explanation", text)
                messages.append(messages_to_dict([message])[0])
            elif kind == "tool_calls":
                if item.tool_name not in TOOL_NAMES:
                    # Structured output is a framework output tool, not a data tool.
                    _ = item.output
                    continue
                calls += 1
                tool_sequence.append(item.tool_name)
                started = perf_counter()
                store.add("action", item.tool_name, item.input,
                          tool_call_id=item.tool_call_id)
                yield snapshot(store, workspace, turns, calls, input_tokens, output_tokens)
                for _ in item.output_deltas:
                    pass  # drive execution until the tool lifecycle finishes
                observation = {"error": item.error} if item.error else unpack_output(item.output)
                store.add("observation", f"{item.tool_name} result", observation,
                          tool_call_id=item.tool_call_id,
                          duration_ms=round((perf_counter() - started) * 1000))
            yield snapshot(store, workspace, turns, calls, input_tokens, output_tokens)
        state = stream.output
        summary = state.get("structured_response")
        messages = messages_to_dict(state.get("messages", []))
    except Exception as exception:
        # Preserve partial progress and a diagnostic; never present it as success.
        error = {"type": type(exception).__name__, "message": redact(str(exception))}
        store.add("error", "Run stopped", error)

    agent_checked = workspace.last_checked_version == workspace.version
    quality = workspace.check()  # independent host verification after model termination
    if error:
        status = "limit_reached" if "Limit" in error["type"] or "limit" in error["message"].lower() else "error"
    elif not agent_checked or summary is None or not quality["passed"]:
        status = "unverified"
    elif not quality["ready_for_analysis"]:
        status = "blocked"
    elif quality["flagged_rows"]:
        status = "needs_review"
    else:
        status = "completed"
    # Only create a clean artifact after successful verification.
    artifacts = workspace.export(store.directory, include_clean=status in {"completed", "needs_review"})
    report = {"status": status, "agent_checked_final_version": agent_checked,
              "quality": quality, "changes": workspace.history,
              "summary": summary.model_dump() if summary is not None else None,
              "error": error, "tool_sequence": tool_sequence,
              "metrics": {"model_turns": turns, "tool_calls": calls,
                          "input_tokens": input_tokens, "output_tokens": output_tokens},
              "artifacts": artifacts}
    store.add("final", "Host-verified outcome", report)
    trace = store.finish(report, messages)
    final = snapshot(store, workspace, turns, calls, input_tokens, output_tokens)
    final.update(report=report, trace=trace, directory=str(store.directory))
    yield final


def snapshot(store, workspace, turns, calls, input_tokens, output_tokens) -> dict:
    """Package live UI state without exposing mutable workspace objects."""
    return {"run_id": store.run_id, "events": list(store.events),
            "original": workspace.original.head(30).to_dict("records"),
            "working": workspace.data.head(30).to_dict("records"),
            "metrics": {"model_turns": turns, "tool_calls": calls,
                        "input_tokens": input_tokens, "output_tokens": output_tokens}}


def main():
    """CLI entry point used by live evaluation and classroom demonstrations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="?", default=str(ROOT / "data" / "formatting.csv"))
    parser.add_argument("--goal", default=DEFAULT_GOAL)
    parser.add_argument("--model-limit", type=int, default=16)
    args = parser.parse_args()
    final = None
    for final in run_agent(args.csv, args.goal, model_limit=args.model_limit):
        pass
    print(json.dumps(final["report"], indent=2))
    print(f"Saved trace: {final['directory']}/run.json")


if __name__ == "__main__":
    main()
