"""Gradio classroom UI: live actions, observations, data changes, and replay."""

from __future__ import annotations

import html
import json
from pathlib import Path

import gradio as gr
import pandas as pd

from agent import CleaningSummary
from eval_live import evaluate_steps
from judges import RUBRICS, evaluate_trace
from runner import DEFAULT_GOAL, ROOT, run_agent
from tools import TOOLS
from tracing import load_trace
from workspace import Workspace

SCENARIOS = {
    "Clean orders": "clean.csv",
    "Formatting and duplicates": "formatting.csv",
    "Conflicting orders": "conflicts.csv",
    "Invalid values": "ambiguous.csv",
    "Notes with instructions": "injection.csv",
    "All rows need review": "all_flagged.csv",
}
COLORS = {"model": "#38bdf8", "action": "#fbbf24", "observation": "#34d399",
          "assistant": "#a3b8cc", "error": "#fb7185", "final": "#38bdf8", "start": "#94a3b8"}
CSS = """
.gradio-container {max-width: 1500px !important;}
#heading h1 {font-size: 2.1rem; letter-spacing: -.035em;}
.loop-map {display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin:12px 0 20px;
  padding:16px; background:#101c2c; border:1px solid #2d435b; border-radius:12px; color:#e2edf8;}
.loop-map b {color:#7dd3fc;} .loop-map span {color:#90a8bf;}
.trace-card {border:1px solid #2d435b; border-left:4px solid var(--accent);
  background:#101c2c; border-radius:10px; padding:13px 16px; margin-bottom:10px; color:#e2edf8;}
.trace-title {display:flex; justify-content:space-between; gap:12px; font-weight:600;}
.trace-kind {font-size:11px; letter-spacing:.09em; text-transform:uppercase; color:var(--accent);}
.trace-time {font-size:12px; color:#90a8bf; font-weight:400;}
.trace-card pre {white-space:pre-wrap; word-break:break-word; font-size:12px; margin-top:10px;}
.trace-card details {margin-top:8px;} .trace-card summary {cursor:pointer; color:#90a8bf;}
.trace-reason {margin-top:8px; font-size:14px;}
"""


def render_trace(events: list[dict]) -> str:
    """Escape every model/data string before rendering the event timeline."""
    cards = []
    for event in events:
        kind = event["kind"]
        body = event.get("body")
        text = body if isinstance(body, str) else json.dumps(body, indent=2, ensure_ascii=False)
        reason = body.get("reason", "") if kind == "action" and isinstance(body, dict) else ""
        label = f"{event.get('index', '')} · {event['title']}"
        cards.append(
            f'<div class="trace-card" style="--accent:{COLORS.get(kind, "#94a3b8")}">'
            f'<div class="trace-kind">{html.escape(kind)}</div>'
            f'<div class="trace-title">{html.escape(label)}'
            f'<span class="trace-time">{event.get("elapsed_ms", 0) / 1000:.2f}s</span></div>'
            f'<div class="trace-reason">{html.escape(reason)}</div>'
            f'<details {"open" if kind in {"error", "assistant"} else ""}>'
            f'<summary>Details</summary><pre>{html.escape(text)}</pre></details></div>'
        )
    return "".join(cards) or "<p>Run the agent to see tool calls and results.</p>"


def select_input(scenario: str, upload: str | None) -> Path:
    """Uploaded CSV takes precedence; otherwise choose a bundled scenario."""
    return Path(upload) if upload else ROOT / "data" / SCENARIOS[scenario]


def preview_input(scenario, upload):
    """Show the actual source data before starting the agent."""
    try:
        workspace = Workspace(select_input(scenario, upload))
        total = len(workspace.original)
        shown = "all rows shown" if total <= 30 else "first 30 rows shown"
        return workspace.original.head(30), f"{total} source rows · {shown}"
    except Exception as error:
        return pd.DataFrame(), f"Input error: {error}"


def run_demo(scenario, upload, goal, model_limit, tool_limit):
    """Yield UI updates at model starts, actions, observations, and completion."""
    try:
        for state in run_agent(select_input(scenario, upload), goal or DEFAULT_GOAL,
                               model_limit=int(model_limit), tool_limit=int(tool_limit)):
            report = state.get("report", {})
            metrics = state["metrics"]
            status = report.get("status", "running").replace("_", " ")
            quality = report.get("quality", {})
            heading = (f"**{status.upper()}** · run `{state['run_id']}`  \n"
                       f"Model turns **{metrics['model_turns']}** · data tool calls **{metrics['tool_calls']}** · "
                       f"tokens **{metrics['input_tokens'] + metrics['output_tokens']}**")
            if quality:
                heading += (f"  \nEligible **{quality['eligible_rows']}** · flagged **{quality['flagged_rows']}** · "
                            f"duplicates removed **{quality['removed_duplicates']}**")
            files = []
            if report:
                directory = Path(state["directory"])
                files = list(report["artifacts"].values()) + [str(directory / name) for name in
                        ("report.json", "run.json", "events.jsonl", "messages.json")]
            yield (render_trace(state["events"]), heading, pd.DataFrame(state["working"]),
                   report, files, state["events"],
                   gr.update(choices=saved_runs()) if report else gr.skip())
    except Exception as error:
        yield (render_trace([{"kind": "error", "title": "Cannot start run", "body": str(error)}]),
               "**INPUT ERROR**", pd.DataFrame(), {}, [], [], gr.update())


def saved_runs() -> list[str]:
    """List local run IDs only; replay does not trust paths in uploaded traces."""
    paths = sorted((ROOT / "runs").glob("*/run.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.parent.name for p in paths]


def open_replay(run_id, upload):
    """Load a saved or uploaded trace into browser-session state."""
    try:
        if upload:
            path = Path(upload)
        elif run_id in saved_runs():
            path = ROOT / "runs" / run_id / "run.json"
        else:
            raise ValueError("Select a saved run or upload run.json.")
        trace = load_trace(path)
        count = len(trace["events"])
        return trace, gr.update(maximum=max(count, 1), value=0), render_trace([]), trace.get("report", {})
    except Exception as error:
        return {}, gr.update(maximum=1, value=0), html.escape(str(error)), {}


def replay_to(trace, count):
    """Reveal the first N events without making any API calls."""
    return render_trace(trace.get("events", [])[:int(count)])


def judge_view(evaluation):
    """Binary scores stay separate; transport errors never become a zero."""
    rows = []
    for name, result in evaluation.get("judges", {}).items():
        rows.append([name.replace("_", " ").capitalize(), result.get("score"),
                     result.get("status"), result.get("explanation") or
                     result.get("error", {}).get("message", "Not evaluated")])
    return rows, evaluation, evaluation.get("artifact")


def load_evaluation(run_id):
    """Read the latest local judge attempt without spending API tokens."""
    if run_id not in saved_runs():
        return [], {}, None
    paths = sorted((ROOT / "runs" / run_id).glob("evaluation-*.json"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return judge_view(json.loads(paths[0].read_text())) if paths else ([], {}, None)


def evaluate_saved_run(run_id):
    """Grade the stored final answer against the actual saved evidence."""
    if run_id not in saved_runs():
        raise gr.Error("Select a saved run first.")
    directory = ROOT / "runs" / run_id
    return judge_view(evaluate_trace(load_trace(directory / "run.json"), directory))


def suite_view(summary):
    """Show exact checks beside both judge scores, including unscored errors."""
    rows = []
    for row in summary.get("results", []):
        scores = [row.get("judges", {}).get(key, {}).get("score") for key in RUBRICS]
        rows.append([row["case"], row.get("repeat", 1), int(row["passed"]), *scores,
                     row["status"], row["run_id"]])
    total = summary.get("total", 0)
    requested = summary.get("requested", total)
    note = (f"{'Finished' if summary.get('complete', True) else 'Running'} · "
            f"{total}/{requested} cases · Python checks {summary.get('passed', 0)}/{total}")
    if summary.get("with_judges"):
        note += f" · Python + both judges {summary.get('overall_passed', 0)}/{total}"
    return rows, note, summary, summary.get("artifact")


def load_suite():
    """CLI results are available in the UI too; loading makes no model calls."""
    path = ROOT / "runs" / "live-evaluation.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    if not summary:
        return [], "No sample tests saved yet.", {}, None
    return suite_view(summary)


def run_suite():
    """Stream the same six-case evaluation used by eval_live.py."""
    for summary in evaluate_steps():
        yield suite_view(summary)


def build_demo():
    """Build the lecture interface; importing this module never launches a server."""
    with gr.Blocks(title="Order Data Cleaner") as demo:
        gr.Markdown("# Order Data Cleaner", elem_id="heading")
        gr.HTML('<div class="loop-map"><b>Model</b><span>→</span><b>Tool call</b>'
                '<span>→</span><b>Result</b><span>↺</span></div>')
        with gr.Tab("Run"):
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    scenario = gr.Dropdown(list(SCENARIOS), value=list(SCENARIOS)[1], label="Sample data")
                    upload = gr.File(label="Upload an orders CSV", file_types=[".csv"], type="filepath", height=110)
                    goal = gr.Textbox(value=DEFAULT_GOAL, label="Task", lines=4)
                    with gr.Accordion("Run limits", open=False):
                        model_limit = gr.Slider(1, 24, value=16, step=1, label="Max. model calls")
                        tool_limit = gr.Slider(1, 22, value=14, step=1, label="Max. tool calls")
                    go = gr.Button("Run agent", variant="primary")
                    metrics = gr.Markdown("Ready.")
                with gr.Column(scale=3):
                    with gr.Tab("Agent activity"):
                        gr.Markdown("Tool calls and results from this run.")
                        timeline = gr.HTML(render_trace([]))
                    with gr.Tab("Before / after"):
                        source = gr.Dataframe(label="Original data", interactive=False)
                        source_note = gr.Markdown()
                        working = gr.Dataframe(label="Working copy · up to 30 rows", interactive=False)
                    with gr.Tab("Results"):
                        report = gr.JSON(label="Validation report")
                        files = gr.Files(label="Download files")
                    with gr.Tab("Details"):
                        events = gr.JSON(label="Run events")
                        gr.JSON({t.name: t.tool_call_schema.model_json_schema() for t in TOOLS}, label="Tool definitions")
                        gr.JSON(CleaningSummary.model_json_schema(), label="Response format")
            scenario.change(preview_input, [scenario, upload], [source, source_note])
            upload.change(preview_input, [scenario, upload], [source, source_note])
            demo.load(preview_input, [scenario, upload], [source, source_note])
        with gr.Tab("Saved runs"):
            replay_state = gr.State({})
            stored = gr.Dropdown(saved_runs(), label="Saved run")
            replay_file = gr.File(label="Upload run.json", file_types=[".json"], type="filepath")
            load = gr.Button("Load trace")
            position = gr.Slider(0, 1, value=0, step=1, label="Events shown")
            replay_timeline = gr.HTML(render_trace([]))
            replay_report = gr.JSON(label="Saved report")
            load.click(open_replay, [stored, replay_file], [replay_state, position, replay_timeline, replay_report])
            position.change(replay_to, [replay_state, position], replay_timeline)
        with gr.Tab("Evaluations"):
            gr.Markdown("**1 = pass · 0 = fail.** Blank scores mean not evaluated or judge error. "
                        "Python checks verify the data; the two judges assess the final answer.")
            with gr.Tab("Saved answer"):
                judge_run = gr.Dropdown(saved_runs(), label="Run to evaluate")
                refresh = gr.Button("Refresh runs")
                assess = gr.Button("Evaluate answer", variant="primary")
                gr.Markdown("Makes two independent judge calls. Select a run to load existing scores for free.")
                judge_scores = gr.Dataframe(headers=["Criterion", "Score", "Status", "Explanation"],
                                             datatype=["str", "number", "str", "str"], interactive=False)
                with gr.Accordion("Judge evidence and rubrics", open=False):
                    judge_details = gr.JSON(label="Saved judge results")
                judge_file = gr.File(label="Download evaluation", interactive=False)
                refresh.click(lambda: gr.update(choices=saved_runs()), outputs=judge_run)
                judge_run.change(load_evaluation, judge_run, [judge_scores, judge_details, judge_file])
                assess.click(evaluate_saved_run, judge_run, [judge_scores, judge_details, judge_file],
                             concurrency_id="evaluation", concurrency_limit=1)
            with gr.Tab("Sample tests"):
                gr.Markdown("Runs all six CSVs through the agent, Python checks, and both judges. "
                            "This makes paid agent and judge calls; previous results load without API calls.")
                suite_go = gr.Button("Run sample tests", variant="primary")
                suite_load = gr.Button("Load latest results")
                suite_note = gr.Markdown("No evaluation loaded.")
                suite_scores = gr.Dataframe(
                    headers=["Case", "Repeat", "Python checks", "Answer relevancy", "Groundedness", "Status", "Run"],
                    datatype=["str", "number", "number", "number", "number", "str", "str"], interactive=False)
                with gr.Accordion("Checks and judge details", open=False):
                    suite_details = gr.JSON(label="Suite results")
                suite_file = gr.File(label="Download suite results", interactive=False)
                suite_outputs = [suite_scores, suite_note, suite_details, suite_file]
                suite_go.click(run_suite, outputs=suite_outputs, concurrency_id="evaluation", concurrency_limit=1)
                suite_load.click(load_suite, outputs=suite_outputs)
                demo.load(load_suite, outputs=suite_outputs)
        go.click(run_demo, [scenario, upload, goal, model_limit, tool_limit],
                 [timeline, metrics, working, report, files, events, stored], concurrency_limit=4)
    return demo


if __name__ == "__main__":
    build_demo().queue().launch(server_name="127.0.0.1", theme=gr.themes.Soft(primary_hue="teal"),
                                css=CSS, allowed_paths=[str(ROOT / "runs")])
