"""Gradio classroom UI: live actions, observations, data changes, and replay."""

from __future__ import annotations

import html
import json
from pathlib import Path

import gradio as gr
import pandas as pd

from agent import CleaningSummary
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
        return workspace.original.head(30), f"{len(workspace.original)} source rows · first 30 shown"
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
    return sorted([p.parent.name for p in (ROOT / "runs").glob("*/run.json")], reverse=True)


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
                        source = gr.Dataframe(label="Original data · first 30 rows", interactive=False)
                        source_note = gr.Markdown()
                        working = gr.Dataframe(label="Working copy · first 30 rows", interactive=False)
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
        go.click(run_demo, [scenario, upload, goal, model_limit, tool_limit],
                 [timeline, metrics, working, report, files, events, stored], concurrency_limit=4)
    return demo


if __name__ == "__main__":
    build_demo().queue().launch(server_name="127.0.0.1", theme=gr.themes.Soft(primary_hue="teal"),
                                css=CSS, allowed_paths=[str(ROOT / "runs")])
