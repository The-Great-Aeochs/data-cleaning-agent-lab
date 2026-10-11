"""UI rendering, replay, and uploaded-data boundaries."""

import json

import pytest

from app import SCENARIOS, build_demo, open_replay, preview_input, render_trace, replay_to
from tracing import load_trace


def test_model_and_csv_html_is_escaped():
    markup = render_trace([{"kind": "action", "title": "<script>bad</script>",
                            "body": {"reason": "<img src=x onerror=alert(1)>"}}])
    assert "<script>" not in markup
    assert "<img" not in markup
    assert "&lt;script&gt;" in markup


def test_preview_displays_source_rows():
    frame, note = preview_input(list(SCENARIOS)[1], None)
    assert len(frame) == 27
    assert frame.loc[0, "order_id"] == " A001 "
    assert "27 source rows" in note


def test_preview_invalid_upload_is_explained(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("bad\nx\n")
    frame, note = preview_input(list(SCENARIOS)[0], str(path))
    assert frame.empty
    assert "Input error" in note


def test_uploaded_trace_replay_never_resolves_artifact_paths(tmp_path):
    path = tmp_path / "run.json"
    trace = {"events": [{"kind": "action", "title": "Inspect", "body": "data"}],
             "report": {"artifacts": {"clean": "/etc/passwd"}}}
    path.write_text(json.dumps(trace))
    state, slider, timeline, report = open_replay(None, str(path))
    assert len(state["events"]) == 1
    assert slider["maximum"] == 1
    assert "Inspect" not in timeline
    assert "Inspect" in replay_to(state, 1)


@pytest.mark.parametrize("value", [[], {}, {"events": [1]}, {"events": [{"kind": "action"}]}])
def test_bad_trace_rejected(tmp_path, value):
    path = tmp_path / "run.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        load_trace(path)


def test_ui_builds_all_classroom_surfaces():
    demo = build_demo()
    config = demo.config
    labels = json.dumps(config, default=str)
    for expected in ("Execution loop", "Before / after", "Verified outcome", "Replay a saved run"):
        assert expected in labels
    assert config["dependencies"]


def test_generator_retains_final_table_and_report(monkeypatch, tmp_path):
    import pandas as pd

    import app
    state = {"run_id": "test", "events": [], "working": [{"order_id": "A001"}],
             "metrics": {"model_turns": 3, "tool_calls": 2, "input_tokens": 10, "output_tokens": 2},
             "directory": str(tmp_path),
             "report": {"status": "completed", "artifacts": {},
                        "quality": {"eligible_rows": 1, "flagged_rows": 0, "removed_duplicates": 0}}}
    monkeypatch.setattr(app, "run_agent", lambda *args, **kwargs: iter([state]))
    updates = list(app.run_demo(list(SCENARIOS)[0], None, "Clean data", 16, 14))
    assert isinstance(updates[-1][2], pd.DataFrame)
    assert updates[-1][2].loc[0, "order_id"] == "A001"
    assert updates[-1][3]["status"] == "completed"
