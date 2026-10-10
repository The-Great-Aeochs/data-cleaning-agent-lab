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
    assert len(frame) == 4
    assert frame.loc[0, "order_id"] == " A001 "
    assert "4 source rows" in note


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
