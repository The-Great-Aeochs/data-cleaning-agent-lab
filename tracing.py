"""A local trace recorder: JSONL events, raw messages, metrics, and replay."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4


def redact(value):
    """Remove API-key-shaped strings and the active key from persisted content."""
    if isinstance(value, str):
        key = os.getenv("OPENAI_API_KEY")
        if key:
            value = value.replace(key, "[REDACTED]")
        return re.sub(r"sk-[A-Za-z0-9_-]{16,}", "[REDACTED]", value)
    if isinstance(value, dict):
        return {key: redact(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    return value


def write_json(path: Path, value):
    """Persist UTF-8 JSON using a temporary file and atomic replacement."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(redact(value), indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


class TraceStore:
    """One folder per run; events stay available even if the model fails."""

    def __init__(self, root: Path, goal: str, model: str):
        self.run_id = uuid4().hex[:12]
        self.directory = root / self.run_id
        self.directory.mkdir(parents=True)
        self.started = perf_counter()
        self.events: list[dict] = []
        self.metadata = {"run_id": self.run_id, "goal": goal, "model": model,
                         "started_at": datetime.now(timezone.utc).isoformat(),
                         "status": "running", "events": self.events}

    def add(self, kind: str, title: str, body=None, **details) -> dict:
        """Record one event with sequence number and elapsed milliseconds."""
        event = redact({"index": len(self.events) + 1, "kind": kind, "title": title,
                        "elapsed_ms": round((perf_counter() - self.started) * 1000),
                        "body": body, **details})
        self.events.append(event)
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str) + "\n")
        return event

    def finish(self, report: dict, messages: list[dict]) -> dict:
        """Save the final report and model-visible conversation for replay."""
        self.metadata.update(status=report["status"], report=report,
                             duration_ms=round((perf_counter() - self.started) * 1000))
        write_json(self.directory / "messages.json", messages)
        write_json(self.directory / "report.json", report)
        write_json(self.directory / "run.json", self.metadata)
        return self.metadata


def load_trace(path: str | Path) -> dict:
    """Read an exported run.json for offline classroom replay."""
    path = Path(path)
    if path.stat().st_size > 10_000_000:
        raise ValueError("Trace exceeds the 10 MB replay limit.")
    trace = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(trace, dict) or not isinstance(trace.get("events"), list):
        raise ValueError("Upload a run.json containing an events list.")
    if len(trace["events"]) > 2000:
        raise ValueError("Too many replay events.")
    for event in trace["events"]:
        if not isinstance(event, dict) or not {"kind", "title", "body"}.issubset(event):
            raise ValueError("Malformed replay event.")
    return redact(trace)
