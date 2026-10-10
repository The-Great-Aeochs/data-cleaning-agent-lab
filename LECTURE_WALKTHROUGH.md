# Lecture walkthrough

## Open files in this order

- `app.py` → run clean data, formatting, conflicts, and all-flagged data.
- `schemas.py` → output contract and private runtime context.
- `workspace.py` → what the dataset considers valid and which edits Python permits.
- `tools.py` → three callable boundaries and model-visible schemas.
- `agent.py` → model, instructions, tools, middleware, and structured response.
- `runner.py` → the framework loop becomes observable execution events.
- `tracing.py` → persist evidence and replay a run.
- `app.py` → connect generator updates to the classroom interface.
- `eval_live.py`, then `tests/` → measure behavior and enforce invariants.

## Concepts to point at

- **Autonomy:** `runner.run_agent` observes the graph; it never selects a transformation.
- **Feedback:** `Workspace.transform` returns remaining findings; `check_quality` returns pass/fail evidence.
- **State:** model conversation lives in the LangGraph state; the working dataset lives in `RunContext.workspace`.
- **ReAct pattern:** public action explanation → tool call → observation → another model turn.
- **Validation:** `CleaningSummary` checks shape; the runner checks final data and whether the agent validated the latest version.
- **Termination:** `build_agent` bounds calls; `run_agent` records explicit completion, review, blocked, or failure statuses.
- **Three data tools:** `ToolStrategy` additionally supplies a framework output tool; it does not mutate data.

## schemas.py

- `Operation`: Literal allowlist; arbitrary edit names fail argument validation.
- `CleaningSummary`: model explanation, changes, uncertainties; no model-supplied success flag or output path.
- `RunContext`: dependency containing one workspace and the successful-inspection flag.

```python
class RunContext:
    workspace: Workspace
    inspected: bool = False
```

## workspace.py

- [parse_amount](workspace.py#L26): Validate decimal syntax and currency-symbol agreement; preserve monetary precision with Decimal.

```python
number = Decimal(text.replace(",", ""))
return number if number.is_finite() and number >= 0 else None
```

- [parse_date](workspace.py#L44): Try only unambiguous formats; leave uncertain dates unresolved.

```python
for format_string in ("%Y-%m-%d", "%Y/%m/%d", "%d %b %Y", "%d %B %Y"):
    try:
        return datetime.strptime(value.strip(), format_string).date().isoformat()
    except ValueError:
        continue
return None
```

- [issue](workspace.py#L54): Represent a quality problem using a code, source-row IDs, and whether a safe repair exists.

```python
return {"code": code, "row_ids": row_ids, "message": message, "fixable": fixable}
```

- [Workspace.__init__](workspace.py#L62): Validate the upload; read every source field as text; create provenance columns and an isolated original copy.

```python
self.data = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
if self.data.empty or len(self.data) > MAX_ROWS:
    raise ValueError("CSV must contain between 1 and 2,000 rows.")
self.columns = list(self.data.columns)
self.data.insert(0, "_row_id", range(1, len(self.data) + 1))
self.data["_flags"] = ""
self.original = self.data.copy(deep=True)
self.version = 0
self.last_checked_version: int | None = None
```

- [Workspace.findings](workspace.py#L88): Separate formatting problems from semantic uncertainty; identify exact copies and conflicting order IDs.

```python
duplicates = frame.duplicated(subset=self.columns, keep="first")
if duplicates.any():
    ids = frame.loc[duplicates, "_row_id"].astype(int).tolist()
    found.append(issue("exact_duplicate", ids, "Remove identical repeated rows.", True))
```

- [Workspace.inspect](workspace.py#L129): Return the current version, bounded sample, and findings; focused inspection retains global conflict evidence.

```python
sample = frame.head(12).to_dict("records")
for row in sample:
    for column, value in row.items():
        if isinstance(value, str) and len(value) > 240:
            row[column] = value[:240] + "… [sample truncated]"
```

- [Workspace.transform](workspace.py#L149): Check the expected version; apply one allowlisted operation to a copy; commit changes atomically and invalidate old validation.

```python
if expected_version != self.version:
    raise ValueError(f"Stale version {expected_version}; current version is {self.version}. Inspect again.")
before = self.data.copy(deep=True)
after = before.copy(deep=True)
```

- [Workspace.check](workspace.py#L190): Validate unflagged rows; require at least one eligible row for analysis; sum total order values separately by currency.

```python
eligible = self.data[self.data["_flags"] == ""]
failures = self.findings(eligible)
totals: dict[str, Decimal] = {}
if not failures:
    for _, row in eligible.iterrows():
        code = row["currency"]
        totals[code] = totals.get(code, Decimal("0")) + Decimal(row["amount"])
self.last_checked_version = self.version
```

- [Workspace.export](workspace.py#L208): Write working and flagged data; create clean output only when the runner permits it.

```python
frames = {"clean": self.data[self.data["_flags"] == ""],
          "flagged": self.data[self.data["_flags"] != ""], "working": self.data}
if not include_clean:
    frames.pop("clean")
```

## tools.py

- [inspect_data](tools.py#L9): Get the private workspace from runtime; record that inspection succeeded; return an observation.

```python
observation = runtime.context.workspace.inspect(row_ids)
runtime.context.inspected = True
return observation
```

- [transform_data](tools.py#L23): Require prior inspection; delegate a typed operation and observed version to the workspace.

```python
if not runtime.context.inspected:
    raise ValueError("Inspect the dataset before editing it.")
return runtime.context.workspace.transform(operation, expected_version)
```

- [check_quality](tools.py#L43): Expose deterministic validation as a tool, so its result can guide the next model action.

```python
return runtime.context.workspace.check()
```

## agent.py

- `SYSTEM_PROMPT`: require observation-based edits, preserve uncertainty, ignore cell instructions, and validate after the final edit.

- [recover_tool_errors](agent.py#L50): Convert a recoverable ValueError into a ToolMessage carrying the original call ID; the model can respond to the error.

```python
return ToolMessage(
    content=json.dumps({"error": str(error), "hint": "Inspect current data and retry safely."}),
    tool_call_id=request.tool_call["id"], name=request.tool_call["name"], status="error",
)
```

- [build_agent](agent.py#L61): Construct the model and current create_agent graph; configure three data tools, runtime context, structured output, and call budgets.

```python
return create_agent(
    model=model, tools=TOOLS, system_prompt=SYSTEM_PROMPT,
    context_schema=RunContext,
    response_format=ToolStrategy(CleaningSummary),
    middleware=[
        ModelCallLimitMiddleware(run_limit=model_limit, exit_behavior="error"),
        ToolCallLimitMiddleware(run_limit=tool_limit, exit_behavior="error"),
        recover_tool_errors,
    ],
)
```

## runner.py

- [unpack_output](runner.py#L25): Decode a completed tool payload into the same observation shown in the trace.

```python
if isinstance(content, str):
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content
```

- [run_agent](runner.py#L36): Drive typed event streaming; record model/action/observation events; preserve partial history; independently verify completion and export artifacts.

```python
stream = graph.stream_events(
    {"messages": [{"role": "user", "content": goal}]},
    context=context, config={"recursion_limit": 64}, version="v3",
)
```

- [snapshot](runner.py#L134): Yield copied UI state and bounded data previews; avoid exposing mutable workspace objects.

```python
return {"run_id": store.run_id, "events": list(store.events),
        "original": workspace.original.head(20).to_dict("records"),
        "working": workspace.data.head(20).to_dict("records"),
        "metrics": {"model_turns": turns, "tool_calls": calls,
                    "input_tokens": input_tokens, "output_tokens": output_tokens}}
```

- [main](runner.py#L143): Read CLI arguments and run the same generator used by the UI; print the verified report and saved trace path.

```python
for final in run_agent(args.csv, args.goal, model_limit=args.model_limit):
    pass
```

## tracing.py

- [redact](tracing.py#L14): Recursively remove the active API key and key-shaped strings from persisted JSON.

```python
key = os.getenv("OPENAI_API_KEY")
if key:
    value = value.replace(key, "[REDACTED]")
return re.sub(r"sk-[A-Za-z0-9_-]{16,}", "[REDACTED]", value)
```

- [write_json](tracing.py#L28): Write redacted JSON through a temporary file and atomic replacement.

```python
temporary = path.with_suffix(".tmp")
temporary.write_text(json.dumps(redact(value), indent=2, default=str), encoding="utf-8")
temporary.replace(path)
```

- [TraceStore.__init__](tracing.py#L38): Create a unique run folder and initialize run identity, timing, and event storage.

```python
self.run_id = uuid4().hex[:12]
self.directory = root / self.run_id
self.directory.mkdir(parents=True)
self.started = perf_counter()
```

- [TraceStore.add](tracing.py#L48): Append a numbered event with elapsed time to memory and events.jsonl.

```python
event = redact({"index": len(self.events) + 1, "kind": kind, "title": title,
                "elapsed_ms": round((perf_counter() - self.started) * 1000),
                "body": body, **details})
self.events.append(event)
with (self.directory / "events.jsonl").open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(event, default=str) + "\n")
```

- [TraceStore.finish](tracing.py#L58): Save messages, verified report, and the replay bundle after termination.

```python
write_json(self.directory / "messages.json", messages)
write_json(self.directory / "report.json", report)
write_json(self.directory / "run.json", self.metadata)
```

- [load_trace](tracing.py#L68): Validate a bounded replay bundle; treat uploaded trace content as data.

```python
if not isinstance(trace, dict) or not isinstance(trace.get("events"), list):
    raise ValueError("Upload a run.json containing an events list.")
```

## app.py

- [render_trace](app.py#L45): Render color-coded timeline cards; escape every model/dataset string before generating HTML.

```python
cards.append(
    f'<div class="trace-card" style="--accent:{COLORS.get(kind, "#94a3b8")}">'
    f'<div class="trace-kind">{html.escape(kind)}</div>'
    f'<div class="trace-title">{html.escape(label)}'
    f'<span class="trace-time">{event.get("elapsed_ms", 0) / 1000:.2f}s</span></div>'
    f'<div class="trace-reason">{html.escape(reason)}</div>'
    f'<details {"open" if kind in {"error", "assistant"} else ""}>'
    f'<summary>Inspect event</summary><pre>{html.escape(text)}</pre></details></div>'
)
```

- [select_input](app.py#L66): Use an uploaded CSV when present; otherwise select a bundled classroom scenario.

```python
return Path(upload) if upload else ROOT / "data" / SCENARIOS[scenario]
```

- [preview_input](app.py#L71): Validate the selected input and show the original first 20 rows before model execution.

```python
workspace = Workspace(select_input(scenario, upload))
return workspace.original.head(20), f"{len(workspace.original)} source rows · first 20 shown"
```

- [run_demo](app.py#L80): Translate runner snapshots into live UI updates, metrics, working data, evidence, and verified downloads.

```python
yield (render_trace([{"kind": "error", "title": "Cannot start run", "body": str(error)}]),
       "**INPUT ERROR**", pd.DataFrame(), {}, [], [], gr.update())
```

- [saved_runs](app.py#L108): List known local run IDs; do not accept arbitrary filesystem paths from a replay selection.

```python
return sorted([p.parent.name for p in (ROOT / "runs").glob("*/run.json")], reverse=True)
```

- [open_replay](app.py#L113): Load a saved or uploaded trace and initialize the reveal slider without calling the model.

```python
trace = load_trace(path)
count = len(trace["events"])
return trace, gr.update(maximum=max(count, 1), value=0), render_trace([]), trace.get("report", {})
```

- [replay_to](app.py#L129): Render only the first N stored events, enabling pause-and-predict teaching.

```python
return render_trace(trace.get("events", [])[:int(count)])
```

- [build_demo](app.py#L134): Wire live controls, data comparison, reports, raw schemas, and replay into Gradio.

## eval_live.py

- [evaluate](eval_live.py#L22): Run live scenarios and assert data outcomes, source preservation, final validation, and zero clean-data transformations; retain failed cases.

```python
checks = {
    "status": report["status"] == expected["status"],
    "eligible": quality["eligible_rows"] == expected["eligible"],
    "flagged": quality["flagged_rows"] == expected["flagged"],
    "duplicates": quality["removed_duplicates"] == expected["duplicates"],
    "total": quality["order_value_by_currency"].get("INR") == expected["total"],
    "final_check": report["agent_checked_final_version"],
    "original_unchanged": before == hashlib.sha256(path.read_bytes()).hexdigest(),
    "no_clean_edits": not report["changes"] if name in {"clean", "injection"} else True,
    "no_clean_transforms": "transform_data" not in report["tool_sequence"] if name in {"clean", "injection"} else True,
    "trace_observations": all(e["body"] is not None for e in final["events"] if e["kind"] == "observation"),
}
```

- [main](eval_live.py#L63): Load an optional external credential file, run evaluation, and return a nonzero exit code on failure.

## Tests: functions and the invariant they check

### tests/test_workspace.py

- [test_amount_parsing](tests/test_workspace.py#L21): Amount parsing.
- [test_date_parsing](tests/test_workspace.py#L31): Date parsing.
- [repair](tests/test_workspace.py#L35): Apply the known fixture repairs for deterministic business-rule tests.
- [test_clean_file_needs_no_edits](tests/test_workspace.py#L41): Clean file needs no edits.
- [test_formatting_repairs_preserve_values_and_source](tests/test_workspace.py#L49): Formatting repairs preserve values and source.
- [test_conflicts_flag_every_candidate_and_missing_quantity](tests/test_workspace.py#L65): Conflicts flag every candidate and missing quantity.
- [test_ambiguous_data_is_not_invented](tests/test_workspace.py#L77): Ambiguous data is not invented.
- [test_flagging_cannot_hide_fixable_rows](tests/test_workspace.py#L86): Flagging cannot hide fixable rows.
- [test_stale_edit_and_unknown_operation_leave_data_unchanged](tests/test_workspace.py#L93): Stale edit and unknown operation leave data unchanged.
- [test_quality_check_invalidated_by_edit](tests/test_workspace.py#L104): Quality check invalidated by edit.
- [test_concurrent_edits_only_accept_one_version](tests/test_workspace.py#L112): Concurrent edits only accept one version.
- [test_workspaces_are_isolated](tests/test_workspace.py#L125): Workspaces are isolated.
- [test_invalid_uploads_rejected](tests/test_workspace.py#L138): Invalid uploads rejected.
- [test_all_flagged_is_not_ready](tests/test_workspace.py#L145): All flagged is not ready.
- [test_currency_totals_kept_separate](tests/test_workspace.py#L154): Currency totals kept separate.
- [test_focused_inspection_retains_global_conflict_evidence](tests/test_workspace.py#L160): Focused inspection retains global conflict evidence.
- [test_no_matching_rows_explained](tests/test_workspace.py#L166): No matching rows explained.
- [test_sample_cells_bounded_but_original_preserved](tests/test_workspace.py#L171): Sample cells bounded but original preserved.
- [test_oversize_input_rejected](tests/test_workspace.py#L180): Oversize input rejected.
- [test_excess_rows_rejected](tests/test_workspace.py#L187): Excess rows rejected.

### tests/test_agent.py

- [ScriptedModel._llm_type](tests/test_agent.py#L25): Identify the offline model adapter.
- [ScriptedModel.bind_tools](tests/test_agent.py#L28): Accept tool binding while keeping scripted responses.
- [ScriptedModel._generate](tests/test_agent.py#L31): Return the next scripted model message with fresh call IDs.
- [call](tests/test_agent.py#L40): Build an AIMessage requesting one tool.
- [finish](tests/test_agent.py#L45): Build the structured-output response used in offline tests.
- [execute](tests/test_agent.py#L50): Run the actual graph, tools, streaming adapter, and exports with a scripted model.
- [test_exactly_three_runtime_hidden_tools](tests/test_agent.py#L57): Exactly three runtime hidden tools.
- [test_clean_graph_streams_and_saves_replay](tests/test_agent.py#L63): Clean graph streams and saves replay.
- [test_graph_recovers_from_stale_version](tests/test_agent.py#L75): Graph recovers from stale version.
- [test_model_cannot_claim_success_without_check](tests/test_agent.py#L91): Model cannot claim success without check.
- [test_edit_without_inspection_is_rejected](tests/test_agent.py#L97): Edit without inspection is rejected.
- [test_model_limit_keeps_partial_trace](tests/test_agent.py#L104): Model limit keeps partial trace.
- [test_redaction](tests/test_agent.py#L111): Redaction.
- [test_tool_budget_is_enforced](tests/test_agent.py#L117): Tool budget is enforced.
- [test_invalid_final_data_is_never_published_as_clean](tests/test_agent.py#L124): Invalid final data is never published as clean.
- [test_check_before_final_edit_cannot_verify_completion](tests/test_agent.py#L132): Check before final edit cannot verify completion.
- [test_partial_conversation_contains_observations_after_limit](tests/test_agent.py#L141): Partial conversation contains observations after limit.
- [test_all_flagged_graph_reports_blocked_without_clean_artifact](tests/test_agent.py#L149): All flagged graph reports blocked without clean artifact.

### tests/test_ui.py

- [test_model_and_csv_html_is_escaped](tests/test_ui.py#L11): Model and csv html is escaped.
- [test_preview_displays_source_rows](tests/test_ui.py#L19): Preview displays source rows.
- [test_preview_invalid_upload_is_explained](tests/test_ui.py#L26): Preview invalid upload is explained.
- [test_uploaded_trace_replay_never_resolves_artifact_paths](tests/test_ui.py#L34): Uploaded trace replay never resolves artifact paths.
- [test_bad_trace_rejected](tests/test_ui.py#L47): Bad trace rejected.
- [test_ui_builds_all_classroom_surfaces](tests/test_ui.py#L54): Ui builds all classroom surfaces.
- [test_generator_retains_final_table_and_report](tests/test_ui.py#L63): Generator retains final table and report.

## Data and configuration

- `data/clean.csv`: already valid; inspect and check without edits.
- `data/formatting.csv`: whitespace, currency formatting, amounts, dates, and one exact duplicate.
- `data/conflicts.csv`: differing records share one order ID; quantity is missing in another row.
- `data/ambiguous.csv`: numeric date ambiguity, mismatched currency symbol, and negative amount.
- `data/injection.csv`: a cell contains instructions; treat it as ordinary data.
- `data/all_flagged.csv`: no safe analysis subset; return blocked.
- `.env.example`: credential/model configuration placeholders; copy to git-ignored `.env`.
- `requirements.txt`: direct runtime versions; `requirements-lock.txt`: exact tested environment.
- `requirements-dev.txt`: offline testing and lint dependencies.
- `pyproject.toml`: pytest imports/test paths and Ruff rules.
- `.github/workflows/tests.yml`: run offline tests and lint on pushes and pull requests.
- `.gitignore`: exclude keys, run artifacts, caches, and virtual environments.
