# Data Cleaning Agent Lab

One LangChain agent, **three tools**, and a Gradio classroom interface. Give it an orders CSV and a goal: repair safe issues, flag uncertain records, and validate the result.

The model chooses actions and arguments from tool feedback. Python controls permitted edits, preserves the input, and independently checks completion.

## Run locally

Python 3.12 is tested.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set OPENAI_API_KEY in .env. Optional: OPENAI_MODEL.
python app.py
```

Open http://127.0.0.1:7860. The default model is `gpt-4.1-mini`; it is configurable. API calls use your OpenAI account. Replay works without an API key.

`requirements.txt` is the single dependency list: pinned agent/UI libraries plus `pytest` and `ruff`. Their indirect dependencies are resolved by pip.

```bash
python runner.py data/formatting.csv
pytest -q
ruff check .
python eval_live.py --repeats 2
```

## What to showcase

| Scenario | Expected behavior |
|---|---|
| Clean data | Inspect, validate, finish; zero transformations |
| Formatting + duplicates | Choose needed normalizations and remove one exact copy |
| Conflicting records | Preserve and flag both conflicting orders and a missing quantity |
| Ambiguous values | Flag ambiguous dates, mismatched symbols, and negative amounts |
| Instructions inside cells | Treat malicious cell text as data; preserve records |
| No eligible records | Flag all uncertain rows and report blocked; create no clean artifact |
| Low call budget | Stop with an explicit limit outcome and retain partial evidence |

- **Execution loop:** model turns, selected tools, short action explanations, inputs, observations, and elapsed time.
- **Before / after:** source rows next to the working copy, including stable `_row_id` and `_flags` columns.
- **Verified outcome:** eligible/flagged/removed counts, amounts per currency, mutation history, final-check status, and downloadable artifacts.
- **Raw evidence:** actual tool schemas and event payloads; downloaded `messages.json` contains the model-visible conversation.
- **Replay:** load a saved run and reveal events with a slider; pause to predict the next action. Replay calls no model.

The UI displays observable execution and brief public explanations. It does not claim to expose private chain-of-thought. This is a tool-calling implementation of the ReAct feedback pattern.

## Three tools

| Tool | Purpose |
|---|---|
| `inspect_data(reason, row_ids)` | Observe samples, findings, and the current version |
| `transform_data(operation, expected_version, reason)` | Apply one supported operation to the working copy |
| `check_quality(reason)` | Validate eligible rows and report remaining failures |

Supported operations: trim text, normalize currencies, normalize amounts, normalize unambiguous dates, remove exact duplicates, and flag unresolved records. `ToolRuntime` supplies the private workspace; it is absent from model-visible schemas. No generated Python or arbitrary filesystem commands are executed. `ToolStrategy` adds a framework output tool for `CleaningSummary`; it is separate from the three data tools.

## Dataset contract

Required columns: `order_id`, `order_date`, `amount`, `quantity`, `currency`. Extra columns are preserved. CSVs must be UTF-8, at most 1 MB, and contain 1–2,000 rows and at most 30 columns. Model-visible samples clip long cells; exported data preserves their full contents. Duplicate headers and reserved `_` columns are rejected.

- `amount` is the **total order value**. Quantity is a positive integer; amounts are nonnegative with at most two decimals.
- Supported currencies: INR, USD, EUR, GBP, JPY. This classroom parser interprets `$` only with USD and `¥` only with JPY. Totals remain separate by currency.
- Safe dates: `YYYY-MM-DD`, `YYYY/MM/DD`, and dates with month names. Numeric day/month ambiguity is flagged.
- Exact duplicates must match every source column. Different records sharing an order ID are flagged together.
- Missing values are never imputed. All-flagged data is blocked, even if the empty eligible subset has no violations.

## Verified completion

`CleaningSummary` validates the model's response shape. The runner separately requires:

1. The agent called `check_quality` on the final dataset version.
2. Independent host validation passes.
3. A structured summary exists and no execution error occurred.

Statuses: `completed`, `needs_review`, `blocked`, `unverified`, `limit_reached`, `error`. Only `completed` and `needs_review` expose a clean download. Flagged records remain available for review. A failed run's working CSV is diagnostic output.

Version checks reject stale transformations, a per-workspace lock serializes edits, and each run gets an isolated copy. Model/tool budgets bound the loop. The OpenAI client has a timeout and one transport retry; input errors become model-visible observations.

## Stored evidence

```text
runs/<run_id>/
  original.csv      source copy with stable row IDs
  working.csv       current state
  clean.csv         eligible subset; created only for verified, analysis-ready runs
  flagged.csv       unresolved rows and issue codes
  events.jsonl      append-only execution events
  messages.json    conversation history
  report.json      verified outcome and metrics
  run.json         replay bundle
```

Runs, uploaded data, `.env`, and virtual environments are git-ignored. Persisted JSON redacts API-key-shaped strings. Traces and exports still contain dataset samples: use synthetic or appropriately approved data for class. The local server listens on loopback and does not publish a Gradio share link.

## Teaching map

The [lecture walkthrough](LECTURE_WALKTHROUGH.md) explains each file and function with source snippets.

| File | Responsibility |
|---|---|
| `workspace.py` | Data rules, permitted edits, validation, and exports |
| `tools.py` | Three tools, allowed operations, and private runtime context |
| `agent.py` | Model, prompt, output schema, and call limits |
| `runner.py` | Observe the framework loop and verify completion |
| `tracing.py` | Save evidence and load replay bundles |
| `app.py` | Gradio live lab and replay interface |
| `eval_live.py` | Check autonomous behavior against real OpenAI responses |
| `tests/`, `data/` | Offline checks and the six classroom scenarios |

## Research decisions

Reviewed 11 October 2026 against official documentation and installed APIs. The tested stack uses LangChain **1.4.4**.

| Source | Applied here |
|---|---|
| [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) | One model, three tools, feedback, explicit stopping conditions |
| [LangChain agents](https://docs.langchain.com/oss/python/langchain/agents) | Current `create_agent`; no legacy agent constructors |
| [LangChain tools](https://docs.langchain.com/oss/python/langchain/tools) | Typed signatures and private `ToolRuntime` context |
| [Event streaming](https://docs.langchain.com/oss/python/langchain/event-streaming) | `stream_events(version="v3")` with messages, tools, and state projections |
| [Structured output](https://docs.langchain.com/oss/python/langchain/structured-output) | `ToolStrategy(CleaningSummary)` plus independent host validation |
| [Middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in) | Call budgets and recoverable error observations |
| [GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini) | Tool-calling default model; configurable through `.env` |

Python defines allowed edits and quality rules; the model chooses actions, arguments, and ordering. The runner observes decisions rather than scripting the cleaning sequence. The v3 streaming API is experimental; its adapter is exercised by integration tests.

## Validation

- **60 automated tests passed** on Python 3.12.2; **Ruff passed**, including in a fresh environment installed from the single `requirements.txt`. Checks cover the real graph with a scripted model, stream payloads, structured completion, budget exhaustion, decimal precision, ambiguity, conflicts, source preservation, stale versions, uploads, replay boundaries, and final UI state.
- **18/18 live OpenAI checks passed** before this file consolidation: five scenarios repeated twice, six final-source scenarios, and two call-budget checks. Live evaluations used `gpt-4.1-mini` with the credential outside the repository.
- **2/2 live regression checks passed after consolidation** in the fresh environment: clean data used no transformations, and formatting completed its multi-edit loop with the expected total and duplicate count.
- Browser runs verified live cards, metrics, before/after tables, reports, and artifact links. Local-server checks verified CSV/JSON downloads and replay without model calls.

| Live scenario | Outcome | Eligible | Flagged | Exact copies removed | Data tool calls |
|---|---|---:|---:|---:|---:|
| Clean | completed | 3 | 0 | 0 | 2 |
| Formatting | completed | 3 | 0 | 1 | 7 |
| Conflicts | needs_review | 2 | 3 | 0 | 4 |
| Ambiguous | needs_review | 1 | 3 | 0 | 5 |
| Instructions in cells | completed | 2 | 0 | 0 | 2 |
| All flagged | blocked | 0 | 2 | 0 | 3 |

- All live scenarios passed status, counts, totals, final-check, unchanged-source, and observation assertions. Clean/instruction-bearing datasets used zero transformations; formatting preserved **INR 2,350.50**.
- Model/tool budgets of one produced `limit_reached`, retained partial evidence, and created no clean artifact. Evaluations check outcomes rather than requiring one action order.
- Run `python eval_live.py --repeats 2` to reproduce live checks; machine-readable results and full traces stay in git-ignored `runs/`.
- These results cover the bundled synthetic datasets and explicit edge cases. Other models, new domains, or dependency upgrades need fresh evaluation; they are not a guarantee for every CSV.
