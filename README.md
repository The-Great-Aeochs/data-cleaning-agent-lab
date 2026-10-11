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
| Formatting + duplicates | Choose needed normalizations and remove three exact copies |
| Conflicting records | Preserve and flag two conflicting order pairs and a missing quantity |
| Ambiguous values | Flag ambiguous dates, mismatched symbols, and negative amounts |
| Instructions inside cells | Treat malicious cell text as data; preserve records |
| No eligible records | Flag all uncertain rows and report blocked; create no clean artifact |
| Low call budget | Stop with an explicit limit outcome and retain partial evidence |

- **Agent activity:** model turns, selected tools, short action explanations, inputs, observations, and elapsed time.
- **Before / after:** source rows next to the working copy, including stable `_row_id` and `_flags` columns.
- **Results:** eligible/flagged/removed counts, amounts per currency, mutation history, final-check status, and downloadable artifacts.
- **Details:** actual tool schemas and event payloads; downloaded `messages.json` contains the model-visible conversation.
- **Saved runs:** load a saved run and reveal events with a slider; pause to predict the next action. Replay calls no model.

The UI displays observable execution and brief public explanations. It does not claim to expose private chain-of-thought. This is a tool-calling implementation of the ReAct feedback pattern.

## Demo data

The CSVs are synthetic September 2026 exports for a fictional Indian stationery shop. Each row represents one complete order; `amount` is its total value in INR, and `quantity` is the number of items. The teaching question is: **Which order value can safely be included in revenue analysis, and which records need review?**

Each dataset has ten columns: the five required fields plus `product_category`, `sales_channel`, `region`, `customer_segment`, and `note`. Categories include study kits, notebooks, desk accessories, and art supplies. Context columns are preserved; no real customer data is used.

| Scenario | Source rows | Eligible | Flagged | Copies removed | Eligible INR total |
|---|---:|---:|---:|---:|---:|
| Clean | 24 | 24 | 0 | 0 | 17,512.50 |
| Formatting | 27 | 24 | 0 | 3 | 17,512.50 |
| Conflicts | 26 | 21 | 5 | 0 | 15,912.00 |
| Ambiguous | 24 | 19 | 5 | 0 | 15,307.00 |
| Instructions in cells | 24 | 24 | 0 | 0 | 17,512.50 |
| All flagged | 12 | 0 | 12 | 0 | No eligible total |

The model receives up to 12 sample rows, with findings computed over the entire dataset. The UI previews the first 30 rows, covering every bundled scenario; CSV downloads retain all rows. Some issues are deliberately placed beyond the model sample to demonstrate why observations include global findings.

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

- **62 automated tests passed** on Python 3.12.2; **Ruff passed**. Tests cover graph execution, streaming, completion, budgets, data rules, source preservation, context-column preservation, findings outside the model sample, and final UI state.
- **6/6 live OpenAI checks passed on the expanded datasets** with `gpt-4.1-mini`. Checks assert status, counts, totals, unchanged source, final validation, and recorded observations. Clean and injection scenarios used zero transformations. Explicit prompt examples were added after a prior run made unnecessary no-op cleanup calls.
- Earlier versions passed 18 live checks plus two consolidation regressions. Their small-dataset totals differ from the current scenarios.
- Model/tool budgets of one previously produced `limit_reached`, retained partial evidence, and created no clean artifact. Evaluations check outcomes rather than requiring one action order.
- Browser and local-server checks verified live cards, before/after tables, reports, downloads, and offline replay.
- Run `python eval_live.py --repeats 2` to reproduce live checks; machine-readable results and full traces stay in git-ignored `runs/`.
- These results cover the bundled synthetic datasets and explicit edge cases. Other models, new domains, or dependency upgrades need fresh evaluation; they are not a guarantee for every CSV.
