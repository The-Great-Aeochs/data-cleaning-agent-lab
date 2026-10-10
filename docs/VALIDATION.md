# Validation results

Validated locally on 11 October 2026 with Python 3.12.2 and the pinned dependencies in `requirements-lock.txt`. Live runs used OpenAI `gpt-4.1-mini` and an authorized credential stored outside the repository.

## Automated checks

- `pytest -q`: **60 passed**.
- `ruff check .`: **passed**.
- Tests exercise the real LangChain graph with a scripted model, including tool execution, streaming observations, structured completion, and partial history on budget exhaustion.
- Dataset checks cover decimal precision, currency separation, ambiguous dates, exact duplicates, conflicting IDs, source preservation, stale versions, upload limits, and preventing flags from hiding safely repairable issues.
- UI checks cover HTML escaping, malformed uploads, replay without dereferencing artifact paths, interface construction, and retaining the final working table and report.
- Two expected warnings identify LangChain's experimental v3 event stream. Versions are locked; the stream adapter has integration tests.

## Live OpenAI checks

**18/18 checks passed after strengthening the prompt:** ten repeated scenario checks, six checks against the final hardened source, and two budget checks. The earlier ten cover five scenarios twice; the final six add the all-flagged scenario. The machine-readable record is [live-evaluation.json](live-evaluation.json).

| Final-source scenario | Outcome | Eligible | Flagged | Exact copies removed | Data tool calls |
|---|---|---:|---:|---:|---:|
| Clean | completed | 3 | 0 | 0 | 2 |
| Formatting | completed | 3 | 0 | 1 | 7 |
| Conflicts | needs_review | 2 | 3 | 0 | 4 |
| Ambiguous | needs_review | 1 | 3 | 0 | 5 |
| Instructions in cells | completed | 2 | 0 | 0 | 2 |
| All flagged | blocked | 0 | 2 | 0 | 3 |

- Every scenario passed its expected status, counts, total, final-check, unchanged-source, and observation-payload assertions.
- Clean and instruction-bearing datasets required no transformation calls.
- Formatting preserved the correct INR total of **2,350.50** after removing an exact duplicate.
- Ambiguous values were retained for review; they were not guessed or silently discarded.
- Model-call limit of one and data-tool limit of one each produced `limit_reached`, retained evidence, and exposed no clean artifact.
- Evaluation asserts outcomes rather than one fixed action order; the model selects the sequence.

## Interface checks

- Browser runs verified clean and formatting scenarios, live event cards, metrics, source/working tables, outcome reports, and artifact links.
- A final-update bug that cleared the working table was fixed and covered by a regression test; a subsequent browser run retained the normalized three-row table.
- Local-server checks verified downloaded CSV/report/replay files and the replay endpoints without making model calls.
- Offline replay and uploaded-trace boundaries are also covered by automated tests.

## Reproduce

```bash
pip install -r requirements-lock.txt
pytest -q
ruff check .
# Set OPENAI_API_KEY in .env first.
python eval_live.py --repeats 2
python app.py
```

## Limits of this evidence

- These results cover the six bundled synthetic datasets and explicit edge cases; they do not prove perfect behavior for every CSV or model.
- Tool selection can vary between runs, and a different model may require different budgets.
- The current v3 streaming API is experimental. Revalidate the graph, stream adapter, and UI when upgrading dependencies.
- This lab's explicit order-data contract defines what a valid result means; new domains require new rules and evaluation fixtures.
