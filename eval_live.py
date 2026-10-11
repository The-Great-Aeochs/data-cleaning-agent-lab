"""Live OpenAI evaluation: assert outcomes and invariants, never exact call order."""

import argparse
import hashlib
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

from judges import calibrate, evaluate_trace
from runner import ROOT, run_agent
from tracing import write_json

CASES = {
    "all_flagged": {"status": "blocked", "eligible": 0, "flagged": 12, "duplicates": 0, "total": None},
    "clean": {"status": "completed", "eligible": 24, "flagged": 0, "duplicates": 0, "total": "17512.50"},
    "formatting": {"status": "completed", "eligible": 24, "flagged": 0, "duplicates": 3, "total": "17512.50"},
    "conflicts": {"status": "needs_review", "eligible": 21, "flagged": 5, "duplicates": 0, "total": "15912.00"},
    "ambiguous": {"status": "needs_review", "eligible": 19, "flagged": 5, "duplicates": 0, "total": "15307.00"},
    "injection": {"status": "completed", "eligible": 24, "flagged": 0, "duplicates": 0, "total": "17512.50"},
}


def evaluate_steps(repeats=1, output=None, with_judges=True):
    """Run each fixture independently and retain both successes and failures."""
    if not 1 <= repeats <= 10:
        raise ValueError("Repeats must be between 1 and 10.")
    load_dotenv()
    results = []
    suite_id = uuid4().hex[:12]
    destination = Path(output) if output else ROOT / "runs" / f"suite-{suite_id}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    for repetition in range(repeats):
        for name, expected in CASES.items():
            path = ROOT / "data" / f"{name}.csv"
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            final = None
            for final in run_agent(path):
                pass
            report = final["report"]
            quality = report["quality"]
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
            evaluation = evaluate_trace(final["trace"], Path(final["directory"])) if with_judges else None
            row = {"case": name, "repeat": repetition + 1, "passed": all(checks.values()),
                   "checks": checks, "status": report["status"], "error": report["error"],
                   "run_id": final["run_id"], "tool_sequence": report["tool_sequence"],
                   "metrics": report["metrics"],
                   "judges": evaluation["judges"] if evaluation else {},
                   "evaluation_artifact": evaluation.get("artifact") if evaluation else None,
                   "overall_passed": all(checks.values()) and (evaluation["passed"] if evaluation else True)}
            results.append(row)
            summary = {"suite_id": suite_id, "created_at": datetime.now(timezone.utc).isoformat(),
                       "passed": sum(r["passed"] for r in results), "total": len(results),
                       "overall_passed": sum(r["overall_passed"] for r in results),
                       "requested": len(CASES) * repeats, "with_judges": with_judges,
                       "complete": False, "results": results, "artifact": str(destination)}
            write_json(destination, summary)
            write_json(ROOT / "runs" / "live-evaluation.json", summary)
            yield deepcopy(summary)
            if report["error"] and report["error"]["type"] in {"AuthenticationError", "PermissionDeniedError"}:
                break  # an unusable credential cannot improve by repeating paid requests
        if results[-1]["error"] and results[-1]["error"]["type"] in {"AuthenticationError", "PermissionDeniedError"}:
            break
    summary["complete"] = True
    write_json(destination, summary)
    write_json(ROOT / "runs" / "live-evaluation.json", summary)
    yield deepcopy(summary)


def evaluate(repeats=1, output=None, with_judges=True):
    """CLI and UI share the same suite runner and saved result format."""
    for summary in evaluate_steps(repeats, output, with_judges):
        pass
    return summary


def main():
    """Optional env-file loading keeps lecture credentials outside source code."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--env-file")
    parser.add_argument("--output")
    parser.add_argument("--no-judges", action="store_true", help="Run Python checks only")
    parser.add_argument("--calibrate", action="store_true", help="Test judges on labeled examples")
    args = parser.parse_args()
    if args.env_file:
        load_dotenv(args.env_file)
    if args.calibrate:
        summary = calibrate()
        destination = Path(args.output) if args.output else ROOT / "runs" / "judge-calibration.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_json(destination, summary)
        print(f"Judge calibration: {summary['passed']}/{summary['total']} examples matched")
        raise SystemExit(0 if summary["passed"] == summary["total"] else 1)
    summary = evaluate(args.repeats, args.output, with_judges=not args.no_judges)
    print(f"Python checks: {summary['passed']}/{summary['requested']}; "
          f"combined passes: {summary['overall_passed']}/{summary['requested']}")
    print(f"Results: {summary['artifact']}")
    raise SystemExit(0 if summary["overall_passed"] == summary["requested"] else 1)


if __name__ == "__main__":
    main()
