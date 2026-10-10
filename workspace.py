"""Own the working copy and enforce the orders dataset's business rules.

`amount` is the total order value, not a unit price. We never guess missing
values, reinterpret ambiguous numeric dates, or convert between currencies.
"""

from __future__ import annotations

import csv
import io
import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from threading import RLock

import pandas as pd

REQUIRED = ["order_id", "order_date", "amount", "quantity", "currency"]
CURRENCIES = {"INR", "USD", "EUR", "GBP", "JPY"}
SYMBOLS = {"₹": "INR", "$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}
MAX_BYTES = 1_000_000
MAX_ROWS = 2_000


def parse_amount(value: str, currency: str) -> Decimal | None:
    """Accept ordinary decimal amounts and correctly grouped thousands."""
    text = value.strip()
    for symbol, expected in SYMBOLS.items():
        if symbol in text:
            if currency.strip().upper() != expected or not text.startswith(symbol):
                return None
            text = text.removeprefix(symbol).strip()
    pattern = r"[+-]?(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d{1,2})?"
    if not re.fullmatch(pattern, text):
        return None
    try:
        number = Decimal(text.replace(",", ""))
        return number if number.is_finite() and number >= 0 else None
    except InvalidOperation:
        return None


def parse_date(value: str) -> str | None:
    """Only parse unambiguous formats; never infer DD/MM versus MM/DD."""
    for format_string in ("%Y-%m-%d", "%Y/%m/%d", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(value.strip(), format_string).date().isoformat()
        except ValueError:
            continue
    return None


def issue(code: str, row_ids: list[int], message: str, fixable: bool) -> dict:
    """Attach stable source-row IDs to a quality finding."""
    return {"code": code, "row_ids": row_ids, "message": message, "fixable": fixable}


class Workspace:
    """One isolated dataset; all mutation happens under a lock on a copy."""

    def __init__(self, path: str | Path):
        path = Path(path)
        if path.stat().st_size > MAX_BYTES:
            raise ValueError("CSV exceeds the 1 MB classroom limit.")
        text = path.read_text(encoding="utf-8-sig")
        header = next(csv.reader(io.StringIO(text)), [])
        if len(header) != len(set(header)):
            raise ValueError("CSV contains duplicate column names.")
        if any(name.startswith("_") for name in header):
            raise ValueError("Column names starting with '_' are reserved for provenance.")
        if not set(REQUIRED).issubset(header):
            raise ValueError(f"CSV must contain: {', '.join(REQUIRED)}.")
        self.data = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
        if self.data.empty or len(self.data) > MAX_ROWS:
            raise ValueError("CSV must contain between 1 and 2,000 rows.")
        self.columns = list(self.data.columns)
        self.data.insert(0, "_row_id", range(1, len(self.data) + 1))
        self.data["_flags"] = ""
        self.original = self.data.copy(deep=True)
        self.version = 0
        self.last_checked_version: int | None = None
        self.history: list[dict] = []
        self.lock = RLock()

    def findings(self, frame: pd.DataFrame | None = None) -> list[dict]:
        """Identify safe formatting repairs and unresolved semantic problems."""
        frame = self.data if frame is None else frame
        found = []
        for _, row in frame.iterrows():
            ids = [int(row["_row_id"])]
            currency = row["currency"].strip().upper()
            if any(row[c] != row[c].strip() for c in self.columns):
                found.append(issue("whitespace", ids, "Trim surrounding whitespace.", True))
            if not row["order_id"].strip():
                found.append(issue("missing_order_id", ids, "Order ID is missing.", False))
            if currency not in CURRENCIES:
                found.append(issue("invalid_currency", ids, "Unsupported/missing currency.", False))
            elif row["currency"] != currency:
                found.append(issue("currency_format", ids, "Use uppercase currency code.", True))
            amount = parse_amount(row["amount"], currency)
            if amount is None:
                found.append(issue("invalid_amount", ids, "Invalid amount or currency symbol.", False))
            elif row["amount"] != format(amount, ".2f"):
                found.append(issue("amount_format", ids, "Use a decimal amount without symbols.", True))
            date = parse_date(row["order_date"])
            if date is None:
                found.append(issue("invalid_date", ids, "Missing, invalid, or ambiguous date.", False))
            elif row["order_date"] != date:
                found.append(issue("date_format", ids, "Use YYYY-MM-DD.", True))
            if not re.fullmatch(r"[1-9]\d*", row["quantity"].strip()):
                found.append(issue("invalid_quantity", ids, "Quantity must be a positive integer.", False))
        duplicates = frame.duplicated(subset=self.columns, keep="first")
        if duplicates.any():
            ids = frame.loc[duplicates, "_row_id"].astype(int).tolist()
            found.append(issue("exact_duplicate", ids, "Remove identical repeated rows.", True))
        # Exact copies are handled above; differing rows with one ID are uncertain.
        unique = frame.drop_duplicates(subset=self.columns)
        ids = unique["order_id"].str.strip()
        conflicts = set(ids[ids.duplicated(keep=False)]) - {""}
        if conflicts:
            rows = frame.loc[frame["order_id"].str.strip().isin(conflicts), "_row_id"]
            found.append(issue("conflicting_order_id", rows.astype(int).tolist(),
                               "Same order ID has differing records; do not choose a winner.", False))
        return found

    def inspect(self, row_ids: list[int] | None = None) -> dict:
        """Return bounded samples, findings, and the current mutation version."""
        with self.lock:
            frame = self.data
            if row_ids:
                frame = frame[frame["_row_id"].isin(row_ids)]
            return {"version": self.version, "rows": len(self.data),
                    "columns": self.columns, "sample": frame.head(12).to_dict("records"),
                    "findings": self.findings(frame)[:80],
                    "finding_count": len(self.findings(frame))}

    def transform(self, operation: str, expected_version: int) -> dict:
        """Apply one allowlisted operation atomically; reject stale decisions."""
        with self.lock:
            if expected_version != self.version:
                raise ValueError(f"Stale version {expected_version}; current version is {self.version}. Inspect again.")
            before = self.data.copy(deep=True)
            after = before.copy(deep=True)
            if operation == "trim_text":
                for column in self.columns:
                    after[column] = after[column].str.strip()
            elif operation == "normalize_currency":
                after["currency"] = after["currency"].str.strip().str.upper()
            elif operation == "normalize_amounts":
                for index, row in after.iterrows():
                    value = parse_amount(row["amount"], row["currency"])
                    if value is not None:
                        after.at[index, "amount"] = format(value, ".2f")
            elif operation == "normalize_dates":
                after["order_date"] = after["order_date"].map(lambda v: parse_date(v) or v)
            elif operation == "remove_exact_duplicates":
                after = after.drop_duplicates(subset=self.columns, keep="first").copy()
            elif operation == "flag_unresolved":
                for finding in self.findings(after):
                    if not finding["fixable"]:
                        for row_id in finding["row_ids"]:
                            index = after.index[after["_row_id"] == row_id][0]
                            flags = set(filter(None, after.at[index, "_flags"].split(";")))
                            after.at[index, "_flags"] = ";".join(sorted(flags | {finding["code"]}))
            else:
                raise ValueError(f"Unsupported operation: {operation}.")
            changed = not before.equals(after)
            if changed:
                self.data = after.reset_index(drop=True)
                self.version += 1
                self.last_checked_version = None
                self.history.append({"operation": operation, "version": self.version,
                                     "rows_before": len(before), "rows_after": len(after)})
            return {"operation": operation, "changed": changed, "version": self.version,
                    "rows": len(self.data), "flagged": int(self.data["_flags"].ne("").sum()),
                    "remaining_findings": self.findings(self.data[self.data["_flags"] == ""])[:80]}

    def check(self) -> dict:
        """Verify eligible rows, and aggregate order amounts per currency."""
        with self.lock:
            eligible = self.data[self.data["_flags"] == ""]
            failures = self.findings(eligible)
            totals: dict[str, Decimal] = {}
            if not failures:
                for _, row in eligible.iterrows():
                    code = row["currency"]
                    totals[code] = totals.get(code, Decimal("0")) + Decimal(row["amount"])
            self.last_checked_version = self.version
            return {"version": self.version, "passed": not failures,
                    "ready_for_analysis": not failures and len(eligible) > 0,
                    "eligible_rows": len(eligible), "flagged_rows": len(self.data) - len(eligible),
                    "removed_duplicates": len(self.original) - len(self.data),
                    "failures": failures[:80], "failure_count": len(failures),
                    "order_value_by_currency": {c: str(v) for c, v in sorted(totals.items())}}

    def export(self, directory: Path) -> dict[str, str]:
        """Write three artifacts without modifying the source CSV."""
        with self.lock:
            directory.mkdir(parents=True, exist_ok=True)
            frames = {"clean": self.data[self.data["_flags"] == ""],
                      "flagged": self.data[self.data["_flags"] != ""], "working": self.data}
            paths = {}
            for name, frame in frames.items():
                path = directory / f"{name}.csv"
                frame.to_csv(path, index=False)
                paths[name] = str(path.resolve())
            return paths
