"""Business-rule tests: expected outputs are independent of the model."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

from workspace import Workspace, parse_amount, parse_date

DATA = Path(__file__).resolve().parents[1] / "data"


@pytest.mark.parametrize("value,currency,expected", [
    ("₹1,200", "INR", "1200"), ("$12.50", "USD", "12.50"),
    ("0", "EUR", "0"), ("1,234,567.89", "INR", "1234567.89"),
    ("$300", "INR", None), ("1,20", "INR", None), ("NaN", "USD", None),
    ("inf", "USD", None), ("-50", "INR", None), ("", "INR", None),
    ("1e9", "INR", None), ("123.456", "INR", None),
])
def test_amount_parsing(value, currency, expected):
    amount = parse_amount(value, currency)
    assert (str(amount) if amount is not None else None) == expected


@pytest.mark.parametrize("value,expected", [
    ("2026/09/01", "2026-09-01"), ("02 Sep 2026", "2026-09-02"),
    ("2 September 2026", "2026-09-02"), ("03/04/2026", None),
    ("2026-02-30", None), ("", None),
])
def test_date_parsing(value, expected):
    assert parse_date(value) == expected


def repair(workspace):
    for operation in ("trim_text", "normalize_currency", "normalize_amounts",
                      "normalize_dates", "remove_exact_duplicates", "flag_unresolved"):
        workspace.transform(operation, workspace.version)


def test_clean_file_needs_no_edits():
    workspace = Workspace(DATA / "clean.csv")
    report = workspace.check()
    assert report["ready_for_analysis"]
    assert report["order_value_by_currency"] == {"INR": "2350.50"}
    assert workspace.version == 0


def test_formatting_repairs_preserve_values_and_source(tmp_path):
    source = DATA / "formatting.csv"
    original_bytes = source.read_bytes()
    workspace = Workspace(source)
    repair(workspace)
    report = workspace.check()
    assert report["ready_for_analysis"]
    assert report["eligible_rows"] == 3
    assert report["removed_duplicates"] == 1
    assert report["flagged_rows"] == 0
    assert report["order_value_by_currency"] == {"INR": "2350.50"}
    paths = workspace.export(tmp_path)
    assert pd.read_csv(paths["clean"]).shape[0] == 3
    assert source.read_bytes() == original_bytes


def test_conflicts_flag_every_candidate_and_missing_quantity():
    workspace = Workspace(DATA / "conflicts.csv")
    repair(workspace)
    report = workspace.check()
    assert report["eligible_rows"] == 2
    assert report["flagged_rows"] == 3
    assert report["order_value_by_currency"] == {"INR": "1600.00"}
    flags = workspace.data.set_index("_row_id")["_flags"].to_dict()
    assert "conflicting_order_id" in flags[2] and "conflicting_order_id" in flags[3]
    assert flags[4] == "invalid_quantity"


def test_ambiguous_data_is_not_invented():
    workspace = Workspace(DATA / "ambiguous.csv")
    repair(workspace)
    assert workspace.check()["flagged_rows"] == 3
    assert workspace.data.loc[1, "order_date"] == "03/04/2026"
    assert workspace.data.loc[2, "amount"] == "$300"
    assert workspace.data.loc[3, "amount"] == "-50"


def test_flagging_cannot_hide_fixable_rows():
    workspace = Workspace(DATA / "formatting.csv")
    workspace.transform("flag_unresolved", 0)
    assert not workspace.check()["passed"]
    assert workspace.data["_flags"].eq("").all()


def test_stale_edit_and_unknown_operation_leave_data_unchanged():
    workspace = Workspace(DATA / "formatting.csv")
    workspace.transform("trim_text", 0)
    before = workspace.data.copy(deep=True)
    with pytest.raises(ValueError, match="Stale"):
        workspace.transform("normalize_amounts", 0)
    with pytest.raises(ValueError, match="Unsupported"):
        workspace.transform("delete_all", 1)
    pd.testing.assert_frame_equal(before, workspace.data)


def test_quality_check_invalidated_by_edit():
    workspace = Workspace(DATA / "formatting.csv")
    workspace.check()
    assert workspace.last_checked_version == 0
    workspace.transform("trim_text", 0)
    assert workspace.last_checked_version is None


def test_concurrent_edits_only_accept_one_version():
    workspace = Workspace(DATA / "formatting.csv")
    def attempt(operation):
        try:
            return workspace.transform(operation, 0)["changed"]
        except ValueError:
            return False
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(attempt, ["trim_text", "normalize_amounts"]))
    assert sum(results) == 1
    assert workspace.version == 1


def test_workspaces_are_isolated():
    first, second = Workspace(DATA / "formatting.csv"), Workspace(DATA / "formatting.csv")
    first.transform("trim_text", 0)
    assert second.version == 0
    assert second.data.loc[0, "order_id"] == " A001 "


@pytest.mark.parametrize("csv_text", [
    "order_id,amount\nA,1\n",
    "order_id,order_date,amount,quantity,currency\n",
    "order_id,order_date,amount,quantity,currency,amount\nA,2026-09-01,1,1,INR,2\n",
    "order_id,order_date,amount,quantity,currency,_flags\nA,2026-09-01,1,1,INR,x\n",
])
def test_invalid_uploads_rejected(tmp_path, csv_text):
    path = tmp_path / "bad.csv"
    path.write_text(csv_text)
    with pytest.raises(ValueError):
        Workspace(path)


def test_all_flagged_is_not_ready(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("order_id,order_date,amount,quantity,currency\nA,2026-09-01,1.00,,INR\n")
    workspace = Workspace(path)
    workspace.transform("flag_unresolved", 0)
    assert workspace.check()["passed"]
    assert not workspace.check()["ready_for_analysis"]


def test_currency_totals_kept_separate(tmp_path):
    path = tmp_path / "mixed.csv"
    path.write_text("order_id,order_date,amount,quantity,currency\nA,2026-09-01,1.00,1,INR\nB,2026-09-01,2.00,1,USD\n")
    assert Workspace(path).check()["order_value_by_currency"] == {"INR": "1.00", "USD": "2.00"}
