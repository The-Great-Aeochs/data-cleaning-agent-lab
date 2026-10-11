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
    assert report["order_value_by_currency"] == {"INR": "17512.50"}
    assert workspace.version == 0


def test_formatting_repairs_preserve_values_and_source(tmp_path):
    source = DATA / "formatting.csv"
    original_bytes = source.read_bytes()
    workspace = Workspace(source)
    repair(workspace)
    report = workspace.check()
    assert report["ready_for_analysis"]
    assert report["eligible_rows"] == 24
    assert report["removed_duplicates"] == 3
    assert report["flagged_rows"] == 0
    assert report["order_value_by_currency"] == {"INR": "17512.50"}
    paths = workspace.export(tmp_path)
    assert pd.read_csv(paths["clean"]).shape[0] == 24
    assert source.read_bytes() == original_bytes


def test_conflicts_flag_every_candidate_and_missing_quantity():
    workspace = Workspace(DATA / "conflicts.csv")
    repair(workspace)
    report = workspace.check()
    assert report["eligible_rows"] == 21
    assert report["flagged_rows"] == 5
    assert report["order_value_by_currency"] == {"INR": "15912.00"}
    flags = workspace.data.set_index("_row_id")["_flags"].to_dict()
    assert all("conflicting_order_id" in flags[row] for row in (2, 16, 25, 26))
    assert flags[3] == "invalid_quantity"


def test_ambiguous_data_is_not_invented():
    workspace = Workspace(DATA / "ambiguous.csv")
    repair(workspace)
    assert workspace.check()["flagged_rows"] == 5
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


def test_focused_inspection_retains_global_conflict_evidence():
    workspace = Workspace(DATA / "conflicts.csv")
    findings = workspace.inspect([2])["findings"]
    assert any(f["code"] == "conflicting_order_id" and 25 in f["row_ids"] for f in findings)


def test_no_matching_rows_explained():
    with pytest.raises(ValueError, match="No matching"):
        Workspace(DATA / "clean.csv").inspect([999])


def test_sample_cells_bounded_but_original_preserved(tmp_path):
    path = tmp_path / "long.csv"
    note = "x" * 5000
    path.write_text("order_id,order_date,amount,quantity,currency,note\nA,2026-09-01,1.00,1,INR," + note + "\n")
    workspace = Workspace(path)
    assert len(workspace.inspect()["sample"][0]["note"]) < 300
    assert workspace.data.loc[0, "note"] == note


def test_oversize_input_rejected(tmp_path):
    path = tmp_path / "large.csv"
    path.write_text("x" * 1_000_001)
    with pytest.raises(ValueError, match="1 MB"):
        Workspace(path)


def test_excess_rows_rejected(tmp_path):
    path = tmp_path / "many.csv"
    path.write_text("order_id,order_date,amount,quantity,currency\n" + "A,2026-09-01,1.00,1,INR\n" * 2001)
    with pytest.raises(ValueError, match="2,000"):
        Workspace(path)


def test_context_columns_survive_cleaning():
    workspace = Workspace(DATA / "formatting.csv")
    repair(workspace)
    clean = Workspace(DATA / "clean.csv").data
    pd.testing.assert_frame_equal(
        workspace.data[workspace.columns].reset_index(drop=True),
        clean[workspace.columns].reset_index(drop=True),
    )


def test_findings_cover_rows_outside_model_sample():
    observation = Workspace(DATA / "ambiguous.csv").inspect()
    assert len(observation["sample"]) == 12
    assert all(row["_row_id"] <= 12 for row in observation["sample"])
    assert any(f["code"] == "invalid_quantity" and 18 in f["row_ids"]
               for f in observation["findings"])
    assert any(f["code"] == "invalid_date" and 22 in f["row_ids"]
               for f in observation["findings"])
