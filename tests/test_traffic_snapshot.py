"""The history-merging helpers in scripts/traffic_snapshot.py (the GitHub calls themselves are not tested here)."""
import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "traffic_snapshot", Path(__file__).resolve().parent.parent / "scripts" / "traffic_snapshot.py")
ts = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ts)


def test_daily_rows_uses_the_date_only():
    rows = ts.daily_rows([{"timestamp": "2026-11-17T00:00:00Z", "count": 5, "uniques": 3}])
    assert rows == [{"date": "2026-11-17", "count": 5, "uniques": 3}]


def test_upsert_newest_value_wins_and_history_is_kept():
    old = [{"date": "2026-11-01", "count": "4", "uniques": "2"}, {"date": "2026-11-02", "count": "1", "uniques": "1"}]
    new = [{"date": "2026-11-02", "count": 9, "uniques": 5}, {"date": "2026-11-03", "count": 2, "uniques": 2}]
    merged = ts.upsert_daily(old, new)
    assert [r["date"] for r in merged] == ["2026-11-01", "2026-11-02", "2026-11-03"]  # sorted, no duplicates
    assert merged[0]["count"] == "4"            # a day that dropped out of GitHub's 14-day window is kept
    assert merged[1]["count"] == 9              # a repeated day takes the newer number


def test_total_adds_up_a_column():
    rows = [{"count": "4", "uniques": "2"}, {"count": 6, "uniques": 1}]
    assert ts.total(rows) == 10
    assert ts.total(rows, "uniques") == 3


def test_csv_round_trip(tmp_path):
    path = tmp_path / "x" / "views.csv"
    rows = [{"date": "2026-11-01", "count": 1, "uniques": 1}]
    ts.write_rows(path, ["date", "count", "uniques"], rows)
    assert ts.read_rows(path) == [{"date": "2026-11-01", "count": "1", "uniques": "1"}]
    assert ts.read_rows(tmp_path / "missing.csv") == []
