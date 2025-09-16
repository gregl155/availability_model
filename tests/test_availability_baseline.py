import os
from datetime import date, datetime, timedelta

from availability_baseline import (
    SnapshotRecord,
    select_latest_snapshot_per_parse_day,
    aggregate_total_availability,
    to_lead_time_series,
    compute_baseline_and_scale,
    evaluate_anomaly,
    compute_pickup_curve,
    predict_curve_from,
)


def make_rec(ci: str, pd: str, ct: str, room: int, avail: int) -> SnapshotRecord:
    return SnapshotRecord(
        check_in_date=date.fromisoformat(ci),
        parse_date=date.fromisoformat(pd),
        creation_dt=datetime.fromisoformat(ct),
        room_id=room,
        availability=avail,
    )


def test_latest_snapshot_selection():
    recs = [
        make_rec("2025-05-20", "2025-05-10", "2025-05-10 10:00:00", 1, 2),
        make_rec("2025-05-20", "2025-05-10", "2025-05-10 12:00:00", 1, 3),
        make_rec("2025-05-21", "2025-05-11", "2025-05-11 09:00:00", 2, 4),
    ]
    latest = select_latest_snapshot_per_parse_day(recs)
    dts = sorted({r.creation_dt for r in latest})
    assert dts == [
        datetime.fromisoformat("2025-05-10 12:00:00"),
        datetime.fromisoformat("2025-05-11 09:00:00"),
    ]


def test_aggregate_total_availability_dedup_by_room():
    recs = [
        make_rec("2025-05-20", "2025-05-10", "2025-05-10 12:00:00", 1, 2),
        make_rec("2025-05-20", "2025-05-10", "2025-05-10 12:00:00", 1, 3),  # max per room
        make_rec("2025-05-20", "2025-05-10", "2025-05-10 12:00:00", 2, 5),
    ]
    totals = aggregate_total_availability(recs)
    assert totals[(date(2025, 5, 20), date(2025, 5, 10))] == 3 + 5


def test_lead_time_series_and_baseline():
    recs = [
        make_rec("2025-05-20", "2025-05-10", "2025-05-10 12:00:00", 1, 2),
        make_rec("2025-05-20", "2025-05-10", "2025-05-10 12:00:00", 2, 4),
        make_rec("2025-05-21", "2025-05-10", "2025-05-10 12:00:00", 3, 3),
    ]
    totals = aggregate_total_availability(recs)
    lt = to_lead_time_series(totals)
    # L for 2025-05-20 vs 2025-05-10 is 10
    assert lt[(date(2025, 5, 20), 10)] == 6
    base = compute_baseline_and_scale(lt)
    assert any(k[0] == 10 for k in base.keys())


def test_anomaly_and_prediction_curve():
    # Build a simple series for one date with leads 3->0
    # A(3)=10, A(2)=9, A(1)=7, A(0)=5 so pickups: L3->2=1, L2->1=2, L1->0=2
    recs = []
    ci = date(2025, 5, 13)
    for L, total in [(3, 10), (2, 9), (1, 7), (0, 5)]:
        parse_d = ci - timedelta(days=L)
        # create 2 rooms that sum to total
        r1 = total // 2
        r2 = total - r1
        recs.append(make_rec(ci.isoformat(), parse_d.isoformat(), f"{parse_d.isoformat()} 12:00:00", 1, r1))
        recs.append(make_rec(ci.isoformat(), parse_d.isoformat(), f"{parse_d.isoformat()} 12:00:00", 2, r2))
    totals = aggregate_total_availability(recs)
    lt = to_lead_time_series(totals)
    pickup = compute_pickup_curve(lt)
    curve = predict_curve_from(lt, pickup, ci, 3)
    assert curve[0][0] == 3
    # Check that curve rolls down with expected pickups (approx)
    # Expected sequence starting from observed 10 at L=3: L2-> ~11, L1-> ~13, L0-> ~15 (using median deltas 1,2,2)
    assert curve[-1][0] == 0


