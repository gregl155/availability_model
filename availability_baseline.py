#!/usr/bin/env python3
"""
Availability baseline and pickup algorithm built from snapshot history.

Focus: total hotel availability per check-in date, not per room.

Key steps:
- For each parse day, take the latest snapshot by creation timestamp
- Within that snapshot, dedupe by room id (ignore meal plans) using max availability per room
- Sum across rooms to total hotel availability for that check-in date
- Compute lead time L = (check_in_date - parse_date).days (keep L >= 0)
- Build robust baselines by (L, weekday) using median and MAD
- Compute pickup curve: median of A(L-1) - A(L) by (L, weekday)
- Provide anomaly z-score and prediction roll-down from current lead time

Designed to be dependency-light and testable.
"""

from __future__ import annotations

import json
from collections import defaultdict, Counter
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from statistics import median
from typing import Dict, List, Tuple, Iterable, Optional


ISO_DT = "%Y-%m-%d %H:%M:%S.%f"
ISO_D = "%Y-%m-%d"


def parse_date(value: str) -> date:
    return datetime.strptime(value, ISO_D).date()


def parse_dt(value: str) -> datetime:
    # Some timestamps may lack microseconds; try fallback
    try:
        return datetime.strptime(value, ISO_DT)
    except ValueError:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


@dataclass
class SnapshotRecord:
    check_in_date: date
    parse_date: date
    creation_dt: datetime
    room_id: int
    availability: int


def load_records_from_json_array(path: str) -> List[SnapshotRecord]:
    """Load a large JSON array file into SnapshotRecord list.

    Note: This loads entire array in memory. For very large files, stream parsing
    could be added, but for ~360k records this is acceptable.
    """
    with open(path, "r") as f:
        data = json.load(f)

    records: List[SnapshotRecord] = []
    for row in data:
        try:
            records.append(
                SnapshotRecord(
                    check_in_date=parse_date(row["raw_check_in_date"]),
                    parse_date=parse_date(row["raw_parse_date"]),
                    creation_dt=parse_dt(row["raw_creation_date"]),
                    room_id=int(row["raw_room_id"]),
                    availability=int(row["raw_availability"]),
                )
            )
        except Exception:
            # Skip malformed rows
            continue
    return records


def select_latest_snapshot_per_parse_day(records: List[SnapshotRecord]) -> List[SnapshotRecord]:
    """Keep only records from the latest creation_dt per (parse_date).

    Implementation: find max creation_dt per parse_date, then filter records to that dt.
    """
    latest_dt_by_parse: Dict[date, datetime] = {}
    for r in records:
        cur = latest_dt_by_parse.get(r.parse_date)
        if cur is None or r.creation_dt > cur:
            latest_dt_by_parse[r.parse_date] = r.creation_dt

    filtered: List[SnapshotRecord] = [
        r for r in records if r.creation_dt == latest_dt_by_parse.get(r.parse_date)
    ]
    return filtered


def aggregate_total_availability(records: List[SnapshotRecord]) -> Dict[Tuple[date, date], int]:
    """Aggregate to total hotel availability by (check_in_date, parse_date).

    Steps per (check_in_date, parse_date): dedupe by room_id using max availability, then sum.
    Returns mapping {(check_in_date, parse_date): total_availability}.
    """
    # First dedupe by (check_in, parse, room) taking max availability
    max_avail_by_key: Dict[Tuple[date, date, int], int] = {}
    for r in records:
        key = (r.check_in_date, r.parse_date, r.room_id)
        prev = max_avail_by_key.get(key)
        if prev is None or r.availability > prev:
            max_avail_by_key[key] = r.availability

    totals: Dict[Tuple[date, date], int] = defaultdict(int)
    for (check_in, parse_d, _room), avail in max_avail_by_key.items():
        totals[(check_in, parse_d)] += max(0, avail)

    return totals


def to_lead_time_series(totals: Dict[Tuple[date, date], int]) -> Dict[Tuple[date, int], int]:
    """Transform totals to lead time index L = (check_in - parse).days >= 0.
    Returns mapping {(check_in_date, L): total_availability}.
    """
    lt_series: Dict[Tuple[date, int], int] = {}
    for (check_in, parse_d), total in totals.items():
        lead = (check_in - parse_d).days
        if lead >= 0:
            lt_series[(check_in, lead)] = total
    return lt_series


def weekday_of(d: date) -> int:
    # Monday=0 ... Sunday=6
    return d.weekday()


def compute_baseline_and_scale(lt_series: Dict[Tuple[date, int], int]) -> Dict[Tuple[int, int], Tuple[float, float, int]]:
    """Compute robust baseline median and MAD per (lead, weekday).

    Returns mapping {(lead, weekday): (median_value, mad_scaled, count)} where
    mad_scaled = 1.4826 * median(|x - median|).
    """
    by_key: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for (check_in, lead), total in lt_series.items():
        by_key[(lead, weekday_of(check_in))].append(total)

    out: Dict[Tuple[int, int], Tuple[float, float, int]] = {}
    for key, values in by_key.items():
        if not values:
            continue
        med = float(median(values))
        abs_dev = [abs(v - med) for v in values]
        mad = float(median(abs_dev)) if abs_dev else 0.0
        mad_scaled = 1.4826 * mad
        # Avoid zero scale
        if mad_scaled == 0.0:
            mad_scaled = 1.0
        out[key] = (med, mad_scaled, len(values))
    return out


def smooth_baseline(baseline: Dict[Tuple[int, int], Tuple[float, float, int]], window: int = 3) -> Dict[Tuple[int, int], Tuple[float, float, int]]:
    """Simple smoothing across neighboring lead times using tri-cube weights.
    Does not change counts; recomputes median proxy as weighted median-approx via weighted mean of medians.
    """
    def tricube(u: float) -> float:
        u = min(1.0, max(0.0, u))
        return (1 - u ** 3) ** 3

    leads = {lead for (lead, _wd) in baseline.keys()}
    weekdays = {wd for (_lead, wd) in baseline.keys()}

    smoothed: Dict[Tuple[int, int], Tuple[float, float, int]] = {}
    for wd in weekdays:
        for lead in leads:
            weights = []
            vals_med = []
            vals_scale = []
            total_n = 0
            for dl in range(-window, window + 1):
                key = (lead + dl, wd)
                if key in baseline:
                    med, scale, n = baseline[key]
                    w = tricube(abs(dl) / (window + 1e-9))
                    weights.append(w * n)
                    vals_med.append(med)
                    vals_scale.append(scale)
                    total_n += n
            if not weights:
                continue
            wsum = sum(weights)
            med_s = sum(w * v for w, v in zip(weights, vals_med)) / wsum
            scale_s = sum(w * v for w, v in zip(weights, vals_scale)) / wsum
            smoothed[(lead, wd)] = (med_s, max(1.0, scale_s), total_n)
    return smoothed


def compute_pickup_curve(lt_series: Dict[Tuple[date, int], int]) -> Dict[Tuple[int, int], float]:
    """Compute median pickup Δ(L) = A(L-1) - A(L) by (lead, weekday).
    Returned key uses current lead L (the step from L to L-1).
    """
    deltas: Dict[Tuple[int, int], List[int]] = defaultdict(list)

    # Build per check-in trajectory
    by_check_in: Dict[date, Dict[int, int]] = defaultdict(dict)
    for (check_in, lead), total in lt_series.items():
        by_check_in[check_in][lead] = total

    for check_in, series in by_check_in.items():
        leads_sorted = sorted(series.keys(), reverse=True)
        for i in range(1, len(leads_sorted)):
            L = leads_sorted[i]
            prev_L = leads_sorted[i - 1]
            # Ensure consecutive step (prev_L == L + 1)
            if prev_L == L + 1:
                d = series[L - 1] - series[L] if (L - 1) in series else None
                if d is None:
                    continue
                deltas[(L, weekday_of(check_in))].append(d)

    pickup: Dict[Tuple[int, int], float] = {}
    for key, values in deltas.items():
        pickup[key] = float(median(values))
    return pickup


@dataclass
class AnomalyResult:
    observed: int
    baseline: float
    scale: float
    z_score: float
    flag: str  # "low", "high", or "normal"


def evaluate_anomaly(observed: int, baseline: Dict[Tuple[int, int], Tuple[float, float, int]], lead: int, weekday: int, z_threshold: float = 2.0) -> AnomalyResult:
    base = baseline.get((lead, weekday))
    if not base:
        # Fallback to any weekday baseline for this lead
        candidates = [v for (L, _wd), v in baseline.items() if L == lead]
        if candidates:
            med, scale, _n = candidates[0]
        else:
            # Hard fallback
            med, scale = 0.0, 1.0
    else:
        med, scale, _n = base
    z = (observed - med) / (scale if scale > 0 else 1.0)
    flag = "normal"
    if z < -z_threshold:
        flag = "low"
    elif z > z_threshold:
        flag = "high"
    return AnomalyResult(observed=observed, baseline=med, scale=scale, z_score=z, flag=flag)


def predict_curve_from(lt_series: Dict[Tuple[date, int], int], pickup: Dict[Tuple[int, int], float], check_in: date, start_lead: int) -> List[Tuple[int, float]]:
    """Roll forward expected availability from start_lead down to 0 using pickup.

    expected(L-1) = max(0, expected(L) + median_delta(L, weekday)).
    """
    weekday = weekday_of(check_in)
    # Start from observed if present; else 0
    expected_at_L = float(lt_series.get((check_in, start_lead), 0))
    curve: List[Tuple[int, float]] = [(start_lead, expected_at_L)]
    for L in range(start_lead, 0, -1):
        delta = pickup.get((L, weekday))
        if delta is None:
            # Fallback to any weekday for this L
            alt = [v for (lead, _wd), v in pickup.items() if lead == L]
            delta = float(median(alt)) if alt else 0.0
        next_val = max(0.0, expected_at_L + delta)
        curve.append((L - 1, next_val))
        expected_at_L = next_val
    return curve


def build_all(path: str) -> Tuple[Dict[Tuple[int, int], Tuple[float, float, int]], Dict[Tuple[int, int], float], Dict[Tuple[date, int], int]]:
    """Convenience pipeline: load -> latest snapshot per parse day -> aggregate -> lead time -> baseline/pickup.
    Returns (smoothed_baseline, pickup_curve, lt_series).
    """
    records = load_records_from_json_array(path)
    latest = select_latest_snapshot_per_parse_day(records)
    totals = aggregate_total_availability(latest)
    lt_series = to_lead_time_series(totals)
    base_raw = compute_baseline_and_scale(lt_series)
    base_smooth = smooth_baseline(base_raw)
    pickup = compute_pickup_curve(lt_series)
    return base_smooth, pickup, lt_series


__all__ = [
    "SnapshotRecord",
    "load_records_from_json_array",
    "select_latest_snapshot_per_parse_day",
    "aggregate_total_availability",
    "to_lead_time_series",
    "compute_baseline_and_scale",
    "smooth_baseline",
    "compute_pickup_curve",
    "AnomalyResult",
    "evaluate_anomaly",
    "predict_curve_from",
    "build_all",
]


