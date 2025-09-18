#!/usr/bin/env python3
"""
Pickup Anomaly Detection Model

Uses historical booking curves to detect future check-in dates with problematic pickup patterns.
"""

from datetime import date, timedelta
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import statistics
from availability_baseline import load_records_from_json_array, weekday_of


@dataclass
class PickupMetrics:
    check_in: date
    current_availability: int
    expected_availability: float
    pickup_velocity: float  # rooms per day
    expected_velocity: float
    days_to_arrival: int
    volatility: float
    trend: str
    issue_flags: List[str]


def compute_pickup_velocity(progression: List[Tuple[date, int]], window_days: int = 7) -> float:
    """Compute pickup velocity (rooms booked per day) over recent window."""
    if len(progression) < 2:
        return 0.0
    
    recent = progression[-min(len(progression), window_days):]
    if len(recent) < 2:
        return 0.0
    
    # Linear regression slope
    n = len(recent)
    x_vals = [(recent[i][0] - recent[0][0]).days for i in range(n)]
    y_vals = [recent[i][1] for i in range(n)]
    
    if len(set(x_vals)) < 2:
        return 0.0
    
    x_mean = sum(x_vals) / n
    y_mean = sum(y_vals) / n
    
    numerator = sum((x_vals[i] - x_mean) * (y_vals[i] - y_mean) for i in range(n))
    denominator = sum((x_vals[i] - x_mean) ** 2 for i in range(n))
    
    return numerator / denominator if denominator != 0 else 0.0


def build_historical_baselines(records, cutoff_date: date) -> Dict:
    """Build historical pickup baselines from past data."""
    # Only use data before cutoff for training
    train_records = [r for r in records if r.parse_date < cutoff_date]
    
    by_checkin = defaultdict(list)
    for r in train_records:
        by_checkin[r.check_in_date].append(r)
    
    velocities = defaultdict(list)
    expected_avail = defaultdict(list)
    
    for ci_date, ci_records in by_checkin.items():
        if len(ci_records) < 10:
            continue
            
        # Build progression
        by_pd = defaultdict(lambda: defaultdict(list))
        for r in ci_records:
            by_pd[r.parse_date][r.room_id].append(r.availability)
        
        progression = []
        for pd in sorted(by_pd.keys()):
            total = sum(max(avails) for avails in by_pd[pd].values())
            progression.append((pd, total))
        
        if len(progression) < 5:
            continue
        
        wd = weekday_of(ci_date)
        
        # Extract patterns at different lead times
        for i, (pd, avail) in enumerate(progression):
            lead = (ci_date - pd).days
            if lead <= 0:
                continue
            
            # Velocity pattern
            if i >= 2:
                window = progression[max(0, i-4):i+1]
                velocity = compute_pickup_velocity(window)
                velocities[(lead//7, wd)].append(velocity)  # Weekly buckets
            
            # Expected availability pattern
            expected_avail[(lead//7, wd)].append(avail)
    
    # Compute baselines
    velocity_baselines = {}
    avail_baselines = {}
    
    for key, vals in velocities.items():
        if len(vals) >= 3:
            velocity_baselines[key] = {
                'median': statistics.median(vals),
                'p25': statistics.quantiles(vals, n=4)[0] if len(vals) >= 4 else min(vals),
                'p75': statistics.quantiles(vals, n=4)[2] if len(vals) >= 4 else max(vals)
            }
    
    for key, vals in expected_avail.items():
        if len(vals) >= 3:
            avail_baselines[key] = {
                'median': statistics.median(vals),
                'p25': statistics.quantiles(vals, n=4)[0] if len(vals) >= 4 else min(vals),
                'p75': statistics.quantiles(vals, n=4)[2] if len(vals) >= 4 else max(vals)
            }
    
    return {
        'velocity_baselines': velocity_baselines,
        'avail_baselines': avail_baselines
    }


def analyze_pickup_issues(check_in: date, progression: List[Tuple[date, int]], 
                         baselines: Dict, as_of: date) -> PickupMetrics:
    """Analyze pickup pattern for anomalies."""
    if not progression:
        return PickupMetrics(
            check_in=check_in, current_availability=0, expected_availability=0,
            pickup_velocity=0, expected_velocity=0, days_to_arrival=0,
            volatility=0, trend="unknown", issue_flags=["no_data"]
        )
    
    current_avail = progression[-1][1]
    days_to_arrival = (check_in - as_of).days
    wd = weekday_of(check_in)
    lead_bucket = days_to_arrival // 7
    
    # Get baselines
    velocity_key = (lead_bucket, wd)
    avail_key = (lead_bucket, wd)
    
    velocity_baseline = baselines['velocity_baselines'].get(velocity_key)
    avail_baseline = baselines['avail_baselines'].get(avail_key)
    
    # Compute current metrics
    recent_velocity = compute_pickup_velocity(progression, window_days=7)
    
    if len(progression) >= 3:
        recent_avails = [p[1] for p in progression[-5:]]
        volatility = statistics.stdev(recent_avails) if len(recent_avails) > 1 else 0
        
        # Trend
        if recent_avails[-1] < recent_avails[0] - 1:
            trend = "declining"
        elif recent_avails[-1] > recent_avails[0] + 1:
            trend = "increasing"
        else:
            trend = "stable"
    else:
        volatility = 0
        trend = "unknown"
    
    # Expected values from baselines
    expected_velocity = velocity_baseline['median'] if velocity_baseline else -0.5
    expected_avail = avail_baseline['median'] if avail_baseline else 3.0
    
    # Issue detection
    issues = []
    
    # Slow pickup (velocity too low)
    if velocity_baseline and recent_velocity > velocity_baseline['p75'] and days_to_arrival < 30:
        issues.append("slow_pickup")
    
    # Fast pickup (velocity too high early)
    if velocity_baseline and recent_velocity < velocity_baseline['p25'] and days_to_arrival > 45:
        issues.append("fast_pickup")
    
    # High availability vs expected
    if avail_baseline and current_avail > avail_baseline['p75'] and days_to_arrival < 21:
        issues.append("high_availability")
    
    # Low availability vs expected (early sellout)
    if avail_baseline and current_avail < avail_baseline['p25'] and days_to_arrival > 30:
        issues.append("early_sellout")
    
    # Stalled pickup
    if trend == "stable" and days_to_arrival < 14 and current_avail > 2:
        issues.append("stalled_pickup")
    
    # High volatility
    if volatility > 3.0:
        issues.append("erratic_pickup")
    
    return PickupMetrics(
        check_in=check_in,
        current_availability=current_avail,
        expected_availability=expected_avail,
        pickup_velocity=recent_velocity,
        expected_velocity=expected_velocity,
        days_to_arrival=days_to_arrival,
        volatility=volatility,
        trend=trend,
        issue_flags=issues
    )


def find_anomaly_dates(data_path: str, train_cutoff: str, future_days: int = 60) -> List[PickupMetrics]:
    """Find future dates with pickup anomalies."""
    records = load_records_from_json_array(data_path)
    cutoff_date = date.fromisoformat(train_cutoff)
    
    print(f"Training baselines on data before {cutoff_date}...")
    baselines = build_historical_baselines(records, cutoff_date)
    print(f"Built {len(baselines['velocity_baselines'])} velocity baselines")
    print(f"Built {len(baselines['avail_baselines'])} availability baselines")
    
    # Analyze future dates
    future_start = cutoff_date + timedelta(days=1)
    future_end = cutoff_date + timedelta(days=future_days)
    
    by_checkin = defaultdict(list)
    for r in records:
        if future_start <= r.check_in_date <= future_end and r.parse_date <= cutoff_date:
            by_checkin[r.check_in_date].append(r)
    
    results = []
    for ci_date in sorted(by_checkin.keys()):
        ci_records = by_checkin[ci_date]
        
        # Build progression
        by_pd = defaultdict(lambda: defaultdict(list))
        for r in ci_records:
            by_pd[r.parse_date][r.room_id].append(r.availability)
        
        progression = []
        for pd in sorted(by_pd.keys()):
            total = sum(max(avails) for avails in by_pd[pd].values())
            progression.append((pd, total))
        
        if len(progression) >= 3:
            metrics = analyze_pickup_issues(ci_date, progression, baselines, cutoff_date)
            if metrics.issue_flags and 'no_data' not in metrics.issue_flags:
                results.append(metrics)
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pickup_anomaly.py TRAIN_CUTOFF_DATE [FUTURE_DAYS]")
        print("Example: python pickup_anomaly.py 2025-08-01 60")
        sys.exit(1)
    
    train_cutoff = sys.argv[1]
    future_days = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    
    anomalies = find_anomaly_dates("raw_hotel_pms_data.json", train_cutoff, future_days)
    
    if not anomalies:
        print("✅ No pickup anomalies detected")
    else:
        print(f"\n🚨 PICKUP ANOMALIES DETECTED ({len(anomalies)} dates)")
        print("=" * 70)
        
        for pm in sorted(anomalies, key=lambda x: x.days_to_arrival):
            flags = ', '.join(pm.issue_flags)
            print(f"{pm.check_in} ({pm.days_to_arrival:2d}d): "
                  f"avail={pm.current_availability:2d} (exp={pm.expected_availability:.1f}), "
                  f"vel={pm.pickup_velocity:+5.2f} (exp={pm.expected_velocity:+5.2f}), "
                  f"trend={pm.trend}, issues=[{flags}]")
