#!/usr/bin/env python3
import argparse
from datetime import datetime, timedelta, date
from availability_baseline import (
    build_all,
    evaluate_anomaly,
    predict_curve_from,
    weekday_of,
    load_records_from_json_array,
    select_latest_snapshot_per_parse_day,
    aggregate_total_availability,
)


def main():
    parser = argparse.ArgumentParser(description="Run availability baseline and anomaly detection")
    parser.add_argument("--data", default="raw_hotel_pms_data.json", help="Path to JSON data file")
    parser.add_argument("--check_in", help="Target check-in date YYYY-MM-DD for prediction/anomaly", required=False)
    parser.add_argument("--lead", type=int, help="Observed lead time for anomaly (default: max lead for date)", required=False)
    parser.add_argument("--report_range_start", help="Start date YYYY-MM-DD for report mode", required=False)
    parser.add_argument("--report_range_end", help="End date YYYY-MM-DD for report mode", required=False)
    parser.add_argument("--asof", help="Use snapshots from this parse date YYYY-MM-DD (default: latest available)", required=False)
    parser.add_argument("--z_threshold", type=float, default=2.0, help="Z-score threshold for abnormal")
    parser.add_argument("--only_abnormal", action="store_true", help="Only print abnormal days in report mode")
    args = parser.parse_args()

    baseline, pickup, lt_series = build_all(args.data)

    # For report mode we also need totals keyed by (check_in, parse_date)
    records = load_records_from_json_array(args.data)
    latest = select_latest_snapshot_per_parse_day(records)
    totals = aggregate_total_availability(latest)

    # Summarize baselines
    leads = sorted(set(L for (L, _wd) in baseline.keys()))
    print(f"Computed baselines for {len(leads)} lead times. Example entries:")
    for key in sorted(baseline.keys())[:10]:
        L, wd = key
        med, scale, n = baseline[key]
        print(f"  L={L:3d} wd={wd} median={med:.1f} scale(MAD)={scale:.1f} N={n}")

    # Report mode across a date range using a specific as-of parse date
    if args.report_range_start or args.report_range_end:
        if args.asof:
            asof = datetime.strptime(args.asof, "%Y-%m-%d").date()
        else:
            # Latest available parse date in data
            asof = max(pd for (_ci, pd) in totals.keys())

        start = datetime.strptime(args.report_range_start, "%Y-%m-%d").date() if args.report_range_start else asof + timedelta(days=1)
        end = datetime.strptime(args.report_range_end, "%Y-%m-%d").date() if args.report_range_end else asof + timedelta(days=183)
        if end < start:
            print("Invalid range: end before start")
            return

        print()
        print(f"Report as-of {asof} for range {start} to {end} (z>{args.z_threshold:.1f} or <{-args.z_threshold:.1f} abnormal)")
        print("date,weekday,lead,observed,baseline,scale,z,flag")

        cur = start
        abnormal_count = 0
        total_count = 0
        while cur <= end:
            # Choose most recent available parse_date <= min(cur, asof)
            cutoff = min(cur, asof)
            candidate_pds = [pd for (ci, pd) in totals.keys() if ci == cur and pd <= cutoff]
            if candidate_pds:
                best_pd = max(candidate_pds)
                lead = (cur - best_pd).days
                obs = totals[(cur, best_pd)]
                ar = evaluate_anomaly(obs, baseline, lead, weekday_of(cur), z_threshold=args.z_threshold)
                row = f"{cur},{weekday_of(cur)},{lead},{ar.observed},{ar.baseline:.1f},{ar.scale:.1f},{ar.z_score:.2f},{ar.flag}"
                total_count += 1
                if args.only_abnormal:
                    if ar.flag != "normal":
                        print(row)
                        abnormal_count += 1
                else:
                    print(row)
                    if ar.flag != "normal":
                        abnormal_count += 1
            cur += timedelta(days=1)

        print()
        print(f"Days evaluated: {total_count}, abnormal: {abnormal_count}")
        return

    if args.check_in:
        ci = datetime.strptime(args.check_in, "%Y-%m-%d").date()
        # Determine observed lead if not provided (max available for this date)
        if args.lead is None:
            leads_for_date = [L for (d, L) in lt_series.keys() if d == ci]
            if not leads_for_date:
                print("No data for requested check-in date.")
                return
            L0 = max(leads_for_date)
        else:
            L0 = args.lead

        observed = lt_series.get((ci, L0), 0)
        ar = evaluate_anomaly(observed, baseline, L0, weekday_of(ci))
        print()
        print(f"Anomaly for {ci} at lead {L0}:")
        print(f"  observed={ar.observed} baseline={ar.baseline:.1f} scale={ar.scale:.1f} z={ar.z_score:.2f} flag={ar.flag}")

        curve = predict_curve_from(lt_series, pickup, ci, L0)
        print("\nPredicted availability curve (lead -> expected):")
        for L, v in curve[:20]:
            print(f"  L={L:3d} -> {v:.1f}")


if __name__ == "__main__":
    main()


