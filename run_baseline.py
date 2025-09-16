#!/usr/bin/env python3
import argparse
from datetime import datetime
from availability_baseline import (
    build_all,
    evaluate_anomaly,
    predict_curve_from,
    weekday_of,
)


def main():
    parser = argparse.ArgumentParser(description="Run availability baseline and anomaly detection")
    parser.add_argument("--data", default="raw_hotel_pms_data.json", help="Path to JSON data file")
    parser.add_argument("--check_in", help="Target check-in date YYYY-MM-DD for prediction/anomaly", required=False)
    parser.add_argument("--lead", type=int, help="Observed lead time for anomaly (default: max lead for date)", required=False)
    args = parser.parse_args()

    baseline, pickup, lt_series = build_all(args.data)

    # Summarize baselines
    leads = sorted(set(L for (L, _wd) in baseline.keys()))
    print(f"Computed baselines for {len(leads)} lead times. Example entries:")
    for key in sorted(baseline.keys())[:10]:
        L, wd = key
        med, scale, n = baseline[key]
        print(f"  L={L:3d} wd={wd} median={med:.1f} scale(MAD)={scale:.1f} N={n}")

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


