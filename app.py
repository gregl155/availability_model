#!/usr/bin/env python3
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from availability_baseline import (
    build_all,
    load_records_from_json_array,
    select_latest_snapshot_per_parse_day,
    aggregate_total_availability,
    weekday_of,
    evaluate_anomaly,
)


DATA_PATH = "raw_hotel_pms_data.json"

app = Flask(__name__, static_url_path="/static", static_folder="static")

print("Loading data and building baselines (this may take a few seconds)...")
baseline, pickup, lt_series = build_all(DATA_PATH)
records = load_records_from_json_array(DATA_PATH)
latest = select_latest_snapshot_per_parse_day(records)
totals = aggregate_total_availability(latest)
print("Ready.")


@app.get("/api/progression")
def api_progression():
    check_in_str = request.args.get("check_in")
    z = float(request.args.get("z", "2.0"))
    if not check_in_str:
        return jsonify({"error": "Missing 'check_in' query param YYYY-MM-DD"}), 400
    try:
        ci = datetime.strptime(check_in_str, "%Y-%m-%d").date()
    except ValueError:
        return jsonify({"error": "Invalid 'check_in' format. Use YYYY-MM-DD"}), 400

    # Build progression specifically for this check-in: for each parse_date,
    # use the latest creation_dt that contains this check-in, then dedupe rooms and sum
    subset = [r for r in records if r.check_in_date == ci and r.parse_date <= ci]
    if not subset:
        return jsonify({
            "check_in": check_in_str,
            "points": [],
            "message": "No snapshots available for this check-in date"
        })

    # Latest creation per parse_date within subset
    latest_dt = {}
    for r in subset:
        cur = latest_dt.get(r.parse_date)
        if cur is None or r.creation_dt > cur:
            latest_dt[r.parse_date] = r.creation_dt

    # For each parse_date, compute total availability (dedupe by room_id with max availability)
    totals_by_pd = {}
    for pd, cdt in latest_dt.items():
        per_room = {}
        for r in subset:
            if r.parse_date == pd and r.creation_dt == cdt:
                prev = per_room.get(r.room_id)
                if prev is None or r.availability > prev:
                    per_room[r.room_id] = r.availability
        totals_by_pd[pd] = sum(max(0, v) for v in per_room.values())

    if not totals_by_pd:
        return jsonify({
            "check_in": check_in_str,
            "points": [],
            "message": "No snapshots available for this check-in date"
        })

    wd = weekday_of(ci)
    points = []
    for pd in sorted(totals_by_pd.keys()):
        obs = totals_by_pd[pd]
        lead = (ci - pd).days
        ar = evaluate_anomaly(obs, baseline, lead, wd)
        lo = ar.baseline - z * ar.scale
        hi = ar.baseline + z * ar.scale
        points.append({
            "parse_date": pd.isoformat(),
            "lead": lead,
            "observed": obs,
            "baseline": ar.baseline,
            "lo": lo,
            "hi": hi,
            "z": ar.z_score,
            "flag": ar.flag,
        })

    return jsonify({
        "check_in": check_in_str,
        "weekday": wd,
        "z": z,
        "points": points,
    })


def _compute_progression_for_checkin(ci_date):
    # Using raw records to ensure progression is built from latest snapshot per parse_date for that CI
    subset = [r for r in records if r.check_in_date == ci_date and r.parse_date <= ci_date]
    if not subset:
        return []
    latest_dt = {}
    for r in subset:
        cur = latest_dt.get(r.parse_date)
        if cur is None or r.creation_dt > cur:
            latest_dt[r.parse_date] = r.creation_dt
    totals_by_pd = {}
    for pd, cdt in latest_dt.items():
        per_room = {}
        for r in subset:
            if r.parse_date == pd and r.creation_dt == cdt:
                prev = per_room.get(r.room_id)
                if prev is None or r.availability > prev:
                    per_room[r.room_id] = r.availability
        totals_by_pd[pd] = sum(max(0, v) for v in per_room.values())
    return sorted([(pd, total) for pd, total in totals_by_pd.items()], key=lambda x: x[0])


@app.get("/api/progression_multi")
def api_progression_multi():
    # Returns multi-series: one line per check-in date, summing raw_availability across rooms per parse day
    # Query params:
    #   start_ci, end_ci (YYYY-MM-DD) optional; if missing, use min/max in data
    #   max_series: limit number of check-in dates (default 30, most recent)
    #   order: 'recent' (default) or 'chron'
    start_ci = request.args.get("start_ci")
    end_ci = request.args.get("end_ci")
    max_series = int(request.args.get("max_series", "30"))
    order = request.args.get("order", "recent")

    all_ci = sorted({r.check_in_date for r in records})
    if not all_ci:
        return jsonify({"series": []})
    if start_ci:
        try:
            sdt = datetime.strptime(start_ci, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"error": "Invalid start_ci"}), 400
        all_ci = [d for d in all_ci if d >= sdt]
    if end_ci:
        try:
            edt = datetime.strptime(end_ci, "%Y-%m-%d").date()
        except ValueError:
            return jsonify({"error": "Invalid end_ci"}), 400
        all_ci = [d for d in all_ci if d <= edt]

    if order == "recent":
        all_ci = list(reversed(all_ci))
    # limit series count
    if max_series > 0:
        all_ci = all_ci[:max_series]
    if order == "recent":
        all_ci = list(reversed(all_ci))  # keep chronological left->right in chart

    # Build series
    series = []
    for ci in all_ci:
        pts = _compute_progression_for_checkin(ci)
        series.append({
            "check_in": ci.isoformat(),
            "weekday": weekday_of(ci),
            "points": [{"parse_date": pd.isoformat(), "observed": tot} for pd, tot in pts],
        })

    # Provide the union of parse_date labels for convenience
    label_set = set()
    for s in series:
        for p in s["points"]:
            label_set.add(p["parse_date"])
    labels = sorted(label_set)
    return jsonify({"labels": labels, "series": series})


@app.get("/")
def root():
    return send_from_directory(app.static_folder, "progression_simple.html")


@app.get("/multi")
def multi_page():
    return send_from_directory(app.static_folder, "progression_multi.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)


