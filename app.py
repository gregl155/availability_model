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


@app.get("/")
def root():
    return send_from_directory(app.static_folder, "progression.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)


