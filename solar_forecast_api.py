#!/usr/bin/env python3
"""
solar_forecast_api.py
─────────────────────
REST service that streams

• 24 × 1 h  production forecast
• 7 × 1 d   production forecast
• 30 × 1 d  production forecast

for the B42_SOLAR meter (Arnhem, NL).

GET  http://0.0.0.0:8000/forecast
"""
import json, os, time, datetime as dt
from datetime import timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from flask import Flask, jsonify, Response
from sklearn.ensemble import GradientBoostingRegressor
from apscheduler.schedulers.background import BackgroundScheduler



# ── configuration ────────────────────────────────────────────────────────────
LAT, LON   = 52.0, 5.87                      # Arnhem, NL
SOLAR_ID   = "B42_SOLAR"
SOLAR_URL  = (
    "https://dashboard.hedge-iot.labs.vu.nl/ui_api/measurements?"
    f"meter_id={SOLAR_ID}&latest=false&raw=false&use_cache=true"
)

# OpenWeather – Developer plan
OWM_KEY   = os.getenv("OWM_KEY", "67d4dbfce7083e9195b41a0f4dafba74")
HIST_URL  = "https://history.openweathermap.org/data/2.5/history/city"
HOURLY_FC = "https://pro.openweathermap.org/data/2.5/forecast/hourly"
DAILY_FC  = "https://api.openweathermap.org/data/2.5/forecast/daily"

CACHE_TTL   = 1_800            # sec
HIST_DAYS   = 30
DAILY_SHORT = 7
DAILY_LONG  = 16

_cache: dict[str, object] = {"ts": 0, "payload": None}
app = Flask(__name__)
# ── helpers ──────────────────────────────────────────────────────────────────
def add_sun_flag(df: pd.DataFrame, sunrise_utc: pd.Timestamp, sunset_utc: pd.Timestamp):
    """
    Append a `sun_up` (0/1) column to *df* based on sunrise/sunset timestamps.
    Assumes df.index is tz-aware UTC datetimes.
    """
    return df.assign(sun_up=((df.index >= sunrise_utc) & (df.index <= sunset_utc)).astype(int))

def _dt_utc(ts):
    """
    Parse *scalar or array* into tz-aware UTC datetime(s).

    • numeric → treat as *seconds* when <1e12 else milliseconds
    • everything else → let pandas infer
    """
    if np.isscalar(ts):
        if isinstance(ts, (int, float, np.integer, np.floating)):
            unit = "s" if ts < 1e12 else "ms"
            return pd.to_datetime(ts, unit=unit, utc=True)
        return pd.to_datetime(ts, utc=True, errors="coerce")

    # Vectorised path (Series / ndarray)
    ts_arr = np.asarray(ts)
    if np.issubdtype(ts_arr.dtype, np.number):
        unit = "s" if np.nanmax(ts_arr) < 1e12 else "ms"
        return pd.to_datetime(ts_arr, unit=unit, utc=True)
    return pd.to_datetime(ts, utc=True, errors="coerce")



# ── solar data ───────────────────────────────────────────────────────────────
def get_solar_history(days: int = HIST_DAYS) -> pd.DataFrame:
    now   = dt.datetime.now(timezone.utc)
    start = now - dt.timedelta(days=days)

    url = f"{SOLAR_URL}&start_ts={int(start.timestamp()*1000)}"
    rec = requests.get(url, timeout=15).json()
    records = rec.get("measurements") if isinstance(rec, dict) else rec
    if not isinstance(records, list):
        raise ValueError("Unexpected solar API response shape")

    df = (
    pd.DataFrame(records)[["timestamp", "val"]]
      .rename(columns={"timestamp": "ts", "val": "kwh"})
      .assign(
        ts  = lambda d: _dt_utc(d["ts"]),

        kwh = lambda d: pd.to_numeric(d["kwh"], errors="coerce") / 1000.0,
        )
      .dropna(subset=["ts", "kwh"])
      .set_index("ts")
      .sort_index()       # ← sort first …
      .loc[start:now]     # ← … then slice (no KeyError)
      .resample("1H").sum(min_count=1)
      .fillna(0.0)
    )
    return df

def flag_sun_up(ts_series: pd.Series, sunrise_s: int, sunset_s: int) -> pd.Series:
    """Return 0/1 Series telling whether each UTC timestamp (in seconds) is in daylight."""
    return ((ts_series >= sunrise_s) & (ts_series <= sunset_s)).astype(int)
# ── weather data ─────────────────────────────────────────────────────────────
# ── weather data ────────────────────────────────────────────────────────────
def get_weather() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    return (30-day historic hourly, 48-h forecast hourly, daily forecast)
    Every DataFrame contains a `sun_up` 0/1 flag built only
    from OpenWeather’s sunrise / sunset values.
    """
    now = dt.datetime.now(timezone.utc)

    # -------- 30-day hourly history -----------------------------------------
    hist_rows: list[dict] = []
    for d in range(1, HIST_DAYS + 1):
        day = now - dt.timedelta(days=d)
        start_ts = int(day.replace(hour=0, minute=0, second=0).timestamp())
        end_ts   = int(day.replace(hour=23, minute=59, second=59).timestamp())

        r = requests.get(
            HIST_URL,
            params=dict(
                lat=LAT, lon=LON, type="hour",
                start=start_ts, end=end_ts,
                units="metric", appid=OWM_KEY,
            ),
            timeout=15,
        )
        r.raise_for_status()
        j = r.json()
        sunrise_s = j["city"]["sunrise"]
        sunset_s  = j["city"]["sunset"]

        for itm in j.get("list", []):
            hist_rows.append(
                dict(
                    ts       = _dt_utc(itm["dt"]),
                    temp     = itm["main"]["temp"],
                    clouds   = itm["clouds"]["all"],
                    humidity = itm["main"]["humidity"],
                    sun_up   = 1 if sunrise_s <= itm["dt"] <= sunset_s else 0,
                )
            )

    wx_hist = (
        pd.DataFrame(hist_rows)
          .set_index("ts")
          .sort_index()
          .drop_duplicates()
    )

    # -------- 48-hour HOURLY forecast ---------------------------------------
    # -------- 48-hour HOURLY forecast ---------------------------------------
    fc_hr = requests.get(
        HOURLY_FC,
        params=dict(lat=LAT, lon=LON, units="metric", appid=OWM_KEY),
        timeout=15,
    ).json()

    # we will fetch sunrise / sunset from the DAILY forecast instead
    fc_dl_short = requests.get(
        DAILY_FC,
        params=dict(lat=LAT, lon=LON, cnt=3, units="metric", appid=OWM_KEY),
        timeout=15,
    ).json()

    # map ISO-date → (sunrise-s, sunset-s)
    sun_map = {
        pd.to_datetime(rec["dt"], unit="s", utc=True).date():
        (rec["sunrise"], rec["sunset"])
        for rec in fc_dl_short["list"]
    }

    wx_hr = (
        pd.DataFrame(fc_hr["list"])[:48]
          .assign(
              ts       = lambda d: pd.to_datetime(d["dt"], unit="s", utc=True),
              temp     = lambda d: d["main"].apply(lambda m: m["temp"]),
              humidity = lambda d: d["main"].apply(lambda m: m["humidity"]),
              clouds   = lambda d: d["clouds"].apply(lambda c: c["all"]),
          )
          .set_index("ts")[["temp", "clouds", "humidity"]]
    )

    # daylight flag using sun_map
    wx_hr["sun_up"] = [
        1 if sun_map[row_ts.date()][0] <= int(row_ts.timestamp()) <= sun_map[row_ts.date()][1] else 0
        for row_ts in wx_hr.index
    ]


    # -------- 16-day DAILY forecast (keep sunrise / sunset) -----------------
    fc_dl = requests.get(
        DAILY_FC,
        params=dict(lat=LAT, lon=LON, cnt=16, units="metric", appid=OWM_KEY),
        timeout=15,
    ).json()

    wx_dl = (
        pd.DataFrame(fc_dl["list"])
          .assign(
              ts       = lambda d: pd.to_datetime(d["dt"], unit="s", utc=True)
                                   + pd.Timedelta(hours=12),
              temp     = lambda d: d["temp"].apply(lambda t: t["max"]),
              clouds   = lambda d: d["clouds"],
              humidity = lambda d: d["humidity"],
              sunrise  = lambda d: d["sunrise"],
              sunset   = lambda d: d["sunset"],
          )
          .set_index("ts")[["temp", "clouds", "humidity", "sunrise", "sunset"]]
          .sort_index()
    )

    return wx_hist, wx_hr, wx_dl


# ── model ────────────────────────────────────────────────────────────────────
# ── model ────────────────────────────────────────────────────────────────────
def train_model(solar: pd.DataFrame, wx_hist: pd.DataFrame):
    # keep only hours where net flow ≠ 0
    solar_hr = solar.loc[lambda s: s["kwh"] != 0].reset_index()
    solar_hr["ts_hour"] = solar_hr["ts"].dt.floor("H")

    wx_hist = (
        wx_hist.reset_index()
               .assign(ts_hour=lambda d: d["ts"].dt.floor("H"))
               .groupby("ts_hour")
               .agg(
                   temp="mean",
                   clouds="mean",
                   humidity="mean",
                   sun_up="max",        # at least one daylight record → 1
               )
               .reset_index()
    )

    merged = solar_hr.merge(wx_hist, on="ts_hour", how="inner").dropna()
    if len(merged) < 48:
        class ZeroModel:
            def predict(self, X): return np.zeros(len(X))
        return ZeroModel()

    merged["hour"] = merged["ts_hour"].dt.hour
    merged["doy"]  = merged["ts_hour"].dt.dayofyear

    X = merged[["temp", "clouds", "humidity", "hour", "doy", "sun_up"]]
    y = merged["kwh"]

    return GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=3, random_state=0
    ).fit(X, y)


# ── prediction helpers ───────────────────────────────────────────────────────
# ── prediction helpers ───────────────────────────────────────────────────────
def _predict_block(df, model, horizon):
    blk = df.iloc[:horizon].copy()
    blk["hour"] = blk.index.hour
    blk["doy"]  = blk.index.dayofyear

    X = blk[["temp", "clouds", "humidity", "hour", "doy", "sun_up"]].fillna(method="ffill")

    try:
        blk["pred_kwh"] = model.predict(X)
    except Exception:
        blk["pred_kwh"] = 0.0

    blk["timestamp"] = (blk.index.view("int64") // 1_000_000_000).astype(int)
    blk["pred_mwh"]  = blk["pred_kwh"] / 1000.0
    return blk[["pred_mwh", "timestamp"]].round(4).to_dict("records")

def make_forecasts(model, wx_hr: pd.DataFrame, wx_dl: pd.DataFrame):
    hourly = _predict_block(wx_hr, model, 24)

    def _expand_day(row_ts, row_vals):
        """Synthetic 24-hour frame for each daily forecast row."""
        base = row_ts.floor("D")
        sunrise_s = row_vals["sunrise"]
        sunset_s  = row_vals["sunset"]

        idx = [base + pd.Timedelta(hours=h) for h in range(24)]
        synth = pd.DataFrame(
            {
                "temp":     row_vals["temp"],
                "clouds":   row_vals["clouds"],
                "humidity": row_vals["humidity"],
                "sun_up":   [1 if sunrise_s <= t.timestamp() <= sunset_s else 0 for t in idx],
            },
            index=idx,
        )
        return synth

    daily_records = []
    day0_ts = pd.to_datetime(hourly[0]["timestamp"], unit="s", utc=True).floor("D")
    day0_val = round(sum(pt["pred_mwh"] for pt in hourly), 4)
    daily_records.append({"pred_mwh": day0_val, "timestamp": int(day0_ts.timestamp())})

    for ts, row in wx_dl.iloc[1:8].iterrows():  # Only 7 days instead of 16
        synth_df = _expand_day(ts, row)
        block = _predict_block(synth_df, model, 24)
        day_val = round(sum(pt["pred_mwh"] for pt in block), 4)
        daily_records.append(
            {"pred_mwh": day_val, "timestamp": int(ts.floor("D").timestamp())}
        )

    return hourly, daily_records  # ← Only return these




# ── Flask endpoint ───────────────────────────────────────────────────────────
@app.route("/forecast")
def forecast():
    now = time.time()
    if now - _cache["ts"] < CACHE_TTL:
        return jsonify(_cache["payload"])

    try:
        solar = get_solar_history()
        wx_hist, wx_hr, wx_dl = get_weather()
        model = train_model(solar, wx_hist)
        hourly, daily_7 = make_forecasts(model, wx_hr, wx_dl)

        payload = {
            "generated_utc": int(now),
            "hourly":  hourly,
            "daily_7": daily_7,
        }
        _cache.update(ts=now, payload=payload)
        return jsonify(payload)

    except Exception as exc:
        return Response(
            json.dumps({"error": str(exc)}), status=500, mimetype="application/json"
        )

if __name__ == "__main__":
    print("→  http://localhost:8000/forecast")
    app.run(host="0.0.0.0", port=8000)
