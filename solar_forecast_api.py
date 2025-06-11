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

def _cloud_bucket(pct: float) -> int:
    """0 clear (<20 %), 1 partly (20–50 %), 2 broken (50–80 %), 3 overcast (≥80 %)."""
    if pct < 20:   return 0
    if pct < 50:   return 1
    if pct < 80:   return 2
    return 3

def _daily_energy(series_kw: pd.Series) -> float:
    """Return absolute MWh energy for one day given hourly kWh series."""
    return series_kw.resample("1D").sum(min_count=12).abs().max()


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
      .resample("1h").sum(min_count=1)
      .fillna(0.0)
    )
    return df

# ── weather data ─────────────────────────────────────────────────────────────
def get_weather() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        wx_hist – 30-day historic hourly  (temp, humidity, clouds, cloud_bucket, sun_up)
        wx_hr   – next-48 h forecast hourly (same columns)
        wx_dl   – 16-day daily forecast    (temp, humidity, clouds, sunrise, sunset)
    Uses only the allowed OpenWeather fields.
    """
    now = dt.datetime.now(timezone.utc)

    # ---------- 48 h HOURLY forecast ----------------------------------------
    fc_hr = requests.get(
        HOURLY_FC,
        params=dict(lat=LAT, lon=LON, units="metric", appid=OWM_KEY),
        timeout=15,
    ).json()

    wx_hr = (
        pd.DataFrame(fc_hr["list"])[:48]
        .assign(
            ts       = lambda d: pd.to_datetime(d["dt"], unit="s", utc=True),
            temp     = lambda d: d["main"].apply(lambda m: m["temp"]),
            humidity = lambda d: d["main"].apply(lambda m: m["humidity"]),
            clouds   = lambda d: d["clouds"].apply(lambda c: c["all"]),
        )
    )

# row-wise derive sun_up from sys.pod, falling back to icon d/n
    wx_hr["sun_up"] = wx_hr.apply(
        lambda row: 1
            if (row.get("sys", {}).get("pod") == "d")
               or (row["weather"][0]["icon"].endswith("d"))
            else 0,
        axis=1,
    )
    wx_hr = wx_hr.set_index("ts")[["temp", "humidity", "clouds", "sun_up"]]
    wx_hr["cloud_bucket"] = wx_hr["clouds"].apply(_cloud_bucket)


    # ---------- 30-day HOURLY history ---------------------------------------
    hist_rows = []
    for back in range(1, HIST_DAYS + 1):
        day      = now - dt.timedelta(days=back)
        start_ts = int(day.replace(hour=0,  minute=0, second=0).timestamp())
        end_ts   = int(day.replace(hour=23, minute=59, second=59).timestamp())

        r = requests.get(
            HIST_URL,
            params=dict(lat=LAT, lon=LON, type="hour",
                        start=start_ts, end=end_ts,
                        units="metric", appid=OWM_KEY),
            timeout=15,
        )
        for itm in r.json().get("list", []):
            clouds = itm["clouds"]["all"]
            hist_rows.append(
                dict(
                    ts           = _dt_utc(itm["dt"]),
                    temp         = itm["main"]["temp"],
                    humidity     = itm["main"]["humidity"],
                    clouds       = clouds,
                    cloud_bucket = _cloud_bucket(clouds),
                    sun_up = 1 if itm.get("sys", {}).get("pod", "n") == "d" \
                                or itm["weather"][0]["icon"].endswith("d") else 0,
                )
            )

    wx_hist = (
        pd.DataFrame(hist_rows)
        .set_index("ts")
        .sort_index()
        .drop_duplicates()
    )

    # ---------- 16-day DAILY forecast (keep sunrise/sunset) -----------------
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
            humidity = lambda d: d["humidity"],
            clouds   = lambda d: d["clouds"],
        )
        .set_index("ts")[["temp", "humidity", "clouds", "sunrise", "sunset"]]
        .sort_index()
    )

    return wx_hist, wx_hr, wx_dl


# ── model ────────────────────────────────────────────────────────────────────
# ── model ────────────────────────────────────────────────────────────────────
from sklearn.linear_model import HuberRegressor
from numpy import log1p, expm1, clip, zeros

def train_model(solar: pd.DataFrame, wx_hist: pd.DataFrame):
    """Fit log-capacity-factor model. Always returns an object with .predict(df)."""
    # ── join solar + weather ────────────────────────────────────────────────
    solar_hr = solar[solar["kwh"] != 0].reset_index()
    solar_hr["ts_hour"] = solar_hr["ts"].dt.floor("h")

    wx_hist_agg = (
        wx_hist.reset_index()
               .assign(ts_hour=lambda d: d["ts"].dt.floor("h"))
               .groupby("ts_hour")
               .agg(
                   temp         = ("temp", "mean"),
                   humidity     = ("humidity", "mean"),
                   clouds       = ("clouds", "mean"),
                   cloud_bucket = ("cloud_bucket", "max"),
                   sun_up       = ("sun_up", "max"),
               )
               .reset_index()
    )

    merged = solar_hr.merge(wx_hist_agg, on="ts_hour", how="inner").dropna()
    if merged.empty:
        class ZeroModel:                   # rare: no overlap
            def predict(self, X): return zeros(len(X))
        return ZeroModel()

    merged["hour"] = merged["ts_hour"].dt.hour
    merged["doy"]  = merged["ts_hour"].dt.dayofyear
    merged["irrad"] = 1.0 - merged["clouds"] / 100.0

    # ── daylight subset ------------------------------------------------------
    daylight = merged[merged["sun_up"] == 1].copy()
    if daylight.empty:                     # all-night data set
        class ZeroModel:
            def predict(self, X): return zeros(len(X))
        return ZeroModel()

    daylight["cf"] = (daylight["kwh"].abs() /
                      daylight["irrad"].clip(lower=0.1))

    y = log1p(daylight["cf"])
    X = daylight[["temp", "humidity", "clouds",
                  "cloud_bucket", "hour", "doy"]]

    huber = HuberRegressor(max_iter=400, epsilon=1.5).fit(X, y)
        # --- calibration: best real vs best fitted day -------------------------
    best_real = merged.groupby("ts_hour")["kwh"].sum().abs().max()

    # predict on merged daylight rows to estimate model’s best
    y_hat = huber.predict(X)
    best_pred = (
        pd.Series(y_hat, index=merged["ts_hour"])
          .groupby(merged["ts_hour"])
          .sum()
          .abs()
          .max()
    )
    calib = 1.0 if best_pred == 0 else best_real / best_pred

    # ── simple wrapper to rebuild kWh ---------------------------------------
    class Wrapper:
        def __init__(self, core, calibration):
            self.core  = core
            self.calib = calibration
        def predict(self, X_df):
            irrad = 1.0 - X_df["clouds"] / 100.0
            y_hat = expm1(self.core.predict(
                          X_df[["temp","humidity","clouds",
                                "cloud_bucket","hour","doy"]]))
            pred  = -y_hat * irrad                 # negative export
            pred[X_df["sun_up"] == 0] = 0.0        # night = 0
            return pred

    
    return Wrapper(huber, calib)


def _predict_block(df: pd.DataFrame, model, horizon: int):
    blk = df.iloc[:horizon].copy()
    blk["hour"] = blk.index.hour
    blk["doy"]  = blk.index.dayofyear

    X = blk[["temp", "humidity", "clouds",
             "cloud_bucket", "hour", "doy", "sun_up"]].ffill()

    blk["pred_kwh"] = model.predict(X) * model.calib
    blk.loc[blk["sun_up"] == 0, "pred_kwh"] = 0.0


    blk["timestamp"] = (blk.index.view("int64") // 1_000_000_000).astype(int)
    blk["pred_mwh"]  = blk["pred_kwh"] / 1000.0           # keep energy
    return blk[["pred_mwh", "timestamp"]].round(4).to_dict("records")



def make_forecasts(model, wx_hr: pd.DataFrame, wx_dl: pd.DataFrame):
    # --- raw hourly prediction (24 h) --------------------------------------
    hourly = _predict_block(wx_hr, model, 24)

    # --- compute calibration factor ----------------------------------------
    # 1) best real energy in the last 30 d
    best_real = _daily_energy(solar_history["kwh"])      # needs solar_history in scope
    # 2) best energy the model just predicted for tomorrow
    tmp = pd.DataFrame(hourly).set_index(
        pd.to_datetime([pt["timestamp"] for pt in hourly], unit="s", utc=True)
    )
    tmp["kwh"] = [pt["pred_mwh"] * 1000 for pt in hourly]
    best_pred = _daily_energy(tmp["kwh"])

    calib = 1.0 if best_pred == 0 else best_real / best_pred

    # apply calibration to every hourly point
    for pt in hourly:
        pt["pred_mwh"] *= calib


    def _expand_day(row_ts, row_vals):
        """Build a 24-hour synthetic frame for one daily forecast row."""
        base       = row_ts.floor("D")
        sunrise_s  = row_vals["sunrise"]
        sunset_s   = row_vals["sunset"]
        clouds_pct = row_vals["clouds"]
        temp_val   = row_vals["temp"]
        hum_val    = row_vals["humidity"]
    
        idx = [base + pd.Timedelta(hours=h) for h in range(24)]
    
        return pd.DataFrame(
            {
                "temp":         temp_val,
                "humidity":     hum_val,
                "clouds":       clouds_pct,
                "cloud_bucket": _cloud_bucket(clouds_pct),
                "sun_up": [
                    1 if sunrise_s <= t.timestamp() <= sunset_s else 0
                    for t in idx
                ],
            },
            index=idx,
        )

    daily_records = []
    day0_ts = pd.to_datetime(hourly[0]["timestamp"], unit="s", utc=True).floor("D")
    day0_val = round(sum(pt["pred_mwh"] for pt in hourly), 4)
    daily_records.append({"pred_mwh": day0_val, "timestamp": int(day0_ts.timestamp())})

    for ts, row in wx_dl.iloc[1:8].iterrows():  # Only 7 days instead of 16
        synth_df = _expand_day(ts, row)
        block = _predict_block(synth_df, model, 24)
        for pt in block:                                # ← ADD these two lines
            pt["pred_mwh"] *= cal
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
