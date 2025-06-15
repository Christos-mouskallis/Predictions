#!/usr/bin/env python3
"""
solar_forecast_api.py
─────────────────────
for the B42_SOLAR meter (Arnhem, NL).

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

# ── helpers ────────────────────────────────────────────────────────────────
def _calibrate_scale(model, solar_df, wx_hist) -> float:
    """
    Compare model output with the last 48 h that have BOTH weather + meter
    data and return a multiplicative scale (clipped to 0.5–10×).
    """
    recent_wx = (
        wx_hist.tail(48)
               .reset_index()                       # “ts” becomes a column
               .assign(ts=lambda d: d["ts"].dt.floor("h"))
    )
    recent_sol = (
        solar_df.reset_index()[["ts", "kwh"]]
                 .groupby("ts", as_index=False).sum()
    )
    merged = recent_wx.merge(recent_sol, on="ts", how="inner").dropna()
    if merged.empty:
        return 1.0

    merged["hour"] = merged["ts"].dt.hour
    merged["doy"]  = merged["ts"].dt.dayofyear
    merged["sun_up"] = (merged["hour"].between(6, 18)).astype(int)  # ← NEW

    X = merged[["temp", "humidity", "clouds",
                "cloud_bucket", "hour", "doy", "sun_up"]]

    preds = model.predict(X)
    good  = np.abs(preds) > 1e-3
    if not good.any():
        return 1.0

    ratio = np.abs(merged.loc[good, "kwh"]) / np.abs(preds[good])
    return float(np.clip(np.median(ratio), 0.5, 10.0))


def _calibrate_level(model, solar_df, wx_hist, min_kw_cutoff=500):
    """
    Robust 72 h linear correction:
        y_real ≈ slope · y_pred + intercept
    Uses only rows with |kWh| > min_kw_cutoff (day-light).
    """
    recent_wx = (
        wx_hist.tail(72)
               .reset_index()
               .assign(ts=lambda d: d["ts"].dt.floor("h"))
    )
    recent_sol = (
        solar_df.reset_index()[["ts", "kwh"]]
                 .groupby("ts", as_index=False).sum()
    )
    merged = recent_wx.merge(recent_sol, on="ts", how="inner").dropna()
    if merged.empty:
        return 1.0, 0.0

    merged["hour"] = merged["ts"].dt.hour
    merged["doy"]  = merged["ts"].dt.dayofyear
    merged["sun_up"] = (merged["hour"].between(6, 18)).astype(int)  # ← NEW

    X = merged[["temp", "humidity", "clouds",
                "cloud_bucket", "hour", "doy", "sun_up"]]

    preds = model.predict(X)
    mask  = np.abs(merged["kwh"]) > min_kw_cutoff
    if not mask.any():
        return 1.0, 0.0

    y = merged.loc[mask, "kwh"].to_numpy()
    x = preds[mask]
    slope = float(np.clip(np.median(y / np.where(x == 0, np.nan, x)), 0.2, 5.0))
    intercept = float(np.median(y - slope * x))
    return slope, intercept



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
    Returns
        wx_hist – 30-day historic hourly    (temp, humidity, clouds,
                                             cloud_bucket, sun_up)
        wx_hr   – next-48 h hourly forecast (same cols)
        wx_dl   – 16-day daily forecast     (temp, humidity, clouds,
                                             sunrise, sunset)
    All three frames GUARANTEE both columns:
        cloud_bucket ∈ {0,1,2,3}   and   sun_up ∈ {0,1}
    """
    now = dt.datetime.now(timezone.utc)

    # ── 48 h HOURLY FORECAST ───────────────────────────────────────────────
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
              sun_up   = lambda d: (
                  d["sys"].apply(lambda s: 1 if s.get("pod", "n") == "d" else 0)
                  if "sys" in d.columns
                  else d["weather"].apply(
                      lambda w: 1 if w[0]["icon"].endswith("d") else 0
                  )
              ),
          )
          .set_index("ts")[["temp", "humidity", "clouds", "sun_up"]]
    )
    wx_hr["cloud_bucket"] = wx_hr["clouds"].apply(_cloud_bucket)
    wx_hr["sun_up"]       = wx_hr["sun_up"].fillna(0).astype(int)

    # ── 30 d HOURLY HISTORY ────────────────────────────────────────────────
    hist_rows = []
    for back in range(1, HIST_DAYS + 1):
        day = now - dt.timedelta(days=back)
        start_ts = int(day.replace(hour=0, minute=0, second=0).timestamp())
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
            pod    = itm.get("sys", {}).get("pod")            # 'd' or 'n'
            hist_rows.append(
                dict(
                    ts           = _dt_utc(itm["dt"]),
                    temp         = itm["main"]["temp"],
                    humidity     = itm["main"]["humidity"],
                    clouds       = clouds,
                    cloud_bucket = _cloud_bucket(clouds),
                    sun_up       = 1 if pod == "d" else 0,
                )
            )

    wx_hist = (
        pd.DataFrame(hist_rows)
          .set_index("ts")
          .sort_index()
          .drop_duplicates()
    )

    # ── 16 d DAILY FORECAST ────────────────────────────────────────────────
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
from numpy import log1p, expm1, clip
import numpy as np

from sklearn.linear_model import HuberRegressor
from numpy import log1p, expm1, clip
import numpy as np

def train_model(solar: pd.DataFrame, wx_hist: pd.DataFrame):
    """Return an object with .predict(df) that outputs kWh (negative = export)."""
    # ---------- guarantee required weather columns -------------------------
    for col in ("sun_up", "cloud_bucket"):
        if col not in wx_hist.columns:
            if col == "sun_up":
                wx_hist[col] = 0
            else:
                wx_hist[col] = wx_hist["clouds"].apply(_cloud_bucket)

    # ---------- merge solar + weather, fit robust model --------------------
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
    merged["hour"] = merged["ts_hour"].dt.hour
    merged["doy"]  = merged["ts_hour"].dt.dayofyear

    # daylight rows for capacity fit
    daylight = merged[merged["sun_up"] == 1].copy()
    if daylight.empty:
        med = (merged.groupby("hour")["kwh"].median()
                        .reindex(range(24), fill_value=0.0).abs() * 1.10)

        class MedianModel:
            def __init__(self, med): self.med = med.values
            def predict(self, X):
                out = -self.med[X["hour"].to_numpy()] * (1 - X["clouds"]/100)
                out[X["sun_up"] == 0] = 0.0
                return out
        return MedianModel(med)

    daylight["irrad"]  = 1.0 - daylight["clouds"] / 100.0
    daylight["y_norm"] = daylight["kwh"].abs() / daylight["irrad"].clip(lower=0.1)
    y_cap = np.log1p(daylight["y_norm"])
    X_cap = daylight[["temp", "humidity", "clouds",
                      "cloud_bucket", "hour", "doy"]]

    huber = HuberRegressor(max_iter=500, epsilon=1.5).fit(X_cap, y_cap)

    hour_cap = (daylight.groupby("hour")["kwh"].quantile(0.95)
                          .reindex(range(24), fill_value=0.0).abs().to_numpy())

    class Wrapper:
        def __init__(self, core, cap):
            self.core = core
            self.cap  = cap      # length-24 ndarray

        def predict(self, X_df):
            irrad = 1.0 - X_df["clouds"] / 100.0
            cap   = self.cap[X_df["hour"].to_numpy()]
            y_hat = expm1(self.core.predict(
                          X_df[["temp","humidity","clouds",
                                "cloud_bucket","hour","doy"]]))
            pred  = -y_hat * irrad
            pred  = clip(pred, -cap, 0)
            pred[X_df["sun_up"] == 0] = 0.0
            return pred

    return Wrapper(huber, hour_cap)
# ────────────────────────────────────────────────────────────────────────────
def _predict_block(
        df: pd.DataFrame,
        model,
        horizon: int,
        *,
        slope: float = 1.0,
        intercept: float = 0.0,
):
    """
    • Takes the first *horizon* rows of *df*.
    • Ensures *all required columns* exist, creating safe defaults when missing.
    • Runs model → kWh, applies (slope, intercept) post-correction.
    • Returns [{"pred_mwh", "timestamp"}, …].
    """
    blk = df.iloc[:horizon].copy()

    # ---------- harden every expected column -------------------------------
    defaults = {
        "temp":         15.0,
        "humidity":     70.0,
        "clouds":       50.0,
        "cloud_bucket": 2,
        "sun_up":       0,
    }
    for col, default in defaults.items():
        if col not in blk:
            blk[col] = default

    # cloud_bucket can be re-derived safely
    blk["cloud_bucket"] = blk["clouds"].apply(_cloud_bucket)

    # time features
    blk.index = pd.to_datetime(blk.index, utc=True)
    blk["hour"] = blk.index.hour
    blk["doy"]  = blk.index.dayofyear

    # ------------------ model → kWh ----------------------------------------
    X = blk[["temp", "humidity", "clouds",
             "cloud_bucket", "hour", "doy", "sun_up"]]
    blk["pred_kwh"] = model.predict(X) * slope + intercept

    # kWh → MWh for the API
    blk["pred_mwh"] = blk["pred_kwh"] / 1_000.0
    blk["timestamp"] = (blk.index.view("int64") // 1_000_000_000).astype(int)

    return blk[["pred_mwh", "timestamp"]].round(4).to_dict("records")
# ────────────────────────────────────────────────────────────────────────────

def make_forecasts(model,
                   wx_hr: pd.DataFrame,
                   wx_dl: pd.DataFrame,
                   *,
                   scale: float   = 1.0,
                   slope: float   = 1.0,
                   intercept: float = 0.0):
    eff_slope = slope * scale    # merge both corrections

    # ----- 24 × 1 h ---------------------------------------------------------
    hourly = _predict_block(wx_hr, model, 24,
                            slope=eff_slope, intercept=intercept)

    # helper – expand one daily row into 24 synthetic hours
    def _expand_day(row_ts, row_vals):
        base = row_ts.floor("D")
        idx  = [base + pd.Timedelta(hours=h) for h in range(24)]
        return pd.DataFrame(
            {
                "temp":         row_vals["temp"],
                "humidity":     row_vals["humidity"],
                "clouds":       row_vals["clouds"],
                "cloud_bucket": _cloud_bucket(row_vals["clouds"]),
                "sun_up": [
                    1 if row_vals["sunrise"] <= t.timestamp() <= row_vals["sunset"] else 0
                    for t in idx
                ],
            },
            index=idx,
        )

    # ----- 7-day horizon ----------------------------------------------------
    daily = []
    d0_ts  = pd.to_datetime(hourly[0]["timestamp"], unit="s", utc=True).floor("D")
    d0_val = round(sum(pt["pred_mwh"] for pt in hourly), 4)
    daily.append({"pred_mwh": d0_val, "timestamp": int(d0_ts.timestamp())})

    for ts, row in wx_dl.iloc[1:8].iterrows():          # next 7 days
        synth = _expand_day(ts, row)
        block = _predict_block(synth, model, 24,
                               slope=eff_slope, intercept=intercept)
        daily.append({
            "pred_mwh": round(sum(pt["pred_mwh"] for pt in block), 4),
            "timestamp": int(ts.floor("D").timestamp())
        })

    return hourly, daily




@app.route("/forecast")
def forecast():
    now = time.time()
    if now - _cache["ts"] < CACHE_TTL:
        return jsonify(_cache["payload"])

    try:
        solar                 = get_solar_history()
        wx_hist, wx_hr, wx_dl = get_weather()

        model  = train_model(solar, wx_hist)
        scale  = _calibrate_scale(model, solar, wx_hist)
        slope, intercept = _calibrate_level(model, solar, wx_hist)

        hourly, daily_7 = make_forecasts(
            model, wx_hr, wx_dl,
            scale=scale, slope=slope, intercept=intercept
        )

        payload = {
            "generated_utc": int(now),
            "hourly":  hourly,
            "daily_7": daily_7,
        }
        _cache.update(ts=now, payload=payload)
        return jsonify(payload)

    except Exception as exc:
        return Response(
            json.dumps({"error": str(exc)}), status=500,
            mimetype="application/json"
        )



if __name__ == "__main__":
    print("→  http://localhost:8000/forecast")
    app.run(host="0.0.0.0", port=8000)
