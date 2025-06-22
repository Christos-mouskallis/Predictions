#!/usr/bin/env python3
"""
solar_forecast_api.py
─────────────────────
Fixed version for the B42_SOLAR meter (Arnhem, NL).
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
from sklearn.linear_model import HuberRegressor
from apscheduler.schedulers.background import BackgroundScheduler


# ── configuration ────────────────────────────────────────────────────────────
LAT, LON   = 52.0, 5.87                      # Arnhem, NL
SOLAR_ID   = "B42_SOLAR"
SOLAR_URL  = (
    "https://dashboard.hedge-iot.labs.vu.nl/ui_api/measurements?"
    f"meter_id={SOLAR_ID}&latest=false&raw=false&use_cache=true"
)

# OpenWeather – Developer plan
OWM_KEY = os.getenv("OWM_KEY", "67d4dbfce7083e9195b41a0f4dafba74")
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
    """Parse *scalar or array* into tz-aware UTC datetime(s)."""
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

def _is_daylight_hour(hour: int, lat: float = LAT) -> bool:
    """Simple daylight detection based on hour and latitude."""
    
    if 7 <= hour <= 18:  # Core daylight hours
        return True
    elif 6 <= hour <= 19:  # Extended daylight hours
        return True
    return False

# ── solar data ───────────────────────────────────────────────────────────────
def get_solar_history(days: int = HIST_DAYS) -> pd.DataFrame:
    try:
        now = dt.datetime.now(timezone.utc)
        start = now - dt.timedelta(days=days)

        # Use seconds for the API call (not milliseconds)
        url = f"{SOLAR_URL}&start_ts={int(start.timestamp())}"
        print(f" Fetching solar data from: {start} to {now}")
        print(f" API URL: {url}")
        
        rec = requests.get(url, timeout=15).json()
        records = rec.get("measurements") if isinstance(rec, dict) else rec
        if not isinstance(records, list):
            raise ValueError("Unexpected solar API response shape")

        print(f" Received {len(records)} raw records from API")
        
        # Debug: Show sample of raw data
        if records:
            sample = records[0]
            print(f" Sample record: timestamp={sample.get('timestamp')}, val={sample.get('val')}")
            
            # Check if timestamp is in seconds or milliseconds
            sample_ts = int(sample.get('timestamp', 0))
            sample_dt_sec = pd.to_datetime(sample_ts, unit='s', utc=True)
            sample_dt_ms = pd.to_datetime(sample_ts, unit='ms', utc=True)
            print(f" Sample timestamp as seconds: {sample_dt_sec}")
            print(f" Sample timestamp as milliseconds: {sample_dt_ms}")

        df_raw = pd.DataFrame(records)[["timestamp", "val"]].rename(columns={"timestamp": "ts", "val": "kwh"})
        print(f" Raw DataFrame shape: {df_raw.shape}")
        
        # Convert timestamp - API uses seconds, not milliseconds
        df_raw["ts"] = pd.to_numeric(df_raw["ts"], errors="coerce")
        df_raw["ts"] = pd.to_datetime(df_raw["ts"], unit="s", utc=True)
        
        # Convert values to numeric (from string)
        df_raw["kwh"] = pd.to_numeric(df_raw["kwh"], errors="coerce") / 1000.0  # Convert Wh to kWh
        
        # Remove invalid data
        df_clean = df_raw.dropna(subset=["ts", "kwh"])
        print(f" After cleaning: {len(df_clean)} records")
        
        if df_clean.empty:
            print(" No valid data after cleaning")
            return pd.DataFrame(columns=['kwh'])
        
        # Show timestamp range
        print(f" Data timestamp range: {df_clean['ts'].min()} to {df_clean['ts'].max()}")
        
        # Set index and sort
        df = df_clean.set_index("ts").sort_index()
        
        # Filter for the requested time range
        df_filtered = df.loc[start:now]
        print(f" After time filtering ({start} to {now}): {len(df_filtered)} records")
        
        if df_filtered.empty:
            print(" No data in requested time range")
            # Try to get any available data for debugging
            print(f" Available data range: {df['kwh'].index.min()} to {df['kwh'].index.max()}")
            # Use all available data if within reasonable bounds
            if len(df) > 0:
                recent_cutoff = now - dt.timedelta(days=90)  # Try last 90 days
                df_filtered = df.loc[recent_cutoff:now]
                print(f" Using extended range (90 days): {len(df_filtered)} records")
        
        # Resample to hourly
        df_hourly = df_filtered.resample("1h").sum(min_count=1).fillna(0.0)
        
        # Process solar data - negative values represent generation (export to grid)
        print(f" Raw solar data stats:")
        print(f"   Records: {len(df_hourly)}")
        print(f"   Min value: {df_hourly['kwh'].min():.4f}")
        print(f"   Max value: {df_hourly['kwh'].max():.4f}")
        print(f"   Mean value: {df_hourly['kwh'].mean():.4f}")
        print(f"   Negative values: {(df_hourly['kwh'] < 0).sum()}")
        print(f"   Positive values: {(df_hourly['kwh'] > 0).sum()}")
        
        # For solar generation, negative values typically mean export to grid (generation)
        # Convert negative values to positive (solar generation)
        df_hourly['kwh'] = df_hourly['kwh'].abs()
        
        print(f" Processed solar data: {len(df_hourly)} records, generation sum: {df_hourly['kwh'].sum():.4f} kWh")
        return df_hourly
        
    except Exception as e:
        print(f"Error getting solar history: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame(columns=['kwh'])

# ── weather data ─────────────────────────────────────────────────────────────
def get_weather() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns: wx_hist, wx_hr, wx_dl"""
    try:
        now = dt.datetime.now(timezone.utc)

        # ---------- 48 h HOURLY forecast ----------------------------------------
        fc_hr = requests.get(
            HOURLY_FC,
            params=dict(lat=LAT, lon=LON, units="metric", appid=OWM_KEY),
            timeout=15,
        ).json()

        wx_hr_data = []
        for item in fc_hr["list"][:48]:
            ts = pd.to_datetime(item["dt"], unit="s", utc=True)
            temp = item["main"]["temp"]
            humidity = item["main"]["humidity"]
            clouds = item["clouds"]["all"]
            
            # Better sun detection - use both icon and hour
            sun_up = 0
            if "sys" in item and item["sys"].get("pod") == "d":
                sun_up = 1
            elif "weather" in item and item["weather"][0]["icon"].endswith("d"):
                sun_up = 1
            elif _is_daylight_hour(ts.hour):
                sun_up = 1
                
            wx_hr_data.append({
                "ts": ts,
                "temp": temp,
                "humidity": humidity,
                "clouds": clouds,
                "sun_up": sun_up,
                "cloud_bucket": _cloud_bucket(clouds)
            })

        wx_hr = pd.DataFrame(wx_hr_data).set_index("ts")

        # ---------- 30-day HOURLY history ---------------------------------------
        hist_rows = []
        for back in range(1, HIST_DAYS + 1):
            try:
                day = now - dt.timedelta(days=back)
                start_ts = int(day.replace(hour=0, minute=0, second=0).timestamp())
                end_ts = int(day.replace(hour=23, minute=59, second=59).timestamp())

                r = requests.get(
                    HIST_URL,
                    params=dict(lat=LAT, lon=LON, type="hour",
                                start=start_ts, end=end_ts,
                                units="metric", appid=OWM_KEY),
                    timeout=15,
                )
                
                if r.status_code == 200:
                    data = r.json()
                    for itm in data.get("list", []):
                        ts = _dt_utc(itm["dt"])
                        clouds = itm["clouds"]["all"]
                        
                        # Better sun detection for historical data
                        sun_up = 0
                        if itm.get("sys", {}).get("pod") == "d":
                            sun_up = 1
                        elif _is_daylight_hour(ts.hour):
                            sun_up = 1
                            
                        hist_rows.append({
                            "ts": ts,
                            "temp": itm["main"]["temp"],
                            "humidity": itm["main"]["humidity"],
                            "clouds": clouds,
                            "cloud_bucket": _cloud_bucket(clouds),
                            "sun_up": sun_up,
                        })
            except Exception as e:
                print(f"Error getting weather history for day {back}: {e}")
                continue

        wx_hist = (
            pd.DataFrame(hist_rows)
            .set_index("ts")
            .sort_index()
            .drop_duplicates()
        )

        # ---------- 16-day DAILY forecast ---------------------------------------
        fc_dl = requests.get(
            DAILY_FC,
            params=dict(lat=LAT, lon=LON, cnt=16, units="metric", appid=OWM_KEY),
            timeout=15,
        ).json()

        wx_dl = (
            pd.DataFrame(fc_dl["list"])
            .assign(
                ts = lambda d: pd.to_datetime(d["dt"], unit="s", utc=True) + pd.Timedelta(hours=12),
                temp = lambda d: d["temp"].apply(lambda t: t["max"]),
                humidity = lambda d: d["humidity"],
                clouds = lambda d: d["clouds"],
            )
            .set_index("ts")[["temp", "humidity", "clouds", "sunrise", "sunset"]]
            .sort_index()
        )

        return wx_hist, wx_hr, wx_dl
        
    except Exception as e:
        print(f"Error getting weather data: {e}")
        # Return empty dataframes with correct structure
        empty_wx = pd.DataFrame(columns=["temp", "humidity", "clouds", "cloud_bucket", "sun_up"])
        empty_daily = pd.DataFrame(columns=["temp", "humidity", "clouds", "sunrise", "sunset"])
        return empty_wx, empty_wx, empty_daily

# ── model ────────────────────────────────────────────────────────────────────
def train_model(solar: pd.DataFrame, wx_hist: pd.DataFrame):
    """Return a robust production model."""
    print("🔹 Training model...")
    print(f"Solar data shape: {solar.shape}")
    print(f"Weather history shape: {wx_hist.shape}")

    if solar.empty or wx_hist.empty:
        print(" Insufficient data for training, using fallback model")
        return create_fallback_model()

    # Prepare solar data - now includes all generation data (originally negative values)
    solar_hr = solar.reset_index()
    solar_hr["ts_hour"] = solar_hr["ts"].dt.floor("h")
    
    # Filter for meaningful generation (>0.001 kWh to avoid noise)
    meaningful_gen = solar_hr[solar_hr["kwh"] > 0.001]
    
    print(f" Total solar records: {len(solar_hr)}")
    print(f" Meaningful generation records: {len(meaningful_gen)}")
    print(f" Generation stats: min={solar_hr['kwh'].min():.4f}, max={solar_hr['kwh'].max():.4f}, mean={meaningful_gen['kwh'].mean():.4f}")

    # Use all solar data for training (not just positive)
    solar_for_training = solar_hr.copy()

    # Aggregate weather data by hour
    if not wx_hist.empty:
        wx_hist_agg = (
            wx_hist.reset_index()
                   .assign(ts_hour=lambda d: d["ts"].dt.floor("h"))
                   .groupby("ts_hour")
                   .agg(
                       temp=("temp", "mean"),
                       humidity=("humidity", "mean"),  
                       clouds=("clouds", "mean"),
                       cloud_bucket=("cloud_bucket", "max"),
                       sun_up=("sun_up", "max"),
                   )
                   .reset_index()
        )
    else:
        print(" No weather history data")
        return create_fallback_model()

    # Merge solar and weather data
    merged = solar_for_training.merge(wx_hist_agg, on="ts_hour", how="inner").dropna()
    
    if merged.empty:
        print(" No merged data available")
        return create_fallback_model()
        
    # Add time features
    merged["hour"] = merged["ts_hour"].dt.hour
    merged["doy"] = merged["ts_hour"].dt.dayofyear
    merged["month"] = merged["ts_hour"].dt.month
    merged["irrad"] = (100 - merged["clouds"]) / 100.0  # Solar irradiance proxy

    print(f" Merged training data: {len(merged)} rows")
    print(f"Hour distribution:\n{merged['hour'].value_counts().sort_index()}")

    # For training, focus on periods with potential for generation
    # Include all data but weight daylight hours more heavily
    training_data = merged.copy()
    
    # Create daylight subset for validation
    daylight = merged[
        (merged["sun_up"] == 1) | 
        (merged["hour"].between(6, 18)) |
        (merged["kwh"] > 0.001)  # Any meaningful generation
    ].copy()

    print(f" Daylight training subset: {len(daylight)} rows")
    print(f" Total training data: {len(training_data)} rows")

    if len(training_data) < 10:
        print(" Insufficient training data")
        return create_fallback_model()

    # Prepare features and target
    feature_cols = ["temp", "humidity", "clouds", "cloud_bucket", "hour", "doy", "month", "irrad"]
    X = training_data[feature_cols].fillna(method='ffill').fillna(0)
    y = training_data["kwh"]

    # Train model
    try:
        model = GradientBoostingRegressor(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
            subsample=0.8
        )
        model.fit(X, y)
        
        # Calculate hourly capacity using daylight data for realistic limits
        if len(daylight) > 0:
            hour_capacity = (
                daylight.groupby("hour")["kwh"]
                .quantile(0.95)
                .reindex(range(24), fill_value=0.0)
            ) * 1.3  # Add 30% buffer for peak conditions
        else:
            # Fallback capacity calculation using all data
            hour_capacity = (
                training_data.groupby("hour")["kwh"]
                .quantile(0.95)
                .reindex(range(24), fill_value=0.0)
            ) * 1.3

        print(" Hourly capacity limits:")
        for h, cap in hour_capacity.items():
            if cap > 0.001:
                print(f"  Hour {h:2d}: {cap:.4f} kWh")

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(feature_cols, model.feature_importances_))
            print(" Feature importance:")
            for feat, importance in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
                print(f"  {feat}: {importance:.3f}")

        return ModelWrapper(model, hour_capacity, feature_cols)
        
    except Exception as e:
        print(f" Model training failed: {e}")
        import traceback
        traceback.print_exc()
        return create_fallback_model()

def create_fallback_model():
    """Create a simple fallback model based on typical solar patterns."""
    print(" Creating fallback model...")
    
    # Typical solar generation pattern (normalized)
    solar_pattern = np.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 0-5: night
        0.01, 0.05, 0.15, 0.35, 0.60, 0.80,  # 6-11: morning
        0.95, 1.0, 0.95, 0.85, 0.65, 0.40,   # 12-17: peak & afternoon
        0.20, 0.05, 0.01, 0.0, 0.0, 0.0      # 18-23: evening & night
    ])
    
    # Assume 3kW peak system (typical residential)
    peak_capacity = 3.0  # kW
    
    class FallbackModel:
        def __init__(self, pattern, peak_cap):
            self.pattern = pattern
            self.peak_cap = peak_cap
            
        def predict(self, X_df):
            predictions = []
            for _, row in X_df.iterrows():
                hour = int(row["hour"])
                cloud_factor = (100 - row["clouds"]) / 100.0
                base_gen = self.pattern[hour] * self.peak_cap
                pred = base_gen * cloud_factor * row.get("sun_up", 1)
                predictions.append(max(0, pred))  # Ensure positive
            return np.array(predictions)
    
    return FallbackModel(solar_pattern, peak_capacity)

class ModelWrapper:
    def __init__(self, model, capacity, feature_cols):
        self.model = model
        self.capacity = capacity
        self.feature_cols = feature_cols

    def predict(self, X_df):
        try:
            # Ensure all required features are present
            X = X_df[self.feature_cols].fillna(method='ffill').fillna(0)
            predictions = self.model.predict(X)
            
            # Apply hourly capacity limits and ensure positive values
            final_predictions = []
            for i, (_, row) in enumerate(X_df.iterrows()):
                hour = int(row["hour"])
                pred = max(0, predictions[i])  # Ensure positive
                
                # Apply capacity limit
                if hour in self.capacity:
                    pred = min(pred, self.capacity[hour])
                
                # Zero out nighttime hours (but be less restrictive)
                if not row.get("sun_up", 0) and hour not in range(5, 20):
                    pred = 0.0
                    
                final_predictions.append(pred)
                
            return np.array(final_predictions)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return np.zeros(len(X_df))

# ── prediction helpers ───────────────────────────────────────────────────────
def _predict_block(df: pd.DataFrame, model, horizon: int):
    """Predict solar generation for a block of weather data."""
    blk = df.iloc[:horizon].copy()
    
    # Add time features
    blk["hour"] = blk.index.hour
    blk["doy"] = blk.index.dayofyear
    blk["month"] = blk.index.month
    blk["irrad"] = (100 - blk["clouds"]) / 100.0

    # Make predictions
    try:
        predictions = model.predict(blk)
        blk["pred_kwh"] = predictions * 0.4
    except Exception as e:
        print(f"Prediction error: {e}")
        blk["pred_kwh"] = 0.0

    # Prepare output
    blk["timestamp"] = (blk.index.view("int64") // 1_000_000_000).astype(int)
    blk["pred_mwh"] = blk["pred_kwh"] / 1000.0  # Convert to MWh
    
    return blk[["pred_mwh", "timestamp"]].round(4).to_dict("records")

def make_forecasts(model, wx_hr: pd.DataFrame, wx_dl: pd.DataFrame):
    """Generate hourly and daily forecasts."""
    print("🔹 Making forecasts...")
    
    if wx_hr.empty:
        print(" No hourly weather data")
        return [], []
    
    print(f" Weather data shape: {wx_hr.shape}")
    print(f"Sun up distribution: {wx_hr['sun_up'].value_counts()}")

    # 24-hour forecast
    hourly = _predict_block(wx_hr, model, 24)
    print(f" Generated {len(hourly)} hourly predictions")
    
    # Print some sample predictions for debugging
    total_daily = sum(h["pred_mwh"] for h in hourly)
    print(f" Total daily forecast: {total_daily:.4f} MWh")
    
    for i, h in enumerate(hourly[:12]):  # Show first 12 hours
        ts = pd.to_datetime(h["timestamp"], unit="s", utc=True)
        print(f"  {ts.strftime('%H:%M')}: {h['pred_mwh']:.4f} MWh")

    # Daily forecasts
    daily_records = []
    
    # Day 0 (today)
    day0_ts = pd.to_datetime(hourly[0]["timestamp"], unit="s", utc=True).floor("D")
    day0_val = round(sum(pt["pred_mwh"] for pt in hourly), 4)
    daily_records.append({"pred_mwh": day0_val, "timestamp": int(day0_ts.timestamp())})

    # Next 6 days from daily weather forecast
    if not wx_dl.empty:
        for ts, row in wx_dl.iloc[1:8].iterrows():
            try:
                synth_df = _expand_daily_to_hourly(ts, row)
                if not synth_df.empty:
                    block = _predict_block(synth_df, model, 24)
                    day_val = round(sum(pt["pred_mwh"] for pt in block), 4)
                    daily_records.append({
                        "pred_mwh": day_val, 
                        "timestamp": int(ts.floor("D").timestamp())
                    })
                    print(f" {ts.date()}: {day_val:.4f} MWh")
            except Exception as e:
                print(f"Error processing daily forecast for {ts}: {e}")
                continue

    return hourly, daily_records

def _expand_daily_to_hourly(row_ts, row_vals):
    """Expand daily weather forecast to hourly data."""
    try:
        base = row_ts.floor("D")
        sunrise_ts = pd.to_datetime(row_vals["sunrise"], unit="s", utc=True)
        sunset_ts = pd.to_datetime(row_vals["sunset"], unit="s", utc=True)
        
        sunrise_hour = sunrise_ts.hour
        sunset_hour = sunset_ts.hour
        
        idx = [base + pd.Timedelta(hours=h) for h in range(24)]
        
        hourly_data = []
        for i, ts in enumerate(idx):
            sun_up = 1 if sunrise_hour <= ts.hour <= sunset_hour else 0
            
            hourly_data.append({
                "temp": row_vals["temp"],
                "humidity": row_vals["humidity"],
                "clouds": row_vals["clouds"],
                "cloud_bucket": _cloud_bucket(row_vals["clouds"]),
                "sun_up": sun_up,
            })
        
        return pd.DataFrame(hourly_data, index=idx)
        
    except Exception as e:
        print(f"Error expanding daily to hourly: {e}")
        return pd.DataFrame()

# ── Flask endpoint ───────────────────────────────────────────────────────────
@app.route("/forecast")
def forecast():
    now = time.time()
    if now - _cache["ts"] < CACHE_TTL:
        print(" Returning cached forecast")
        return jsonify(_cache["payload"])

    try:
        print(" Generating new forecast...")
        solar = get_solar_history()
        wx_hist, wx_hr, wx_dl = get_weather()
        
        model = train_model(solar, wx_hist)
        hourly, daily_7 = make_forecasts(model, wx_hr, wx_dl)

        payload = {
            "generated_utc": int(now),
            "hourly": hourly,
            "daily_7": daily_7,
            "source": "model" if hasattr(model, 'model') else "fallback"
        }
        
        _cache.update(ts=now, payload=payload)
        print(" Forecast generated successfully")
        return jsonify(payload)

    except Exception as exc:
        print(f" Forecast error: {exc}")
        import traceback
        traceback.print_exc()
        
        return Response(
            json.dumps({"error": str(exc)}), 
            status=500, 
            mimetype="application/json"
        )

@app.route("/debug")
def debug():
    """Debug endpoint to check data availability."""
    try:
        solar = get_solar_history()
        wx_hist, wx_hr, wx_dl = get_weather()
        
        # Convert dataframes to JSON-serializable format
        solar_sample = {}
        if not solar.empty:
            sample_df = solar.head(24)  # Show more samples
            solar_sample = {
                "timestamps": [ts.isoformat() for ts in sample_df.index],
                "values": sample_df["kwh"].tolist(),
                "total_generation": float(solar["kwh"].sum()),
                "max_hourly": float(solar["kwh"].max()),
                "avg_positive": float(solar[solar["kwh"] > 0]["kwh"].mean()) if len(solar[solar["kwh"] > 0]) > 0 else 0
            }
        
        weather_sample = {}
        if not wx_hr.empty:
            sample_df = wx_hr.head()
            weather_sample = {
                "timestamps": [ts.isoformat() for ts in sample_df.index],
                "temp": sample_df["temp"].tolist(),
                "humidity": sample_df["humidity"].tolist(),
                "clouds": sample_df["clouds"].tolist(),
                "sun_up": sample_df["sun_up"].tolist()
            }
        
        debug_info = {
            "solar_records": len(solar),
            "solar_generation_records": len(solar[solar["kwh"] > 0]) if not solar.empty else 0,
            "solar_total_generation": float(solar["kwh"].sum()) if not solar.empty else 0,
            "solar_max_generation": float(solar["kwh"].max()) if not solar.empty else 0,
            "solar_avg_generation": float(solar[solar["kwh"] > 0]["kwh"].mean()) if not solar.empty and len(solar[solar["kwh"] > 0]) > 0 else 0,
            "weather_history_records": len(wx_hist),
            "weather_forecast_records": len(wx_hr),
            "daily_forecast_records": len(wx_dl),
            "solar_sample": solar_sample,
            "weather_sample": weather_sample,
            "sun_up_hours_forecast": int(wx_hr["sun_up"].sum()) if not wx_hr.empty else 0,
            "avg_cloud_cover": float(wx_hr["clouds"].mean()) if not wx_hr.empty else 0,
            "data_ready_for_training": not solar.empty and not wx_hist.empty and len(solar) > 0
        }
        
        return jsonify(debug_info)
        
    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc()
        })

if __name__ == "__main__":
    print(" Solar Forecast API Starting...")
    print("→  http://localhost:8000/forecast")
    print("→  http://localhost:8000/debug")
    app.run(host="0.0.0.0", port=8000, debug=True)
