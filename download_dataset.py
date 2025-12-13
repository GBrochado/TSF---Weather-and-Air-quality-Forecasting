import requests
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

# ==============================
# CONFIGURATION
# ==============================

# Porto, Portugal coordinates
LATITUDE = 41.1496
LONGITUDE = -8.6109

# Last 5 years from today
END_DATE = datetime.today().date()
START_DATE = END_DATE - relativedelta(years=5)

# Open-Meteo endpoints
WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Output CSV file
OUTPUT_CSV = "porto_weather_air_quality_5y.csv"

# ==============================
# HELPER: chunk date ranges (Open-Meteo prefers <= 1 year chunks)
# ==============================

def generate_year_chunks(start_date, end_date):
    """Yield (start, end) date strings in <=1-year chunks."""
    current_start = start_date
    while current_start < end_date:
        current_end = min(current_start + relativedelta(years=1) - timedelta(days=1), end_date)
        yield current_start.isoformat(), current_end.isoformat()
        current_start = current_end + timedelta(days=1)

# ==============================
# DOWNLOAD FUNCTIONS
# ==============================

def fetch_weather_chunk(start_date_str, end_date_str):
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "hourly": [
            "temperature_2m",
            "relative_humidity_2m",
            "dew_point_2m",
            "apparent_temperature",
            "pressure_msl",
            "surface_pressure",
            "precipitation",
            "rain",
            "snowfall",
            "cloud_cover",
            "cloud_cover_low",
            "cloud_cover_mid",
            "cloud_cover_high",
            "shortwave_radiation",
            "direct_radiation",
            "diffuse_radiation",
            "windspeed_10m",
            "windspeed_80m",
            "winddirection_10m",
            "winddirection_80m"
        ],
        "timezone": "auto"
    }
    print(f"Fetching WEATHER {start_date_str} -> {end_date_str}")
    r = requests.get(WEATHER_URL, params=params)
    r.raise_for_status()
    data = r.json()

    if "hourly" not in data or "time" not in data["hourly"]:
        print(f"⚠ No weather data for {start_date_str} -> {end_date_str}")
        return pd.DataFrame()

    df = pd.DataFrame(data["hourly"])
    return df


def fetch_air_chunk(start_date_str, end_date_str):
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "hourly": [
            "pm10",
            "pm2_5",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "aerosol_optical_depth",
            "dust",
            "uv_index",
            "uv_index_clear_sky"
        ],
        "timezone": "auto"
    }
    print(f"Fetching AIR QUALITY {start_date_str} -> {end_date_str}")
    r = requests.get(AIR_URL, params=params)
    r.raise_for_status()
    data = r.json()

    if "hourly" not in data or "time" not in data["hourly"]:
        print(f"⚠ No air-quality data for {start_date_str} -> {end_date_str}")
        return pd.DataFrame()

    df = pd.DataFrame(data["hourly"])
    return df

# ==============================
# MAIN PIPELINE
# ==============================

def main():
    print(f"Downloading data for Porto, Portugal")
    print(f"From {START_DATE} to {END_DATE}")

    weather_dfs = []
    air_dfs = []

    # Fetch in yearly chunks to be safe
    for start_str, end_str in generate_year_chunks(START_DATE, END_DATE):
        # Weather
        try:
            wdf = fetch_weather_chunk(start_str, end_str)
            if not wdf.empty:
                weather_dfs.append(wdf)
        except Exception as e:
            print(f"Error fetching weather {start_str} -> {end_str}: {e}")

        # Air quality
        try:
            adf = fetch_air_chunk(start_str, end_str)
            if not adf.empty:
                air_dfs.append(adf)
        except Exception as e:
            print(f"Error fetching air quality {start_str} -> {end_str}: {e}")

    if not weather_dfs:
        print("❌ No weather data downloaded. Exiting.")
        return
    if not air_dfs:
        print("❌ No air-quality data downloaded. Exiting.")
        return

    # Concatenate and clean
    weather_all = pd.concat(weather_dfs, ignore_index=True)
    air_all = pd.concat(air_dfs, ignore_index=True)

    # Ensure 'time' is datetime and sort
    weather_all["time"] = pd.to_datetime(weather_all["time"])
    air_all["time"] = pd.to_datetime(air_all["time"])

    weather_all = weather_all.sort_values("time").reset_index(drop=True)
    air_all = air_all.sort_values("time").reset_index(drop=True)

    # Merge on timestamp (inner join: only timestamps present in both)
    merged = pd.merge(weather_all, air_all, on="time", how="inner")

    # Optional: set time as index
    merged = merged.set_index("time")

    # Save to CSV
    merged.to_csv(OUTPUT_CSV)
    print(f"✅ Done! Saved merged dataset to: {OUTPUT_CSV}")
    print(f"Rows: {len(merged)}, Columns: {len(merged.columns)}")

if __name__ == "__main__":
    main()
