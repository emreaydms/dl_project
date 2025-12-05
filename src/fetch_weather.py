import pandas as pd
import requests
from tqdm import tqdm
COUNTRY_COORDS = {
    "RO": (44.4268, 26.1025),  # Bucharest
    "HU": (47.4979, 19.0402),  # Budapest
    "DE": (52.5200, 13.4050),  # Berlin
}

def fetch_weather_chunk(country, start, end):
    lat, lon = COUNTRY_COORDS[country]

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={start}&end_date={end}"
        "&hourly=temperature_2m,relative_humidity_2m,cloudcover"
        "&timezone=UTC"
    )

    res = requests.get(url).json()
    if "hourly" not in res:
        print(f"No data for {country} {start}→{end}")
        return None

    df = pd.DataFrame({
        "timestamp": res["hourly"]["time"],
        "temp_C": res["hourly"]["temperature_2m"],
        "humidity_pct": res["hourly"]["relative_humidity_2m"],
        "cloud_pct": res["hourly"]["cloudcover"],
    })

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["country"] = country

    return df


def fetch_weather_full(country, start_year, end_year):
    dfs = []
    for year in range(start_year, end_year + 1):
        start = f"{year}-01-01"
        end   = f"{year}-12-31"
        df = fetch_weather_chunk(country, start, end)
        if df is not None:
            dfs.append(df)

    return pd.concat(dfs).sort_values("timestamp")


if __name__ == "__main__":
    load = pd.read_csv("../data_processed/load_europe.csv", parse_dates=["timestamp"])
    start_year = load["timestamp"].min().year
    end_year   = load["timestamp"].max().year

    countries = load["country"].unique()
    all_weather = []

    for c in tqdm(countries):
        df = fetch_weather_full(c, start_year, end_year)
        all_weather.append(df)

    weather = pd.concat(all_weather).reset_index(drop=True)
    weather.to_csv("../data_raw/weather_data.csv", index=False)

    print("Weather saved to data_raw/weather_data.csv")
    print(weather.head())
