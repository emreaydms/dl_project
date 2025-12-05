import pandas as pd

def merge_data():
    load = pd.read_csv("../data_processed/load_europe.csv", parse_dates=["timestamp"])
    weather = pd.read_csv("../data_raw/weather_data.csv", parse_dates=["timestamp"])

    if hasattr(weather['timestamp'].dt, 'tz'):
        weather['timestamp'] = weather['timestamp'].dt.tz_localize(None)

    if hasattr(load['timestamp'].dt, 'tz'):
        load['timestamp'] = load['timestamp'].dt.tz_localize(None)

    df = pd.merge(load, weather, on=["timestamp", "country"], how="inner")

    df = df.sort_values(["country", "timestamp"]).reset_index(drop=True)
    df.to_csv("../data_processed/merged_raw.csv", index=False)
    print("Saved: data_processed/merged_raw.csv")
    print(df.head())


if __name__ == "__main__":
    merge_data()
