import pandas as pd
import numpy as np
import holidays

HOLIDAY_MAP = {
    "DE": "DE",
    "HU": "HU",
    "RO": "RO"
}

def add_calendar_features(df):
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["month"] = df["timestamp"].dt.month
    df["dayofyear"] = df["timestamp"].dt.dayofyear
    df["season"] = ((df["month"] % 12) // 3).astype(int)
    return df


def add_hdd_cdd(df, t_heat=18, t_cool=22):
    df["HDD"] = np.maximum(0, t_heat - df["temp_C"])
    df["CDD"] = np.maximum(0, df["temp_C"] - t_cool)
    return df


def add_holiday_flag(df):
    df["holiday"] = 0
    for country, code in HOLIDAY_MAP.items():
        mask = df["country"] == country
        years = range(df["timestamp"].dt.year.min(), df["timestamp"].dt.year.max() + 1)
        country_holidays = holidays.country_holidays(code, years=years)
        df.loc[mask, "holiday"] = df.loc[mask, "timestamp"].dt.date.astype(str).isin(
            country_holidays
        ).astype(int)
    return df


def add_lags(df, lags=[1, 24, 48, 72, 168]):
    for lag in lags:
        df[f"load_lag_{lag}"] = df.groupby("country")["load_MW"].shift(lag)
    return df


def add_rolling_stats(df):
    df["roll_mean_24"] = df.groupby("country")["load_MW"].transform(lambda x: x.rolling(24).mean())
    df["roll_std_24"]  = df.groupby("country")["load_MW"].transform(lambda x: x.rolling(24).std())
    return df


def process_features():
    df = pd.read_csv("../data_processed/merged_raw.csv", parse_dates=["timestamp"])
    df = df.sort_values(["country", "timestamp"]).reset_index(drop=True)

    df = add_calendar_features(df)
    df = add_hdd_cdd(df)
    df = add_holiday_flag(df)
    df = add_lags(df)
    df = add_rolling_stats(df)

    df = df.dropna().reset_index(drop=True)

    df.to_csv("../data_processed/features_full.csv", index=False)
    print("Saved: data_processed/features_full.csv")
    print(df.head())
    print("Features:", len(df.columns))


if __name__ == "__main__":
    process_features()
