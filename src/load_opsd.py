import pandas as pd

COUNTRIES = ["RO", "HU", "DE"]

def load_and_transform(filepath="../data_raw/time_series_60min_singleindex.csv"):
    df = pd.read_csv(filepath, parse_dates=["utc_timestamp"])

    df = df.rename(columns={"utc_timestamp": "timestamp"})

    load_cols = [f"{c}_load_actual_entsoe_transparency" for c in COUNTRIES]
    load_cols_exists = [c for c in load_cols if c in df.columns]

    if len(load_cols_exists) == 0:
        raise ValueError("No matching load columns found!")

    df = df[["timestamp"] + load_cols_exists]

    df_long = df.melt(
        id_vars="timestamp",
        value_vars=load_cols_exists,
        var_name="country",
        value_name="load_MW"
    )

    df_long["country"] = (
        df_long["country"].str.replace("_load_actual_entsoe_transparency", "")
    )

    df_long = df_long.sort_values(["country", "timestamp"])

    df_long.to_csv("../data_processed/load_europe.csv", index=False)
    print("Saved: data_processed/load_europe.csv")
    print(df_long.head())
    print(df_long["country"].unique())

if __name__ == "__main__":
    load_and_transform()
