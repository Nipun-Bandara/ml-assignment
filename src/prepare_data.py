"""
Step 2 - Data preparation for "PolPriceLK" (Sri Lankan food price forecaster).

Input : data/raw/wfp_food_prices_lka.csv  (WFP / HDX, downloaded once, no scraping)
Output: data/processed/prices_clean.csv      (cleaned long table, all years)
        data/processed/model_table.csv       (one row = market x commodity x month, with lag features)
        data/processed/rice_colombo_monthly.csv (monthly series for RF + ARIMA hybrid)

Run:  python src/prepare_data.py
The same code is pasted into the notebook's "Data cleaning" section.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd()  # notebook runs from project root
RAW = ROOT / "data" / "raw" / "wfp_food_prices_lka.csv"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- 1. load
df = pd.read_csv(RAW)
df = df[~df["date"].astype(str).str.startswith("#")]          # drop HXL tag row if present
df["date"] = pd.to_datetime(df["date"])
df["price"] = pd.to_numeric(df["price"], errors="coerce")     # price is already numeric LKR
print("raw rows:", len(df))

# ---------------------------------------------------------------- 2. filter
df = df[(df["pricetype"] == "Retail") & (df["currency"] == "LKR")]
df = df[df["market"] != "National Average"]                  # an aggregate, not a real market
df = df[df["category"] != "non-food"]                         # fuel is not food
df = df.dropna(subset=["price"])
df = df[df["price"] > 0]

# rare commodities (too few rows to learn from)
counts = df["commodity"].value_counts()
df = df[df["commodity"].isin(counts[counts >= 100].index)]

# month key: all WFP dates are the 15th -> use month start
df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
df = df.drop_duplicates(["month", "market", "commodity"], keep="last")
print("after filtering:", len(df))

# ---------------------------------------------------------------- 3. missing values
# admin1/admin2/lat/lon are only missing for rows we already dropped (National Average);
# fill any leftovers from the markets file so nothing breaks.
mk = pd.read_csv(ROOT / "data" / "raw" / "wfp_markets_lka.csv")
mk = mk[~mk["market_id"].astype(str).str.startswith("#")]
mk["market_id"] = mk["market_id"].astype(int)
df = df.merge(mk[["market_id", "admin1", "admin2", "latitude", "longitude"]].astype({"latitude": float, "longitude": float}),
              on="market_id", how="left", suffixes=("", "_mk"))
for c in ["admin1", "admin2", "latitude", "longitude"]:
    df[c] = df[c].fillna(df[c + "_mk"])
df = df.drop(columns=[c for c in df.columns if c.endswith("_mk")])
df[["admin1", "admin2"]] = df[["admin1", "admin2"]].fillna("Unknown")
print("missing values left:\n", df[["admin1", "admin2", "latitude", "longitude", "price"]].isna().sum().to_string())

# ---------------------------------------------------------------- 4. outliers
# Crisis spikes are REAL, so we do not clip big moves blindly.
# We only drop obvious data-entry errors: price more than 4x or less than 1/4 of the
# same commodity's national median in that month.
med = df.groupby(["commodity", "month"])["price"].transform("median")
ratio = df["price"] / med
bad = (ratio > 4) | (ratio < 0.25)
print("outliers dropped:", int(bad.sum()))
df = df[~bad].copy()

keep = ["month", "admin1", "admin2", "market", "latitude", "longitude",
        "category", "commodity", "unit", "price"]
clean = df[keep].sort_values(["market", "commodity", "month"]).reset_index(drop=True)
clean.to_csv(OUT / "prices_clean.csv", index=False)

# ---------------------------------------------------------------- 5. lag features
# Goal: predict THIS month's price using only information known LAST month.
# We also add one empty "next month" row per market x commodity (price unknown), so the dashboard
# can forecast the month after the data ends using exactly the same feature code.
NEXT_MONTH = clean["month"].max() + pd.DateOffset(months=1)
last_obs = clean.groupby(["market", "commodity"]).tail(1)
future = last_obs[last_obs["month"] >= NEXT_MONTH - pd.DateOffset(months=6)].assign(month=NEXT_MONTH, price=np.nan)
clean = pd.concat([clean, future]).sort_values(["market", "commodity", "month"]).reset_index(drop=True)
g = clean.groupby(["market", "commodity"], sort=False)
clean["lag_1"] = g["price"].shift(1)                       # last observed price
clean["lag_2"] = g["price"].shift(2)
clean["lag_3"] = g["price"].shift(3)
clean["months_since_last"] = (
    (clean["month"].dt.year * 12 + clean["month"].dt.month)
    - g["month"].shift(1).pipe(lambda s: s.dt.year * 12 + s.dt.month)
)
clean["roll_mean_3"] = g["price"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
clean["roll_std_3"] = g["price"].transform(lambda s: s.shift(1).rolling(3, min_periods=2).std())
clean["pct_change_1"] = clean["lag_1"] / clean["lag_2"] - 1

# national signal: median price of this commodity across all markets LAST month
nat = clean.groupby(["commodity", "month"])["price"].median().rename("nat_median").reset_index()
nat["month"] = nat["month"] + pd.DateOffset(months=1)        # shift -> "last month's national median"
clean = clean.merge(nat.rename(columns={"nat_median": "nat_median_lag1"}),
                    on=["commodity", "month"], how="left")
clean["rel_to_national"] = clean["lag_1"] / clean["nat_median_lag1"]   # is this market dear or cheap?

# national momentum and last year's seasonal move (both known at the end of last month)
nm = clean.groupby(["commodity", "month"])["price"].median().unstack(0).asfreq("MS")
nat_change = (nm.shift(1) / nm.shift(2) - 1).stack().rename("nat_change_1").reset_index()
ly_change = (nm.shift(11) / nm.shift(12) - 1).stack().rename("ly_change").reset_index()  # same move, one year ago
clean = clean.merge(nat_change, on=["month", "commodity"], how="left").merge(ly_change, on=["month", "commodity"], how="left")
clean["lag1_vs_roll"] = clean["lag_1"] / clean["roll_mean_3"] - 1                  # is last month above its 3-month trend?

# calendar (seasonality: Sinhala & Tamil New Year in April, Maha/Yala harvests)
clean["year"] = clean["month"].dt.year
clean["month_num"] = clean["month"].dt.month
clean["month_sin"] = np.sin(2 * np.pi * clean["month_num"] / 12)
clean["month_cos"] = np.cos(2 * np.pi * clean["month_num"] / 12)

# ---------------------------------------------------------------- 6. modelling window
def fill_features(d):
    d = d.copy()
    d["roll_std_3"] = d["roll_std_3"].fillna(0)
    for c in ["lag_2", "lag_3"]:
        d[c] = d[c].fillna(d["lag_1"])
    d["pct_change_1"] = d["pct_change_1"].fillna(0)
    d["nat_median_lag1"] = d["nat_median_lag1"].fillna(d["lag_1"])
    d["rel_to_national"] = d["rel_to_national"].fillna(1)
    d[["nat_change_1", "ly_change", "lag1_vs_roll"]] = d[["nat_change_1", "ly_change", "lag1_vs_roll"]].fillna(0)
    return d[d["months_since_last"] <= 6]                    # old lags are not informative

# Wide market coverage starts June 2023 (34-36 markets/month). Earlier years only have 1-3 markets.
model = clean[(clean["month"] >= "2023-06-01") & clean["price"].notna()].dropna(subset=["lag_1"])  # need a previous price
model = fill_features(model)
model["split"] = np.where(model["month"] >= "2025-01-01", "test", "train")
model.to_csv(OUT / "model_table.csv", index=False)
print("\nmodel table:", model.shape)
print(model["split"].value_counts().to_string())
print("markets:", model["market"].nunique(), "| commodities:", model["commodity"].nunique())

# features for the month after the data ends (used by the dashboard's forecast form)
nxt = fill_features(clean[clean["price"].isna()].dropna(subset=["lag_1"]))
nxt.to_csv(OUT / "next_month_features.csv", index=False)
print("next-month rows:", len(nxt), "for", NEXT_MONTH.strftime("%B %Y"))

# ---------------------------------------------------------------- 7. time series for RF + ARIMA
rice = (clean[(clean["market"] == "Colombo City") & (clean["commodity"] == "Rice (white)") & clean["price"].notna()]
        .set_index("month")["price"].asfreq("MS"))
print("\nrice series months:", len(rice), "| gaps filled by interpolation:", int(rice.isna().sum()))
rice = rice.interpolate(method="time")
rice.rename("price").to_frame().to_csv(OUT / "rice_colombo_monthly.csv")
