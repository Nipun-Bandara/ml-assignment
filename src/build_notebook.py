"""
Builds KadeMila.ipynb from code kept in src/ (prepare_data.py, nb_models.py, nb_discussion.py).
Run:  python src/build_notebook.py && jupyter nbconvert --to notebook --execute --inplace KadeMila.ipynb
"""
import re
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

# ================================================================ 0. Title
md("""
# KadeMila – Next-month food price forecasts for 36 Sri Lankan markets
### Ensemble Learning Dashboard (Bagging · Boosting · Stacking · Hybrid)

**Author:** [Your name] · **Index no:** [Your index number] · **Module:** [Module code / name] · **Date:** November 2026
""")

md("""
**Contents**
1. Problem & innovation · 2. Data source · 3. Data cleaning & feature engineering · 4. EDA · 5. Modelling setup · 6. Baselines ·
7. Bagging · 8. Boosting · 9. Stacking · 10. Hybrid models (RF + ANN, RF + ARIMA) · 11. Model comparison · 12. Explainability (SHAP) ·
13. Research discussion · 14. Dashboard · 15. Conclusion & limitations · 16. References

**Key result:** stacking (RF + XGBoost + LightGBM + CatBoost → Ridge) is the best model on 2025 prices and the only one that beats the naive
"same as last month" forecast on every metric; on big price moves (> 20 %) it cuts the error by about 21 %.
""")

md("""
## 1. Problem & innovation
After the 2022 economic crisis, food prices in Sri Lanka became volatile and uneven: the same kilo of rice or dhal can cost very
different amounts in different districts. Families, small traders and aid programmes (Aswesuma, WFP) have no simple tool that
says whether a price is fair or where it is heading. **KadeMila** ("shop price") uses ensemble learning on official WFP / DCS
market data to forecast next month's retail price of 33 food items in 36 markets, and shows which districts pay more than the
national level.

**What is new**
1. **"Is this price fair?" check** – enter market, item and quoted price; the dashboard shows the expected price and whether the quote is high or low.
2. **Regional gap feature** (`rel_to_national`) – how much dearer/cheaper a market is than the national median.
3. **Early warning** – markets and items predicted to rise (or fall) more than 10 % next month.
4. **Crisis lens** – RF + ARIMA hybrid on 20 years of Colombo rice prices, including the 2022 shock.
""")

md("""
### Setup
Run once if libraries are missing:
```
pip install -r requirements.txt
```
""")

code("""
import warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
pd.set_option("display.max_columns", 30)
sns.set_theme(style="whitegrid", context="notebook")
FIG = Path("reports/figures"); FIG.mkdir(parents=True, exist_ok=True)
""")

# ================================================================ 2. Data source
md("""
## 2. Data source
* **Dataset:** WFP *Sri Lanka – Food Prices* on the Humanitarian Data Exchange (HDX):
  https://data.humdata.org/dataset/wfp-food-prices-for-sri-lanka
* **Original sources:** Department of Census and Statistics (DCS), HARTI, CEPETCO and WFP market surveys.
* **Licence:** CC BY-IGO. Downloaded once as CSV (`data/raw/wfp_food_prices_lka.csv`), so no scraping is needed and the notebook runs offline.
* **Columns:** date, province (admin1), district (admin2), market, lat/lon, category, commodity, unit, price type, price (LKR).
""")

code("""
raw = pd.read_csv("data/raw/wfp_food_prices_lka.csv")
raw = raw[~raw["date"].astype(str).str.startswith("#")]
raw["date"] = pd.to_datetime(raw["date"])
print("Rows:", len(raw), "| Markets:", raw["market"].nunique(), "| Commodities:", raw["commodity"].nunique(),
      "| Period:", raw["date"].min().date(), "to", raw["date"].max().date())
raw.head()
""")

# ================================================================ 3. Cleaning (from src/prepare_data.py)
md("""
## 3. Data cleaning & feature engineering
Same code as `src/prepare_data.py`. Key decisions:
* Retail prices only; drop the "National Average" pseudo-market, fuel (non-food) and commodities with < 100 rows.
* Prices are already numeric LKR (no text cleaning needed).
* Outliers: crisis spikes are real, so we only drop data-entry errors (> 4x or < 1/4 of that month's national median).
* **Lag features use only last month's information** → no data leakage.
* Modelling window starts June 2023 (when 34–36 markets are surveyed every month). **Train: Jun 2023 – Dec 2024, Test: Jan – Sep 2025.**
""")
src = (ROOT / "src" / "prepare_data.py").read_text()
src = src.split('"""', 2)[2]                                   # drop module docstring
parts = re.split(r"\n# -{10,} ", src)
code(parts[0])                                                  # imports + paths
for p in parts[1:]:
    title, body = p.split("\n", 1)
    code(f"# {title.strip()}\n{body}")

code("""
model_df = pd.read_csv("data/processed/model_table.csv", parse_dates=["month"])
clean_df = pd.read_csv("data/processed/prices_clean.csv", parse_dates=["month"])
rice = pd.read_csv("data/processed/rice_colombo_monthly.csv", parse_dates=["month"], index_col="month")["price"]
model_df.head()
""")

# ================================================================ 4. EDA
md("## 4. Exploratory data analysis (EDA)")

md("### 4.1 Data coverage over time")
code("""
cov = raw[raw["pricetype"] == "Retail"].groupby(raw["date"].dt.year).agg(rows=("price", "size"), markets=("market", "nunique"))
fig, ax = plt.subplots(figsize=(11, 4))
ax.bar(cov.index, cov["rows"], color="#4C72B0")
ax.set_ylabel("Rows"); ax.set_title("Rows per year (bars) and markets surveyed (line)")
ax2 = ax.twinx(); ax2.plot(cov.index, cov["markets"], color="#DD8452", marker="o"); ax2.set_ylabel("Markets"); ax2.grid(False)
plt.tight_layout(); plt.savefig(FIG / "eda_01_coverage.png", dpi=120); plt.show()
""")
md("""**Insight:** before 2023 only 1–10 markets were surveyed in most years; from mid-2023 WFP covers 34–36 markets every month. This is why we train on Jun 2023 – Dec 2024 and test on Jan – Sep 2025, not on 2004–2021.""")

md("### 4.2 The 2022 crisis: Colombo white rice, 2006–2025")
code("""
fig, ax = plt.subplots(figsize=(11, 4))
obs = clean_df[(clean_df["market"] == "Colombo City") & (clean_df["commodity"] == "Rice (white)")].set_index("month")["price"]
ax.plot(rice.index, rice.values, color="grey", ls="--", lw=1, label="interpolated gap")
ax.plot(obs.reindex(rice.index).index, obs.reindex(rice.index).values, color="#4C72B0", label="observed")
ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2023-06-30"), color="red", alpha=0.12, label="2022 economic crisis")
ax.set_ylabel("LKR per kg"); ax.set_title("Retail price of white rice – Colombo City"); ax.legend()
plt.tight_layout(); plt.savefig(FIG / "eda_02_rice_crisis.png", dpi=120); plt.show()
print(rice.resample("YE").mean().round(1).loc["2019":].to_string())
""")
md("""**Insight:** white rice cost about Rs 100–110/kg in 2020–21, jumped to about Rs 230 in 2022 and settled near Rs 250 from 2024: the crisis roughly **doubled** the price, and it never came back down. (The dashed part is a stretch of missing months filled by interpolation.)""")

md("### 4.3 Price distribution: raw vs log")
code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(model_df["price"], bins=60, ax=axes[0]); axes[0].set_title("Price (LKR)")
sns.histplot(np.log1p(model_df["price"]), bins=60, ax=axes[1], color="#55A868"); axes[1].set_title("log(1 + price)")
plt.tight_layout(); plt.savefig(FIG / "eda_03_price_dist.png", dpi=120); plt.show()
print("Skewness raw:", round(model_df["price"].skew(), 2), "| log:", round(np.log1p(model_df["price"]).skew(), 2))
""")
md("""**Insight:** raw prices are right-skewed (Rs 20 eggs to Rs 3,000+ fish), but log prices are close to symmetric, so the models train on log(price) and we convert back to LKR for the metrics.""")

md("### 4.4 Price index by food category (Jun 2023 = 100)")
code("""
w = clean_df[clean_df["month"] >= "2023-06-01"]
idx = w.groupby(["commodity", "month"])["price"].median().reset_index()
idx["base"] = idx.groupby("commodity")["price"].transform("first")
idx["index"] = idx["price"] / idx["base"] * 100
idx = idx.merge(w[["commodity", "category"]].drop_duplicates(), on="commodity")
cat_idx = idx.groupby(["category", "month"])["index"].mean().unstack(0)
ax = cat_idx.plot(figsize=(11, 4.5), title="Average price index by category (national median, Jun 2023 = 100)")
ax.axhline(100, color="grey", ls="--", lw=1); ax.set_ylabel("Index")
plt.tight_layout(); plt.savefig(FIG / "eda_04_category_index.png", dpi=120); plt.show()
""")
md("""**Insight:** vegetables & fruits swing the most (index peaked around 160 in Dec 2023), cereals stay close to 100–115, and dried chilli (the only 'miscellaneous' item) and pulses got cheaper after the crisis.""")

md("### 4.5 Which items are most volatile?")
code("""
mom = w.sort_values("month").groupby(["market", "commodity"])["price"].pct_change()
vol = (w.assign(mom=mom).groupby("commodity")["mom"].std() * 100).sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(11, 6))
vol.plot.barh(ax=ax, color="#C44E52"); ax.invert_yaxis()
ax.set_xlabel("Std. dev. of month-on-month change (%)"); ax.set_title("Price volatility by commodity (Jun 2023 – Sep 2025)")
plt.tight_layout(); plt.savefig(FIG / "eda_05_volatility.png", dpi=120); plt.show()
""")
md("""**Insight:** tomatoes and carrots move about 50 % month to month, while rice and lentils move under 3 %. Errors will be largest on upcountry vegetables, so we report results per category too.""")

md("### 4.6 Seasonality: average month-on-month change by calendar month")
code("""
s = w.assign(mom=mom * 100, m=w["month"].dt.month)
s["group"] = np.where(s["category"] == "vegetables and fruits", "Vegetables & fruits", "Other foods")
season = s.groupby(["m", "group"])["mom"].median().unstack()
ax = season.plot.bar(figsize=(11, 4), title="Median month-on-month price change by month")
ax.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], rotation=0)
ax.axhline(0, color="black", lw=0.8); ax.set_ylabel("%")
plt.tight_layout(); plt.savefig(FIG / "eda_06_seasonality.png", dpi=120); plt.show()
""")
md("""**Insight:** vegetable prices tend to rise in Nov–Jan (north-east monsoon rains and festive demand) and fall in Mar and Aug (harvests). We only have about two years of seasons, so this is a tendency, not a rule; it is why the month features are included.""")

md("### 4.7 Regional price gaps by province")
code("""
order = model_df.groupby("admin1")["rel_to_national"].median().sort_values().index
fig, ax = plt.subplots(figsize=(11, 4.5))
sns.boxplot(data=model_df, x="admin1", y="rel_to_national", order=order, showfliers=False, ax=ax)
ax.axhline(1, color="red", ls="--", lw=1); ax.set_xlabel(""); ax.set_ylabel("Market price / national median")
ax.set_title("How dear is each province compared with the national median? (1.0 = national level)")
plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(FIG / "eda_07_regional_gap.png", dpi=120); plt.show()
print(model_df.groupby("admin1")["rel_to_national"].median().sort_values().round(3).to_string())
""")
md("""**Insight:** Western Province (Colombo, Gampaha, Kalutara) is the dearest, about 3 % above the national median on average. The Northern Province has the widest spread, meaning some northern markets are far above and others far below national prices.""")

md("### 4.8 Last month's price vs this month's price")
code("""
smp = model_df.sample(4000, random_state=RANDOM_STATE)
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(smp["lag_1"], smp["price"], s=5, alpha=0.4)
lim = [smp[["lag_1", "price"]].min().min(), smp[["lag_1", "price"]].max().max()]
ax.plot(lim, lim, "r--", lw=1); ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Price last month (lag_1, LKR)"); ax.set_ylabel("Price this month (LKR)"); ax.set_title("Strong month-to-month persistence")
plt.tight_layout(); plt.savefig(FIG / "eda_08_lag_scatter.png", dpi=120); plt.show()
print("Correlation log(lag_1) vs log(price):", round(np.corrcoef(np.log(model_df["lag_1"]), np.log(model_df["price"]))[0, 1], 4))
""")
md("""**Insight:** this month's price is very close to last month's (correlation of the logs about 0.98). So a naive 'same as last month' forecast is already strong, and our models must **beat this naive baseline** to be useful.""")

md("### 4.9 Correlation between numeric features")
code("""
num = ["price", "lag_1", "lag_2", "lag_3", "roll_mean_3", "roll_std_3", "pct_change_1", "nat_median_lag1",
       "rel_to_national", "months_since_last", "latitude", "longitude", "month_sin", "month_cos"]
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(model_df[num].corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 7}, ax=ax)
ax.set_title("Correlation matrix"); plt.tight_layout(); plt.savefig(FIG / "eda_09_corr.png", dpi=120); plt.show()
""")
md("""**Insight:** the lag features, rolling mean and national median are almost perfectly correlated with each other (about 0.97–0.99). That is fine for tree ensembles, but it is why the linear baseline uses Ridge regularisation. Location and month features have weak *linear* correlation, but trees can still use them in interactions (e.g. carrots in Nuwara Eliya in December).""")

# ================================================================ 5+. Models (from src/nb_models.py, percent format)
for block in re.split(r"^# %%", "\n".join((ROOT / "src" / f).read_text() for f in ["nb_models.py", "nb_discussion.py"]), flags=re.M)[1:]:
    head, _, body = block.partition("\n")
    if "[markdown]" in head:
        md("\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.strip().splitlines()))
    else:
        code(body)

nb["cells"] = cells
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, ROOT / "KadeMila.ipynb")
print("wrote KadeMila.ipynb with", len(cells), "cells")
