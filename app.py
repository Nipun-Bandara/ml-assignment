"""
KadeMila – Sri Lankan food price ensemble dashboard (Streamlit).

Run the notebook first (it trains and saves the models), then:
    streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.kademila_utils import FEATURES, set_categories, to_price  # noqa: F401  (also needed to unpickle models)

ROOT = Path(__file__).parent
DATA, MODELS, REPORTS = ROOT / "data" / "processed", ROOT / "models", ROOT / "reports"
NAIVE = "Naive (last month)"

st.set_page_config(page_title="KadeMila – Food Price Forecasts", page_icon="🛒", layout="wide")


# ------------------------------------------------------------------ loading (cached)
@st.cache_data
def load_data():
    clean = pd.read_csv(DATA / "prices_clean.csv", parse_dates=["month"])
    model_df = pd.read_csv(DATA / "model_table.csv", parse_dates=["month"])
    nxt = pd.read_csv(DATA / "next_month_features.csv", parse_dates=["month"])
    comp = pd.read_csv(REPORTS / "model_comparison.csv")
    preds = pd.read_csv(REPORTS / "test_predictions.csv", parse_dates=["month"])
    imp = pd.read_csv(REPORTS / "feature_importance.csv", index_col=0)
    shap_imp = pd.read_csv(REPORTS / "shap_importance.csv", index_col=0).iloc[:, 0]
    rice_ts = pd.read_csv(REPORTS / "rice_ts_comparison.csv")
    rice_fc = pd.read_csv(REPORTS / "rice_forecasts.csv", index_col=0, parse_dates=True)
    rice_hist = pd.read_csv(DATA / "rice_colombo_monthly.csv", index_col=0, parse_dates=True)["price"]
    return clean, model_df, nxt, comp, preds, imp, shap_imp, rice_ts, rice_fc, rice_hist


@st.cache_resource
def load_meta():
    return json.load(open(MODELS / "meta.json"))


@st.cache_resource
def load_model(file_key):
    return joblib.load(MODELS / f"{file_key}.joblib")


def predict_price(model_name, rows):
    """Predict next-month price in LKR for feature rows."""
    if model_name == NAIVE:
        return rows["lag_1"].values
    meta = load_meta()
    model = load_model(meta["model_files"][model_name])
    X = set_categories(rows, meta["categories"])[FEATURES]
    return to_price(model.predict(X), rows)


@st.cache_data
def forecast_all(model_name):
    nxt = load_data()[2]
    out = nxt[["month", "admin1", "admin2", "market", "category", "commodity", "unit", "lag_1"]].copy()
    out["forecast"] = predict_price(model_name, nxt)
    out["change_%"] = (out["forecast"] / out["lag_1"] - 1) * 100
    return out


try:
    clean, model_df, nxt, comp, preds, imp, shap_imp, rice_ts, rice_fc, rice_hist = load_data()
    meta = load_meta()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Run `python src/prepare_data.py` and the notebook (Restart & Run All) first.")
    st.stop()

BEST = meta["best_model"]
MODEL_NAMES = [NAIVE] + list(meta["model_files"].keys())
LAST_MONTH = clean["month"].max()
NEXT_MONTH = nxt["month"].max()


def model_error_by(group_col, model_name):
    """Mean absolute % error of a model on the 2025 test set, by a grouping column."""
    e = (preds[model_name] - preds["price"]).abs() / preds["price"]
    return e.groupby(preds[group_col]).mean()


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🛒 KadeMila")
st.sidebar.caption("Next-month food price forecasts for Sri Lankan markets · ensemble learning")
page = st.sidebar.radio("Go to", ["🏠 Overview", "🏆 Model comparison", "🔍 Model explorer", "🧠 Explainability",
                                  "🍚 Rice forecast (RF + ARIMA)", "💰 Price check & forecast", "🚨 Early warning"])
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Best model:** {BEST}")
st.sidebar.markdown(f"**Data:** WFP / HDX, Department of Census & Statistics, HARTI · up to **{LAST_MONTH:%B %Y}**")

# ------------------------------------------------------------------ 1. overview
if page == "🏠 Overview":
    st.title("KadeMila – Sri Lankan food prices at a glance")
    st.write("Monthly retail food prices across Sri Lankan markets (WFP / DCS / HARTI). "
             "After the 2022 crisis, prices became volatile and uneven between districts. "
             "KadeMila forecasts next month's price and flags unfair quotes and coming price jumps.")
    c = st.columns(4)
    c[0].metric("Price records", f"{len(clean):,}")
    c[1].metric("Markets (all years)", clean["market"].nunique(), help=f"{model_df['market'].nunique()} markets in the modelling period")
    c[2].metric("Food items", clean["commodity"].nunique(), help=f"{model_df['commodity'].nunique()} items in the modelling period")
    c[3].metric("Period", f"{clean['month'].min():%Y}–{LAST_MONTH:%y}", help=f"Jan {clean['month'].min():%Y} to {LAST_MONTH:%B %Y}")

    st.subheader("Sample rows (modelling table)")
    st.dataframe(model_df.sample(8, random_state=1).assign(month=lambda d: d["month"].dt.strftime("%Y-%m"))[["month", "market", "admin2", "commodity", "unit", "price",
                                                     "lag_1", "rel_to_national", "nat_change_1"]], use_container_width=True)

    left, right = st.columns(2)
    with left:
        items = sorted(clean["commodity"].unique())
        pick = st.multiselect("National median price trend", items, default=["Carrots", "Tomatoes", "Rice (white)", "Coconut"])
        nat = (clean[clean["commodity"].isin(pick) & (clean["month"] >= "2023-06-01")]
               .groupby(["month", "commodity"])["price"].median().reset_index())
        st.plotly_chart(px.line(nat, x="month", y="price", color="commodity", markers=True,
                                labels={"price": "LKR (national median)", "month": ""}), use_container_width=True)
    with right:
        item = st.selectbox("Regional price map – item", items, index=items.index("Carrots"))
        latest = clean[(clean["commodity"] == item) & (clean["month"] == LAST_MONTH)].copy()
        latest["vs national"] = latest["price"] / latest["price"].median()
        fig = px.scatter_geo(latest, lat="latitude", lon="longitude", color="vs national", size="price",
                             hover_name="market", hover_data={"price": ":.0f", "vs national": ":.2f",
                                                              "latitude": False, "longitude": False},
                             color_continuous_scale="RdYlGn_r", range_color=(0.7, 1.3), height=520)
        fig.update_geos(resolution=50, showcountries=True, showland=True, landcolor="#f2efe6", showocean=True,
                        oceancolor="#dbe9f4", lataxis_range=[5.8, 10.0], lonaxis_range=[79.4, 82.1], projection_type="mercator")
        fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), title=f"{item}, {LAST_MONTH:%B %Y}: red = dearer than national")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("How volatile is each item? (std. dev. of monthly % change)")
    w = clean[clean["month"] >= "2023-06-01"].sort_values("month")
    w = w.assign(mom=w.groupby(["market", "commodity"])["price"].pct_change() * 100)
    vol = w.groupby("commodity")["mom"].std().sort_values(ascending=False).reset_index()
    st.plotly_chart(px.bar(vol, x="commodity", y="mom", labels={"mom": "% volatility", "commodity": ""}), use_container_width=True)

# ------------------------------------------------------------------ 2. model comparison
elif page == "🏆 Model comparison":
    st.title("Model comparison – 2025 test set (Jan–Sep 2025)")
    st.write("All models trained on Jun 2023 – Dec 2024 and tested on the same 9,061 unseen 2025 prices. "
             "The **naive** forecast (price = last month's price) is the bar to beat.")
    fam = st.multiselect("Families", sorted(comp["Family"].unique()), default=sorted(comp["Family"].unique()))
    view = comp[comp["Family"].isin(fam)].copy()
    view.insert(0, "", np.where(view["Model"] == BEST, "🏆", ""))
    st.dataframe(view.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.4f}", "MAPE_%": "{:.2f}",
                                    "Train time (s)": "{:.1f}"})
                 .highlight_min(subset=["MAE", "RMSE", "MAPE_%"], color="#c6efce"),
                 use_container_width=True, hide_index=True)

    metric = st.radio("Metric", ["MAE", "RMSE", "MAPE_%", "Train time (s)"], horizontal=True)
    v = view.sort_values(metric)
    colors = ["#2ca02c" if m == BEST else ("#9e9e9e" if f == "Baseline" else "#4C72B0") for m, f in zip(v["Model"], v["Family"])]
    fig = go.Figure(go.Bar(x=v[metric], y=v["Model"], orientation="h", marker_color=colors,
                           text=v[metric].round(2), textposition="outside"))
    if metric != "Train time (s)":
        fig.add_vline(x=float(comp.loc[comp["Model"] == NAIVE, metric].iloc[0]), line_dash="dash", line_color="red",
                      annotation_text="naive")
    fig.update_layout(height=480, yaxis={"autorange": "reversed"}, xaxis_title=metric, margin=dict(l=10, r=40, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Where does the best model help? (MAE by category, LKR)")
    cat_mae = pd.DataFrame({m: (preds[m] - preds["price"]).abs().groupby(preds["category"]).mean()
                            for m in [NAIVE, "Random Forest", "LightGBM", BEST]}).round(1)
    cat_mae["best vs naive"] = ((cat_mae[NAIVE] - cat_mae[BEST]) / cat_mae[NAIVE] * 100).round(1).astype(str) + " %"
    st.dataframe(cat_mae, use_container_width=True)
    st.info("Best model wins on volatile foods (vegetables, fish) and on big price moves (> 20 %), where it cuts the error by about a fifth. "
            "On stable items like rice and lentils, 'same as last month' is already almost perfect.")

# ------------------------------------------------------------------ 3. model explorer
elif page == "🔍 Model explorer":
    st.title("Model explorer")
    model_name = st.selectbox("Model", MODEL_NAMES, index=MODEL_NAMES.index(BEST))
    r = comp[comp["Model"] == model_name].iloc[0]
    c = st.columns(4)
    c[0].metric("MAE (LKR)", f"{r['MAE']:.2f}"); c[1].metric("RMSE (LKR)", f"{r['RMSE']:.2f}")
    c[2].metric("MAPE", f"{r['MAPE_%']:.2f} %"); c[3].metric("R²", f"{r['R2']:.4f}")

    left, right = st.columns(2)
    with left:
        s = preds.sample(min(3000, len(preds)), random_state=0)
        fig = px.scatter(s, x="price", y=model_name, color="category", opacity=0.5, log_x=True, log_y=True,
                         hover_data=["market", "commodity", "month"],
                         labels={"price": "Actual (LKR)", model_name: "Predicted (LKR)"}, title="Actual vs predicted (2025)")
        lo, hi = s["price"].min(), s["price"].max()
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="red"), name="perfect"))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        mk = st.selectbox("Market", sorted(preds["market"].unique()), index=sorted(preds["market"].unique()).index("Nuwara Eliya"))
        items = sorted(preds.loc[preds["market"] == mk, "commodity"].unique())
        cm = st.selectbox("Item", items, index=items.index("Carrots") if "Carrots" in items else 0)
        hist = clean[(clean["market"] == mk) & (clean["commodity"] == cm) & (clean["month"] >= "2024-01-01")]
        p = preds[(preds["market"] == mk) & (preds["commodity"] == cm)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["month"], y=hist["price"], name="actual", mode="lines+markers", line=dict(color="black")))
        fig.add_trace(go.Scatter(x=p["month"], y=p[NAIVE], name="naive", line=dict(dash="dash", color="grey")))
        fig.add_trace(go.Scatter(x=p["month"], y=p[model_name], name=model_name, mode="lines+markers", line=dict(color="#2ca02c")))
        fig.add_vline(x=pd.Timestamp("2025-01-01"), line_color="red")
        fig.update_layout(title=f"{cm} – {mk} (test period after red line)", yaxis_title="LKR")
        st.plotly_chart(fig, use_container_width=True)

    err = pd.DataFrame({"naive": model_error_by("commodity", NAIVE), model_name: model_error_by("commodity", model_name)}) * 100
    err["better than naive"] = err["naive"] > err[model_name]
    st.subheader("Average % error by item")
    st.plotly_chart(px.bar(err.sort_values(model_name).reset_index().melt(id_vars=["commodity", "better than naive"],
                                                                          value_vars=["naive", model_name]),
                           x="commodity", y="value", color="variable", barmode="group",
                           labels={"value": "MAPE %", "commodity": "", "variable": ""}), use_container_width=True)

# ------------------------------------------------------------------ 4. explainability
elif page == "🧠 Explainability":
    st.title("What drives the forecasts?")
    left, right = st.columns(2)
    with left:
        col = st.selectbox("Built-in feature importance – model", list(imp.columns), index=list(imp.columns).index("Average"))
        top = imp[col].sort_values(ascending=False).head(15)[::-1].reset_index()
        top.columns = ["feature", "importance"]
        st.plotly_chart(px.bar(top, x="importance", y="feature", orientation="h", height=520), use_container_width=True)
    with right:
        st.markdown("**SHAP (LightGBM)** – average effect on the predicted % change")
        sv = shap_imp.sort_values(ascending=False).head(15)[::-1].reset_index()
        sv.columns = ["feature", "mean |SHAP|"]
        st.plotly_chart(px.bar(sv, x="mean |SHAP|", y="feature", orientation="h", height=520,
                               color_discrete_sequence=["#ff7f0e"]), use_container_width=True)
    fig_path = REPORTS / "figures" / "shap_dependence.png"
    if fig_path.exists():
        st.image(str(fig_path), caption="SHAP dependence: dearer-than-national markets and recent spikes are predicted to fall back")
    st.markdown("""
* **Prices return to the national level** – a market about 5 % cheaper than the national median is predicted to rise, a dearer one to fall.
* **Spikes fade** – if last month was above its 3-month average, the price is expected to come down.
* **National momentum** – when an item gets cheaper (or dearer) across the country, every market tends to follow next month.
* **Location matters less** than *how dear the market already is*.
""")

# ------------------------------------------------------------------ 5. rice time series
elif page == "🍚 Rice forecast (RF + ARIMA)":
    st.title("Colombo white rice – RF + ARIMA hybrid (time series)")
    st.write("Monthly price 2006–2025, including the 2022 crisis. Train to Dec 2023, one-step-ahead forecasts for Jan 2024 – Sep 2025.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rice_hist.index, y=rice_hist.values, name="actual", line=dict(color="black")))
    for c, dash in [("ARIMA", "dash"), ("SARIMA", "dot"), ("RF_lag", "dashdot"), ("RF_ARIMA_hybrid", "solid")]:
        fig.add_trace(go.Scatter(x=rice_fc.index, y=rice_fc[c], name=c.replace("_", " "), line=dict(dash=dash)))
    fig.add_vrect(x0="2022-01-01", x1="2023-06-30", fillcolor="red", opacity=0.08, line_width=0, annotation_text="2022 crisis")
    fig.update_layout(height=450, yaxis_title="LKR per kg")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(rice_ts.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE_%": "{:.2f}", "Train time (s)": "{:.2f}"}),
                 use_container_width=True, hide_index=True)
    st.info("After the crisis, rice stayed flat at about Rs 250/kg, so the naive forecast and ARIMA were hard to beat. "
            "A hybrid only helps when the residuals still contain a pattern.")

# ------------------------------------------------------------------ 6. price check & forecast
elif page == "💰 Price check & forecast":
    st.title(f"Is this price fair? · Forecast for {NEXT_MONTH:%B %Y}")
    st.write("Pick a market and an item. KadeMila forecasts next month's price from the latest data and tells you whether a quoted price is fair.")
    c1, c2, c3 = st.columns(3)
    markets = sorted(nxt["market"].unique())
    mk = c1.selectbox("Market", markets, index=markets.index("Colombo City") if "Colombo City" in markets else 0)
    items = sorted(nxt.loc[nxt["market"] == mk, "commodity"].unique())
    cm = c2.selectbox("Item", items, index=items.index("Carrots") if "Carrots" in items else 0)
    model_name = c3.selectbox("Model", MODEL_NAMES, index=MODEL_NAMES.index(BEST))

    row = nxt[(nxt["market"] == mk) & (nxt["commodity"] == cm)].copy()
    unit = row["unit"].iloc[0]
    with st.form("predict"):
        f1, f2 = st.columns(2)
        last_price = f1.number_input(f"Last known price (LKR per {unit}) – change it to test a what-if", min_value=1.0,
                                     value=float(round(row["lag_1"].iloc[0], 2)), step=5.0)
        quote = f2.number_input(f"Price you were quoted (LKR per {unit}, optional)", min_value=0.0, value=0.0, step=5.0)
        st.form_submit_button("Forecast")

    if abs(last_price - row["lag_1"].iloc[0]) > 1e-6:          # what-if: update features that depend on last price
        row["lag_1"] = last_price
        row["roll_mean_3"] = row[["lag_1", "lag_2", "lag_3"]].mean(axis=1)
        row["pct_change_1"] = row["lag_1"] / row["lag_2"] - 1
        row["rel_to_national"] = row["lag_1"] / row["nat_median_lag1"]
        row["lag1_vs_roll"] = row["lag_1"] / row["roll_mean_3"] - 1
    pred = float(predict_price(model_name, row)[0])
    band = float(model_error_by("commodity", model_name).get(cm, 0.12))
    lo, hi = pred * (1 - band), pred * (1 + band)

    m = st.columns(4)
    m[0].metric(f"Last price ({row['month'].iloc[0] - pd.DateOffset(months=int(row['months_since_last'].iloc[0])):%b %Y})", f"Rs {last_price:,.0f}")
    m[1].metric(f"Forecast {NEXT_MONTH:%b %Y}", f"Rs {pred:,.0f}", f"{(pred / last_price - 1) * 100:+.1f} %")
    m[2].metric("Typical range", f"Rs {lo:,.0f} – {hi:,.0f}", help="± this model's average % error for this item in 2025")
    m[3].metric("vs national median", f"{row['rel_to_national'].iloc[0]:.2f}×")

    if quote > 0:
        if quote > hi:
            st.error(f"🔴 Rs {quote:,.0f} looks **HIGH** – {(quote / pred - 1) * 100:.0f} % above the expected price.")
        elif quote < lo:
            st.success(f"🟢 Rs {quote:,.0f} looks **LOW / a good deal** – {(1 - quote / pred) * 100:.0f} % below the expected price.")
        else:
            st.info(f"🟡 Rs {quote:,.0f} is **FAIR** – within the typical range.")

    hist = clean[(clean["market"] == mk) & (clean["commodity"] == cm) & (clean["month"] >= "2023-06-01")]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["month"], y=hist["price"], name="actual", mode="lines+markers", line=dict(color="black")))
    fig.add_trace(go.Scatter(x=[NEXT_MONTH, NEXT_MONTH], y=[lo, hi], mode="lines", line=dict(color="#2ca02c", width=8),
                             opacity=0.3, name="typical range"))
    fig.add_trace(go.Scatter(x=[NEXT_MONTH], y=[pred], mode="markers", marker=dict(size=14, color="#2ca02c", symbol="star"),
                             name="forecast"))
    if quote > 0:
        fig.add_trace(go.Scatter(x=[NEXT_MONTH], y=[quote], mode="markers", marker=dict(size=12, color="red", symbol="x"), name="your quote"))
    fig.update_layout(title=f"{cm} – {mk}", yaxis_title=f"LKR per {unit}", height=420)
    st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ 7. early warning
elif page == "🚨 Early warning":
    st.title(f"Early warning – expected price moves in {NEXT_MONTH:%B %Y}")
    model_name = st.selectbox("Model", MODEL_NAMES[1:], index=MODEL_NAMES[1:].index(BEST))
    with st.spinner("Forecasting every market and item..."):
        fc = forecast_all(model_name)
    thr = st.slider("Alert threshold (% change)", 5, 30, 10)
    prov = st.multiselect("Province", sorted(fc["admin1"].unique()))
    v = fc[fc["admin1"].isin(prov)] if prov else fc

    nat = v.groupby("commodity")["change_%"].median().sort_values().reset_index()
    nat["direction"] = np.where(nat["change_%"] >= 0, "rise", "fall")
    st.plotly_chart(px.bar(nat, x="change_%", y="commodity", orientation="h", color="direction",
                           color_discrete_map={"rise": "#d62728", "fall": "#2ca02c"}, height=700,
                           labels={"change_%": "Median forecast change across markets (%)", "commodity": ""},
                           title="Forecast change by item (median across markets)"), use_container_width=True)
    a, b = st.columns(2)
    cols = ["market", "admin2", "commodity", "lag_1", "forecast", "change_%"]
    fmt = {"lag_1": "{:,.0f}", "forecast": "{:,.0f}", "change_%": "{:+.1f}"}
    a.subheader(f"🔺 Rising more than {thr} %")
    a.dataframe(v[v["change_%"] >= thr].sort_values("change_%", ascending=False)[cols].style.format(fmt),
                use_container_width=True, hide_index=True)
    b.subheader(f"🔻 Falling more than {thr} %")
    b.dataframe(v[v["change_%"] <= -thr].sort_values("change_%")[cols].style.format(fmt),
                use_container_width=True, hide_index=True)
    st.caption("Forecasts use only data up to the last survey month. Typical error is about 12 % (MAPE), higher for vegetables.")
