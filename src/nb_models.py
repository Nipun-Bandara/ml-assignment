# %% [markdown]
# ## 5. Modelling setup
# * **Task:** predict this month's retail price (LKR) for a market × commodity, using only information known at the end of last month.
# * **Target:** `log(price / last month's price)`, i.e. the model predicts the **% change**, then we convert back to LKR.
#   (In a quick test, predicting log(price) directly lost to the naive forecast; predicting the change works better because trees cannot extrapolate to new price levels.)
# * **Split (time-based, no shuffling):** train Jun 2023 – Dec 2024, test Jan – Sep 2025.
# * **Cross-validation:** expanding-window, month-based folds inside the training period (validate on Apr–Jun, Jul–Sep, Oct–Dec 2024).
# * **Metrics:** MAE, RMSE, R² and MAPE, all in LKR on the 2025 test set. `random_state = 42` everywhere.

# %%
import json, joblib
import matplotlib.dates
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_val_predict
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor, ExtraTreesRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from src.kademila_utils import (CAT_COLS, NUM_COLS, FEATURES, make_target, to_price, regression_metrics,
                                set_categories, CatBoostWrap, RFANNResidualHybrid, WeightedBlend)

MODELS = Path("models"); MODELS.mkdir(exist_ok=True)
REPORTS = Path("reports"); REPORTS.mkdir(exist_ok=True)

data = model_df.sort_values(["month", "market", "commodity"]).reset_index(drop=True)
CATEGORIES = {c: sorted(data[c].astype(str).unique()) for c in CAT_COLS}
data = set_categories(data, CATEGORIES)

train, test = data[data["split"] == "train"].reset_index(drop=True), data[data["split"] == "test"].reset_index(drop=True)
X_train, X_test = train[FEATURES], test[FEATURES]
y_train = make_target(train)
y_test_price = test["price"].values
print("Train:", X_train.shape, train["month"].min().date(), "to", train["month"].max().date())
print("Test :", X_test.shape, test["month"].min().date(), "to", test["month"].max().date())

# month-based expanding-window CV folds (no future data in any training fold)
def month_folds(months, starts=("2024-04-01", "2024-07-01", "2024-10-01"), length=3):
    folds = []
    for s in pd.to_datetime(list(starts)):
        e = s + pd.DateOffset(months=length)
        folds.append((np.where(months < s)[0], np.where((months >= s) & (months < e))[0]))
    return folds
CV = month_folds(train["month"])
for i, (a, b) in enumerate(CV, 1):
    print(f"fold {i}: train {len(a):>5} rows | validate {len(b):>4} rows ({train['month'].iloc[b].min().date()} to {train['month'].iloc[b].max().date()})")

# preprocessing
tree_pre = ColumnTransformer([("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT_COLS)],
                             remainder="passthrough")                      # trees: ordinal codes are enough
lin_pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)],
                            remainder=StandardScaler())                    # linear / ANN: one-hot + scaling

results, test_preds, fitted = [], {}, {}

def record(name, family, model, train_time, pred_log_change=None, pred_price=None):
    """Store test-set metrics (in LKR) for one model."""
    p = to_price(pred_log_change, test) if pred_price is None else pred_price
    test_preds[name] = p
    fitted[name] = model
    results.append({"Model": name, "Family": family, **regression_metrics(y_test_price, p), "Train time (s)": train_time})
    r = results[-1]
    print(f"{name:<28} MAE {r['MAE']:8.2f} | RMSE {r['RMSE']:8.2f} | R2 {r['R2']:.4f} | MAPE {r['MAPE_%']:.2f}% | {train_time:.1f}s")

def tune(name, family, estimator, space, n_iter):
    t = time.time()
    search = RandomizedSearchCV(estimator, space, n_iter=n_iter, cv=CV, scoring="neg_mean_absolute_error",
                                random_state=RANDOM_STATE, n_jobs=1, refit=True)
    search.fit(X_train, y_train)
    print(f"  best CV MAE (log-change): {-search.best_score_:.4f} | params: {search.best_params_}")
    record(name, family, search.best_estimator_, time.time() - t, search.best_estimator_.predict(X_test))
    return search.best_estimator_

# %% [markdown]
# ## 6. Baselines
# * **Naive:** "price this month = price last month". Any useful model must beat this.
# * **Ridge regression:** a regularised linear model (one-hot categories + scaled numbers).

# %%
record("Naive (last month)", "Baseline", None, 0.0, pred_price=test["lag_1"].values)

t = time.time()
ridge = Pipeline([("pre", lin_pre), ("model", RidgeCV(alphas=np.logspace(-2, 3, 12)))]).fit(X_train, y_train)
record("Ridge (linear)", "Baseline", ridge, time.time() - t, ridge.predict(X_test))

# %% [markdown]
# ## 7. Bagging: Random Forest, Bagging, Extra Trees
# Bagging trains many trees **independently** on bootstrap samples and averages them, which **reduces variance**.
# Random Forest also picks a random subset of features at each split, so the trees are less alike.

# %%
rf = tune("Random Forest", "Bagging",
          Pipeline([("pre", tree_pre), ("model", RandomForestRegressor(n_jobs=-1, random_state=RANDOM_STATE))]),
          {"model__n_estimators": [200, 300],
           "model__max_depth": [None, 20, 30],
           "model__min_samples_leaf": [1, 3, 5, 10],
           "model__max_features": [0.3, 0.5, 0.8]}, n_iter=6)

t = time.time()
bag = Pipeline([("pre", tree_pre), ("model", BaggingRegressor(DecisionTreeRegressor(min_samples_leaf=5, random_state=RANDOM_STATE),
                                                              n_estimators=100, max_samples=0.8, n_jobs=-1,
                                                              random_state=RANDOM_STATE))]).fit(X_train, y_train)
record("Bagging (decision trees)", "Bagging", bag, time.time() - t, bag.predict(X_test))

t = time.time()
et = Pipeline([("pre", tree_pre), ("model", ExtraTreesRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                                                n_jobs=-1, random_state=RANDOM_STATE))]).fit(X_train, y_train)
record("Extra Trees", "Bagging", et, time.time() - t, et.predict(X_test))

# %% [markdown]
# ## 8. Boosting: XGBoost, LightGBM, CatBoost
# Boosting builds trees **one after another**; each new tree fixes the errors of the previous ones, which **reduces bias**.
# The loss function is part of the light tuning: squared error, absolute error (robust to price spikes) or Huber.
# * XGBoost uses ordinal-encoded categories.
# * LightGBM and CatBoost use their **native categorical handling** (no one-hot needed).

# %%
xgb = tune("XGBoost", "Boosting",
           Pipeline([("pre", tree_pre), ("model", XGBRegressor(tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE))]),
           {"model__n_estimators": [400, 800],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [4, 6, 8],
            "model__min_child_weight": [1, 5, 10],
            "model__subsample": [0.7, 0.9],
            "model__colsample_bytree": [0.7, 0.9],
            "model__objective": ["reg:squarederror", "reg:absoluteerror", "reg:pseudohubererror"]}, n_iter=8)

# %%
lgbm = tune("LightGBM", "Boosting",
            LGBMRegressor(random_state=RANDOM_STATE, verbose=-1, subsample_freq=1),
            {"n_estimators": [400, 800],
             "learning_rate": [0.03, 0.05, 0.1],
             "num_leaves": [15, 31, 63],
             "min_child_samples": [20, 40, 80],
             "subsample": [0.7, 0.9],
             "colsample_bytree": [0.7, 0.9],
             "objective": ["l2", "l1", "huber"]}, n_iter=8)

# %%
cat = tune("CatBoost", "Boosting",
           CatBoostWrap(cat_features=CAT_COLS, random_state=RANDOM_STATE),
           {"iterations": [500, 800],
            "learning_rate": [0.05, 0.1],
            "depth": [4, 6, 8],
            "loss_function": ["RMSE", "MAE"]}, n_iter=4)

# %% [markdown]
# ## 9. Stacking
# The tuned RF, XGBoost, LightGBM and CatBoost become **base models**. A simple **Ridge** meta-model learns how much to trust each one,
# using their out-of-fold predictions. The training data is sorted by month and split into 5 consecutive blocks (no shuffling).
# *Limitation:* sklearn stacking needs every row to get an out-of-fold prediction, so some blocks are predicted by models that saw later months.

# %%
t = time.time()
stack = StackingRegressor(
    estimators=[("rf", rf), ("xgb", xgb), ("lgbm", lgbm), ("cat", cat)],
    final_estimator=Ridge(alpha=1.0), cv=KFold(5, shuffle=False), n_jobs=1)
stack.fit(X_train, y_train)
record("Stacking (RF+XGB+LGBM+Cat → Ridge)", "Stacking", stack, time.time() - t, stack.predict(X_test))
print("Meta-model weights:", dict(zip(["RF", "XGB", "LGBM", "Cat"], stack.final_estimator_.coef_.round(3))))

# %% [markdown]
# ## 10. Hybrid models
# ### 10.1 RF + ANN (tabular)
# * **ANN alone:** an MLP neural network (64 → 32 neurons) on one-hot + scaled features.
# * **RF + ANN residual hybrid (main):** RF predicts first; the ANN learns RF's leftover errors (out-of-fold residuals), with the RF prediction as an extra input. Final = RF + ANN correction.
# * **RF + ANN weighted blend:** `w × RF + (1 − w) × ANN`, with `w` chosen on out-of-fold training predictions (not on the test set).

# %%
ann_params = dict(hidden_layer_sizes=(64, 32), alpha=1e-3, learning_rate_init=1e-3, early_stopping=True,
                  validation_fraction=0.15, max_iter=300, random_state=RANDOM_STATE)

t = time.time()
ann = Pipeline([("pre", lin_pre), ("model", MLPRegressor(**ann_params))]).fit(X_train, y_train)
record("ANN (MLP)", "Hybrid", ann, time.time() - t, ann.predict(X_test))

rf_best = rf.named_steps["model"].get_params()
rf_for_hybrid = Pipeline([("pre", tree_pre), ("model", RandomForestRegressor(**rf_best))])
ann_pre_plus = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)], remainder=StandardScaler())

t = time.time()
hybrid = RFANNResidualHybrid(rf=rf_for_hybrid, ann=Pipeline([("pre", ann_pre_plus), ("model", MLPRegressor(**ann_params))]))
hybrid.fit(X_train, y_train)
record("RF + ANN residual hybrid", "Hybrid", hybrid, time.time() - t, hybrid.predict(X_test))

# %%
t = time.time()
oof_rf = cross_val_predict(rf_for_hybrid, X_train, y_train, cv=KFold(5, shuffle=False))
oof_ann = cross_val_predict(Pipeline([("pre", lin_pre), ("model", MLPRegressor(**ann_params))]), X_train, y_train, cv=KFold(5, shuffle=False))
weights = np.round(np.arange(0, 1.01, 0.05), 2)
oof_mae = [np.mean(np.abs(w * oof_rf + (1 - w) * oof_ann - y_train)) for w in weights]
best_w = float(weights[int(np.argmin(oof_mae))])
print("Best RF weight on out-of-fold data:", best_w)
blend = WeightedBlend(model_a=rf_for_hybrid, model_b=Pipeline([("pre", lin_pre), ("model", MLPRegressor(**ann_params))]), weight=best_w)
blend.fit(X_train, y_train)
record(f"RF + ANN blend (w={best_w})", "Hybrid", blend, time.time() - t, blend.predict(X_test))

# %% [markdown]
# ### 10.2 RF + ARIMA (time series: Colombo white rice, 2006–2025)
# * Monthly log price; **train to Dec 2023, test Jan 2024 – Sep 2025** (21 months), one-step-ahead walk-forward forecasts.
# * **ARIMA** captures trend; **SARIMA** adds yearly seasonality; **RF (lag features)** predicts the monthly change from past changes.
# * **RF + ARIMA hybrid:** the RF learns ARIMA's leftover errors (residuals) from lagged residuals and the month; final = ARIMA forecast + RF residual forecast.
# * Note: 11 missing months (mostly during 2022–23) were filled by interpolation, which makes that stretch smoother than reality.

# %%
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

ly = np.log(rice.asfreq("MS"))
ts_train, ts_test = ly[:"2023-12-01"], ly["2024-01-01":]
print("Train months:", len(ts_train), "| Test months:", len(ts_test))

# pick ARIMA(p,1,q) by AIC on the training data
aic = {(p, q): ARIMA(ts_train, order=(p, 1, q)).fit().aic for p in range(3) for q in range(3)}
p_best, q_best = min(aic, key=aic.get)
print("Best ARIMA order by AIC:", (p_best, 1, q_best))

def walk_forward(res, test_series):
    """One-step-ahead forecasts; after each month the true value is appended (no refit)."""
    out = []
    for t, val in test_series.items():
        out.append(res.forecast(1).iloc[0])
        res = res.append(test_series.loc[[t]], refit=False)
    return pd.Series(out, index=test_series.index)

t = time.time(); arima_res = ARIMA(ts_train, order=(p_best, 1, q_best)).fit()
arima_fc = walk_forward(arima_res, ts_test); t_arima = time.time() - t
t = time.time(); sarima_res = SARIMAX(ts_train, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12)).fit(disp=False)
sarima_fc = walk_forward(sarima_res, ts_test); t_sarima = time.time() - t

# RF on lag features: predict this month's change from the last 3 changes + month
def lag_frame(s, n=3):
    f = pd.DataFrame({f"lag{i}": s.shift(i) for i in range(1, n + 1)})
    f["m_sin"], f["m_cos"] = np.sin(2 * np.pi * s.index.month / 12), np.cos(2 * np.pi * s.index.month / 12)
    return f
dy = ly.diff()
F = lag_frame(dy).assign(target=dy).dropna()
t = time.time()
rf_ts = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE)
rf_ts.fit(F.loc[:"2023-12-01"].drop(columns="target"), F.loc[:"2023-12-01", "target"])
rf_lag_fc = ly.shift(1)[ts_test.index] + rf_ts.predict(F.loc[ts_test.index].drop(columns="target")); t_rflag = time.time() - t

# RF + ARIMA hybrid: RF learns ARIMA residuals
t = time.time()
resid_train = arima_res.resid.iloc[1:]                                      # first residual is not meaningful after differencing
resid_test = ts_test - arima_fc                                             # known after each month is observed
resid_all = pd.concat([resid_train, resid_test])
R = lag_frame(resid_all).assign(target=resid_all).dropna()
rf_res = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=RANDOM_STATE)
rf_res.fit(R.loc[:"2023-12-01"].drop(columns="target"), R.loc[:"2023-12-01", "target"])
hybrid_fc = arima_fc + rf_res.predict(R.loc[ts_test.index].drop(columns="target")); t_hyb = time.time() - t + t_arima

actual = np.exp(ts_test)
ts_rows = []
for name, fc, tt in [("Naive (last month)", ly.shift(1)[ts_test.index], 0.0), (f"ARIMA{(p_best, 1, q_best)}", arima_fc, t_arima),
                     ("SARIMA(1,1,1)(1,0,1,12)", sarima_fc, t_sarima), ("RF (lag features)", rf_lag_fc, t_rflag),
                     ("RF + ARIMA hybrid", hybrid_fc, t_hyb)]:
    m = regression_metrics(actual, np.exp(fc)); m.pop("R2")
    ts_rows.append({"Model": name, **m, "Train time (s)": tt})
ts_comp = pd.DataFrame(ts_rows).sort_values("MAE").reset_index(drop=True)
ts_comp.to_csv(REPORTS / "rice_ts_comparison.csv", index=False)
pd.DataFrame({"actual": actual, "ARIMA": np.exp(arima_fc), "SARIMA": np.exp(sarima_fc), "RF_lag": np.exp(rf_lag_fc),
              "RF_ARIMA_hybrid": np.exp(hybrid_fc)}).to_csv(REPORTS / "rice_forecasts.csv")
ts_comp.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE_%": "{:.2f}", "Train time (s)": "{:.2f}"})

# %%
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(np.exp(ly["2021-01-01":]), color="black", lw=2, label="actual")
ax.plot(np.exp(arima_fc), ls="--", label="ARIMA")
ax.plot(np.exp(sarima_fc), ls=":", label="SARIMA")
ax.plot(np.exp(hybrid_fc), color="#2ca02c", label="RF + ARIMA hybrid")
ax.axvline(pd.Timestamp("2024-01-01"), color="grey", lw=1); ax.text(pd.Timestamp("2024-01-15"), ax.get_ylim()[0] + 5, "test →")
ax.set_ylabel("LKR per kg"); ax.set_title("Colombo white rice – one-step-ahead forecasts"); ax.legend()
plt.tight_layout(); plt.savefig(FIG / "rice_forecast.png", dpi=120); plt.show()

# %% [markdown]
# **Rice result:** the naive forecast was best (MAE about Rs 2.8, MAPE 1.1 %), with ARIMA(1,1,0) very close (Rs 3.0). SARIMA, the RF + ARIMA hybrid and lag-feature RF were
# worse (Rs 3.8–4.0). Why: after the crisis, rice settled at about Rs 250/kg and stayed flat through 2024–25 (stable supply, partly helped by gazetted maximum retail prices), so there was
# almost nothing left to forecast. The models learned from the volatile 2008 and 2022 periods and expected more movement than happened. **Lesson:** a hybrid only helps when the
# residuals have a pattern; on a calm, flat series a simple model is the right choice.

# %% [markdown]
# ## 11. Model comparison (2025 test set, LKR)

# %%
comp = pd.DataFrame(results)
naive_mae, naive_rmse = comp.loc[0, "MAE"], comp.loc[0, "RMSE"]
comp["MAE vs naive"] = ((naive_mae - comp["MAE"]) / naive_mae * 100).round(2).astype(str) + " %"
comp["RMSE vs naive"] = ((naive_rmse - comp["RMSE"]) / naive_rmse * 100).round(2).astype(str) + " %"
comp = comp.sort_values("MAE").reset_index(drop=True)
best_name = comp.loc[comp["Family"] != "Baseline", "Model"].iloc[0]
print("Best model (lowest test MAE):", best_name)
comp.to_csv(REPORTS / "model_comparison.csv", index=False)
comp.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.4f}", "MAPE_%": "{:.2f}", "Train time (s)": "{:.1f}"}) \
    .highlight_min(subset=["MAE", "RMSE", "MAPE_%"], color="#c6efce").highlight_max(subset=["R2"], color="#c6efce")

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
order = comp.sort_values("MAE", ascending=False)
colors = order["Model"].map(lambda m: "#2ca02c" if m == best_name else ("#7f7f7f" if "Naive" in m or "Ridge" in m else "#4C72B0"))
axes[0].barh(order["Model"], order["MAE"], color=colors); axes[0].set_xlabel("MAE (LKR)"); axes[0].set_title("Test MAE (lower is better)")
axes[0].axvline(naive_mae, color="red", ls="--", lw=1, label="naive"); axes[0].legend()
axes[1].barh(order["Model"], order["RMSE"], color=colors); axes[1].set_xlabel("RMSE (LKR)"); axes[1].set_title("Test RMSE (lower is better)")
axes[1].axvline(naive_rmse, color="red", ls="--", lw=1)
for ax, col in zip(axes, ["MAE", "RMSE"]):
    lo = order[col].min(); ax.set_xlim(lo * 0.9, order[col].max() * 1.03)
plt.tight_layout(); plt.savefig(FIG / "model_comparison.png", dpi=120); plt.show()

# %% [markdown]
# ### Error by food category (MAE, LKR)

# %%
by_cat = pd.DataFrame({"category": test["category"].astype(str), "actual": y_test_price})
for m in ["Naive (last month)", "Random Forest", "XGBoost", "LightGBM", "CatBoost", best_name]:
    by_cat[m] = np.abs(test_preds[m] - y_test_price)
cat_table = by_cat.drop(columns="actual").groupby("category").mean().round(1)
cat_table.insert(0, "rows", by_cat.groupby("category").size())
cat_table = cat_table.loc[:, ~cat_table.columns.duplicated()]
cat_table

# %%
gain = ((cat_table["Naive (last month)"] - cat_table[best_name]) / cat_table["Naive (last month)"] * 100).sort_values()
ax = gain.plot.barh(figsize=(9, 3.8), color=["#C44E52" if v < 0 else "#2ca02c" for v in gain])
ax.axvline(0, color="black", lw=0.8); ax.set_xlabel("% lower MAE than naive (positive = model better)")
ax.set_title(f"{best_name} vs naive, by food category")
plt.tight_layout(); plt.savefig(FIG / "gain_by_category.png", dpi=120); plt.show()

# %% [markdown]
# ### When does the model help? Big price moves vs small moves

# %%
change = np.abs(y_test_price / test["lag_1"].values - 1)
moves = pd.DataFrame({"move": np.where(change > 0.20, "big move (> 20 %)", "small move (≤ 20 %)"),
                      "Naive": np.abs(test_preds["Naive (last month)"] - y_test_price),
                      best_name: np.abs(test_preds[best_name] - y_test_price)})
move_table = moves.groupby("move").agg(rows=("Naive", "size"), naive_MAE=("Naive", "mean"), model_MAE=(best_name, "mean")).round(1)
move_table["model better by"] = ((move_table["naive_MAE"] - move_table["model_MAE"]) / move_table["naive_MAE"] * 100).round(1).astype(str) + " %"
move_table

# %% [markdown]
# ### Actual vs predicted (best model) and example forecasts

# %%
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_test_price, test_preds[best_name], s=4, alpha=0.3)
lim = [y_test_price.min(), y_test_price.max()]
ax.plot(lim, lim, "r--", lw=1); ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Actual price (LKR)"); ax.set_ylabel("Predicted price (LKR)"); ax.set_title(f"Actual vs predicted – {best_name}\n(2025 test set)")
plt.tight_layout(); plt.savefig(FIG / "actual_vs_predicted.png", dpi=120); plt.show()

# %%
examples = [("Nuwara Eliya", "Carrots"), ("Colombo City", "Tomatoes"), ("Jaffna", "Onions (red)"), ("Colombo City", "Rice (white)")]
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
for ax, (mk, cm) in zip(axes.ravel(), examples):
    hist = clean_df[(clean_df["market"] == mk) & (clean_df["commodity"] == cm) & (clean_df["month"] >= "2024-01-01")]
    sel = ((test["market"] == mk) & (test["commodity"] == cm)).values
    ax.plot(hist["month"], hist["price"], color="black", marker="o", ms=3, label="actual")
    ax.plot(test.loc[sel, "month"], test_preds["Naive (last month)"][sel], ls="--", color="grey", label="naive")
    ax.plot(test.loc[sel, "month"], test_preds[best_name][sel], color="#2ca02c", marker="s", ms=3, label="stacking")
    ax.axvline(pd.Timestamp("2025-01-01"), color="red", lw=0.8); ax.set_title(f"{cm} – {mk}"); ax.set_ylabel("LKR")
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %y")); ax.tick_params(axis="x", rotation=30)
axes[0, 0].legend(); plt.tight_layout(); plt.savefig(FIG / "example_forecasts.png", dpi=120); plt.show()

# %% [markdown]
# ### Which model won and why
# * **Best model: Stacking (RF + XGBoost + LightGBM + CatBoost → Ridge).** On the 2025 test set it has the lowest MAE (Rs 63.0), RMSE (Rs 103.7) and MAPE (11.9 %),
#   and it is the only model that beats the naive "same as last month" forecast on **every** metric (MAE 3.7 % lower, RMSE 7.6 % lower).
# * **Why stacking won:** each single model makes different mistakes. The Ridge meta-model learned to rely mostly on **LightGBM and CatBoost**
#   (the two with native categorical handling) and to give RF and XGBoost small negative weights, which corrects their shared errors. Averaging different views reduces variance.
# * **RF vs gradient boosting:** among single models, LightGBM has the lowest RMSE (106.9), followed by RF (108.0), CatBoost (108.9) and XGBoost (109.8). MAE is almost a tie (65.5–66.1).
#   LightGBM also trained about 4x faster than RF. Boosting wins slightly because it reduces bias step by step and its Huber loss is less affected by extreme price spikes.
# * **Where the model helps:** on months with **big price moves (> 20 %)** the best model's error is about 21 % lower than naive, especially for upcountry vegetables
#   (carrots, tomatoes) and red onions. On **stable items** (rice, lentils, eggs) naive is almost perfect and the model adds small noise.
#   So the model is most useful exactly where shoppers and traders need it: volatile foods.
# * **Hybrids:** the RF + ANN **residual** hybrid was worse than RF alone: the MLP could not find a pattern in RF's errors and added noise. The weighted blend chose 95 % RF on training data,
#   which confirms the ANN adds little on this tabular data. Tree ensembles beat neural networks here.
# * **R² is not useful for ranking** here (about 0.97 for every model, even naive) because prices range from Rs 20 to Rs 3,000; MAE and "vs naive" are the fair comparison.

# %% [markdown]
# ### Save models and results for the dashboard

# %%
save = {"ridge": ridge, "random_forest": rf, "bagging": bag, "extra_trees": et, "xgboost": xgb, "lightgbm": lgbm,
        "catboost": cat, "stacking": stack, "ann": ann, "rf_ann_hybrid": hybrid, "rf_ann_blend": blend}
for k, m in save.items():
    joblib.dump(m, MODELS / f"{k}.joblib", compress=3)
name_to_file = {r["Model"]: k for r, k in zip(results[1:], save.keys())}
json.dump({"features": FEATURES, "cat_cols": CAT_COLS, "num_cols": NUM_COLS, "categories": CATEGORIES,
           "best_model": best_name, "model_files": name_to_file},
          open(MODELS / "meta.json", "w"), indent=2)
pred_df = test[["month", "market", "admin1", "admin2", "category", "commodity", "unit", "price"]].copy()
for k, v in test_preds.items():
    pred_df[k] = v
pred_df.to_csv(REPORTS / "test_predictions.csv", index=False)
print("Saved:", sorted(p.name for p in MODELS.glob("*")))
print("Total model size: %.1f MB" % (sum(p.stat().st_size for p in MODELS.glob("*.joblib")) / 1e6))

# %% [markdown]
# ## 12. Explainability: feature importance and SHAP
# * **Built-in importance:** how much each feature reduces error inside the trees (RF, XGBoost, LightGBM gain, CatBoost), scaled to sum to 1.
# * **SHAP (LightGBM):** for each prediction, how much each feature pushed the predicted % change up or down. LightGBM is used because it has the largest
#   weight in the stacking meta-model and SHAP's TreeExplainer is exact and fast for it.

# %%
def norm(v):
    v = np.asarray(v, dtype=float); return v / v.sum()
imp = pd.DataFrame({
    "Random Forest": norm(rf.named_steps["model"].feature_importances_),
    "XGBoost": norm(xgb.named_steps["model"].feature_importances_),
    "LightGBM (gain)": norm(lgbm.booster_.feature_importance("gain")),
    "CatBoost": norm(cat.feature_importances_)}, index=FEATURES)
imp["Average"] = imp.mean(axis=1)
imp = imp.sort_values("Average", ascending=False)
imp.to_csv(REPORTS / "feature_importance.csv")

top = imp.head(12).drop(columns="Average")[::-1]
ax = top.plot.barh(figsize=(10, 7), width=0.8)
ax.set_xlabel("Share of importance"); ax.set_title("Top 12 features – built-in importance by model")
plt.tight_layout(); plt.savefig(FIG / "feature_importance.png", dpi=120); plt.show()

# %%
import shap
X_shap = X_test.sample(2000, random_state=RANDOM_STATE)
explainer = shap.TreeExplainer(lgbm)
shap_values = explainer.shap_values(X_shap)
shap_imp = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES).sort_values(ascending=False)
shap_imp.rename("mean_abs_shap").to_csv(REPORTS / "shap_importance.csv")

shap.summary_plot(shap_values, X_shap, plot_type="bar", max_display=12, show=False)
plt.title("SHAP – mean impact on predicted % change (LightGBM)"); plt.tight_layout()
plt.savefig(FIG / "shap_bar.png", dpi=120); plt.show()

# %%
num_idx = [FEATURES.index(c) for c in NUM_COLS]          # beeswarm colours need numeric features
shap.summary_plot(shap_values[:, num_idx], X_shap[NUM_COLS], max_display=12, show=False)
plt.title("SHAP beeswarm – numeric features (LightGBM)"); plt.tight_layout()
plt.savefig(FIG / "shap_beeswarm.png", dpi=120); plt.show()

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for ax, f in zip(axes, ["rel_to_national", "lag1_vs_roll"]):
    i = FEATURES.index(f)
    ax.scatter(X_shap[f], shap_values[:, i], s=6, alpha=0.4)
    ax.axhline(0, color="black", lw=0.8); ax.set_xlabel(f); ax.set_ylabel("SHAP value (effect on log % change)")
    ax.set_title(f"How {f} changes the forecast")
axes[0].set_xlim(0.5, 1.6); axes[1].set_xlim(-0.5, 0.5)
plt.tight_layout(); plt.savefig(FIG / "shap_dependence.png", dpi=120); plt.show()
print(shap_imp.head(10).round(4).to_string())

# %% [markdown]
# ### What the model learned, in Sri Lankan terms
# * **Prices return to the national level (`rel_to_national`).** When a market was about 5 % *cheaper* than the national median last month, the model predicts a rise
#   (average SHAP about +4 %); when it was about 5 % *dearer*, it predicts a fall (about −3 %). Markets are linked through wholesale hubs like Dambulla and Manning Market,
#   so local price gaps close quickly.
# * **Spikes fade (`lag1_vs_roll`).** If last month's price was above its own 3-month average, the model expects it to come down, and the other way round.
# * **National momentum (`nat_change_1`).** If the national price of an item fell more than about 6 % last month, the model expects it to keep falling; a national rise pushes forecasts up.
#   Supply shocks (rain in the hill country, import changes) hit all markets together.
# * **Which item it is (`commodity`)** matters most overall: each food has its own trend in 2023–25 (for example pumpkin, eggs and bananas were pushed down, beans and pineapples up).
# * **Season (`month_sin`, `month_cos`)** captures the year-end rise and the March/August harvest falls seen in the EDA.
# * **Location columns (province, district, lat/lon) matter little** once `rel_to_national` is known: *how dear the market already is* tells the model more than *where it is*.
# * `ly_change` (last year's move) ranks high in built-in gain importance but lower in SHAP. Gain counts how often a feature is used to split; SHAP measures its real effect on predictions,
#   so SHAP is the more trustworthy view.
