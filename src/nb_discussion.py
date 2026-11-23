# %% [markdown]
# ## 13. Research discussion
# ### 13.1 Random Forest vs gradient boosting (XGBoost, LightGBM, CatBoost)

# %%
rf_vs_gb = comp[comp["Model"].isin(["Random Forest", "XGBoost", "LightGBM", "CatBoost"])][
    ["Model", "MAE", "RMSE", "MAPE_%", "Train time (s)"]].sort_values("RMSE").reset_index(drop=True)
rf_vs_gb.style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "MAPE_%": "{:.2f}", "Train time (s)": "{:.1f}"})

# %% [markdown]
# | | **Random Forest (bagging)** | **Gradient boosting (XGBoost, LightGBM, CatBoost)** |
# |---|---|---|
# | **How trees are built** | Many deep trees built **independently, in parallel**, each on a bootstrap sample with random feature subsets; predictions are averaged. | Shallow trees built **one after another**; each new tree is fitted to the errors (gradients) of the trees so far. |
# | **What it reduces** | **Variance**: averaging many noisy trees gives a stable model. | **Bias**: each step corrects what the model still gets wrong. |
# | **Speed** | Easy to parallelise, but deep trees are slow and large (our RF: about 88 s including tuning, about 31 MB on disk). | LightGBM is the fastest here (about 23 s incl. tuning) thanks to histogram binning and leaf-wise growth; XGBoost about 40 s; CatBoost about 89 s. |
# | **Tuning effort** | Low: works well with defaults; main knobs are `max_features` and `min_samples_leaf`. | Higher: learning rate, number of trees, depth/leaves, subsampling and loss all interact. |
# | **Overfitting risk** | Low: more trees never hurt; deep trees are averaged out. | Higher: too many trees or a high learning rate memorise noise, so you need early stopping, regularisation or CV. |
# | **Categorical features** | sklearn needs encoding (we used ordinal codes). | **CatBoost**: built-in ordered target statistics; **LightGBM**: native optimal category splits; **XGBoost**: needs encoding (or its experimental categorical mode). |
# | **Robust loss** | Squared error (absolute error is very slow in sklearn). | Easy to switch to absolute / Huber loss, which helps with price spikes (LightGBM picked Huber). |
#
# **Which wins on our data and why?** The four models are very close (MAE Rs 65.5–66.1). **LightGBM has the lowest RMSE (106.9)** and is the fastest, so it is the best
# single model. Boosting edges out RF because (1) food price changes depend on interactions like *item × season × how dear the market is*, which boosting learns step by step,
# (2) its Huber loss is not dragged around by extreme vegetable spikes, and (3) LightGBM and CatBoost handle the 36 markets and 33 items as true categories. RF is a close
# second with almost no tuning, so it remains a strong, safe default. The **stacked combination beat all of them**, which shows that RF and the boosters make *different*
# errors, and blending them is more valuable than picking one winner.

# %% [markdown]
# ### 13.2 Strengths and limitations of tree ensembles: tabular data vs image / text data
# **On tabular data (like ours), tree ensembles are usually the best choice:**
# * They handle **mixed feature types** (prices, ratios, months, market names) with little preprocessing, and no scaling.
# * They capture **non-linear effects and interactions** automatically (e.g. carrots in Nuwara Eliya in December).
# * They are **robust to outliers and irrelevant features**, and boosting libraries handle **missing values** natively.
# * They train fast on CPUs, need little data compared with deep learning, and are **explainable** with feature importance and SHAP.
# * Benchmarks agree: tree ensembles still beat deep learning on typical tabular datasets (Grinsztajn et al., 2022). Our results match this: the MLP (ANN) was the weakest non-linear model.
#
# **Limitations on tabular data:** trees **cannot extrapolate** beyond values seen in training (that is why we predict the % change instead of the price level),
# they do not understand time order by themselves (we had to build lag features), and big forests can be large and slow to load.
#
# **On image and text data, tree ensembles are weak:**
# * An image is thousands of raw pixels whose meaning comes from **spatial patterns** (edges, shapes). Trees split on one pixel at a time and cannot learn
#   these patterns; **CNNs and vision transformers** do this far better.
# * Text meaning depends on **word order and context**. Trees on bag-of-words / TF-IDF features ignore order and struggle with very high-dimensional sparse data;
#   **transformer models (e.g. BERT)** are much stronger.
# * Ensembles still help in a supporting role: gradient boosting on **features extracted by a neural network** (image embeddings, sentence embeddings), or as a fast baseline
#   for simple text classification.
#
# **Summary:** structured rows and columns → tree ensembles (bagging / boosting / stacking); raw pixels, audio or language → deep learning, possibly with an ensemble on top.

# %% [markdown]
# ## 14. Dashboard (Streamlit)
# The dashboard is in **`app.py`** and loads the models and results saved above (`models/`, `reports/`), so it starts quickly and does not retrain anything.
#
# ```
# streamlit run app.py        # opens http://localhost:8501
# ```
#
# | Page | What it shows |
# |---|---|
# | Overview | Dataset size, sample rows, national price trends, **map of markets coloured by how dear they are vs the national median**, item volatility |
# | Model comparison | Metrics table of all models (best marked 🏆), bar chart with selectable metric and the naive line, error by category |
# | Model explorer | **Model selector** (all 12 models), actual vs predicted, 2025 forecast line for any market and item, % error per item |
# | Explainability | Built-in feature importance per model, SHAP importance and dependence plots |
# | Rice forecast | RF + ARIMA hybrid vs ARIMA, SARIMA and lag-RF on Colombo rice |
# | Price check & forecast | **"Make a prediction" form**: choose market, item and model → next-month forecast in LKR with a typical range; enter a quoted price → *high / fair / low* verdict; what-if on last month's price |
# | Early warning | Forecast for every market and item; lists those expected to rise or fall more than a chosen % next month, filter by province |
#
# [Optional: paste 1–2 dashboard screenshots here, e.g. `![Price check](reports/figures/dashboard_price_check.png)`]

# %% [markdown]
# ## 15. Conclusion & limitations
# **Conclusion**
# * Using official WFP / DCS / HARTI market data, we forecast next month's retail price of 33 foods in 36 Sri Lankan markets with bagging, boosting, stacking and hybrid models.
# * **Stacking (RF + XGBoost + LightGBM + CatBoost → Ridge) was the best model** (test MAE Rs 63.0, MAPE 11.9 %) and the only one that beat the naive "same as last month" forecast on every metric.
#   Among single models, **LightGBM** was the most accurate (lowest RMSE) and fastest.
# * Next month's price is mostly last month's price, but on **big price moves (> 20 %)**, the ones that hurt households most, the best model **cuts the error by about 21 %**,
#   especially for upcountry vegetables and onions.
# * SHAP shows the model learned real market behaviour: **local prices move back toward the national level, spikes fade, and national trends spread to all markets.**
# * RF + ANN hybrids did not help: on tabular data, the neural network added noise. On Colombo rice, prices were flat after the crisis, so simple models (naive, ARIMA) were best.
# * The **KadeMila dashboard** turns this into a practical tool: a price-fairness check for shoppers and an early-warning list for traders and aid programmes.
#
# **Limitations**
# * Wide market coverage only starts in **June 2023**, so models learn from about 1.5 years and are tested on one 9-month period; results could differ in another year.
# * The data are **monthly survey averages**, not real shop quotes, and some months are missing (11 rice months were interpolated).
# * No outside drivers yet: **rainfall, fuel prices, the exchange rate, import rules** strongly affect Sri Lankan food prices.
# * sklearn stacking uses consecutive blocks for out-of-fold predictions, so some blocks are predicted by models that saw later months (a mild optimistic bias).
# * The ANN was only lightly tuned; a better-tuned network might do better.
#
# **Future improvements**
# * Add Department of Meteorology rainfall, CBSL exchange rates and CEYPETCO fuel prices as features.
# * **Smart switching:** use the ensemble for volatile items and the naive forecast for stable ones (rice, lentils), choosing per item with cross-validation.
# * Prediction intervals with conformal prediction instead of the simple ± average-error band.
# * Weekly HARTI price data, multi-month forecasts, and an SMS / WhatsApp price-check bot in **Sinhala and Tamil**.

# %% [markdown]
# ## 16. References
# 1. World Food Programme (2025). *Sri Lanka – Food Prices* [dataset]. Humanitarian Data Exchange. https://data.humdata.org/dataset/wfp-food-prices-for-sri-lanka (sources: Department of Census and Statistics, HARTI, WFP).
# 2. Breiman, L. (1996). Bagging predictors. *Machine Learning*, 24(2), 123–140.
# 3. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
# 4. Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized trees. *Machine Learning*, 63(1), 3–42.
# 5. Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.
# 6. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD 2016*, 785–794.
# 7. Ke, G., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems 30*.
# 8. Prokhorenkova, L., et al. (2018). CatBoost: Unbiased boosting with categorical features. *Advances in Neural Information Processing Systems 31*.
# 9. Wolpert, D. H. (1992). Stacked generalization. *Neural Networks*, 5(2), 241–259.
# 10. Zhang, G. P. (2003). Time series forecasting using a hybrid ARIMA and neural network model. *Neurocomputing*, 50, 159–175.
# 11. Box, G. E. P., & Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*. Holden-Day.
# 12. Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems 30*.
# 13. Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS 2022 Datasets and Benchmarks Track*.
# 14. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
