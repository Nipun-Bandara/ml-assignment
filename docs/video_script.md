# KadeMila – 3-minute video: script + slide outline

**Format:** 8 slides, screen recording with your voice. Speak at a calm pace (about 145 words per minute).
The script below is about 400 spoken words, which fits 3:00 with a few seconds spare.
All numbers come from the notebook outputs (`reports/model_comparison.csv`, `reports/rice_ts_comparison.csv`).

---

## Slide 1 – Title + problem · 0:00–0:20
**On screen:** title "KadeMila – Is this food price fair?", your name, index number. Right side: `reports/figures/eda_02_rice_crisis.png` (rice price doubling).

**Say:**
> "Hi, I'm [name]. After the 2022 crisis, rice in Colombo doubled from about 110 to over 230 rupees a kilo, and never came back down.
> Tomatoes and carrots swing about 50 percent a month. Shoppers can't tell if a price is fair, or where it's heading."

## Slide 2 – Innovative idea · 0:20–0:35
**On screen:** three icons/boxes: 🟢 *Price fairness check* · 🚨 *Early warning* · 🗺️ *Regional price gap*.

**Say:**
> "My idea is KadeMila, which means 'shop price'. It forecasts next month's retail price of 33 foods in 36 Sri Lankan markets, checks whether a price you're quoted is fair,
> and gives an early warning for items expected to jump."

## Slide 3 – Data · 0:35–0:55
**On screen:** "WFP Sri Lanka Food Prices – HDX (Dept. of Census & Statistics, HARTI)" · key facts: 32,907 prices · 36 markets · 33 foods · 2004–Sep 2025. Image: `reports/figures/eda_07_regional_gap.png` or `eda_01_coverage.png`.

**Say:**
> "The data is WFP's official Sri Lanka food price dataset, from the Department of Census and Statistics and HARTI: about 33,000 monthly retail prices.
> I train on June 2023 to December 2024 and test on nine unseen months of 2025, with no shuffling."

## Slide 4 – Method (one diagram) · 0:55–1:30
**On screen:** `reports/figures/method_diagram.png`.

**Say:**
> "The features are recent prices, how dear the market is versus the national median, the national trend, last year's move, and the season.
> I trained all three ensemble types: bagging with Random Forest and Extra Trees; boosting with XGBoost, LightGBM and CatBoost; and stacking, where a Ridge model combines the best four.
> I also built hybrids: Random Forest plus a neural network, and Random Forest plus ARIMA on twenty years of rice prices.
> Every model must beat a simple baseline: 'same price as last month'."

## Slide 5 – Results · 1:30–1:55
**On screen:** `reports/figures/model_comparison.png` and a small table:

| Model | MAE (Rs) | RMSE (Rs) | vs naive (MAE) |
|---|---|---|---|
| **Stacking** | **63.0** | **103.7** | **3.7 % better** |
| LightGBM | 65.5 | 106.9 | ≈ same |
| Random Forest | 65.6 | 108.0 | ≈ same |
| Naive | 65.4 | 112.2 | – |

**Say:**
> "Stacking won: it's the only model that beats the baseline on every metric, with a mean error of 63 rupees.
> LightGBM beat Random Forest on RMSE and trained four times faster. The neural-network hybrids were worse; on tabular data, trees win.
> On big swings of over 20 percent, stacking cuts the error by 21 percent."

## Slide 6 – Why it works (SHAP) · 1:55–2:10
**On screen:** `reports/figures/shap_dependence.png` (or `shap_bar.png`).

**Say:**
> "SHAP shows the model learned real market behaviour: when a market is cheaper than the national price, it predicts a rise, and when it's dearer, a fall. Spikes fade, and national trends spread to every market."

## Slide 7 – Dashboard demo (live, screen recording) · 2:10–2:40
**On screen:** switch to the Streamlit app (`streamlit run app.py`). Practise this clicking path:
1. **Overview** → show the Sri Lanka map for Carrots (2 s).
2. **Model comparison** → bar chart with the 🏆 best model (3 s).
3. **Price check & forecast** → Colombo City, Carrots → type quoted price **450** → click **Forecast** → red "HIGH" message (15 s).
4. **Early warning** → the "rising more than 10 %" table (5 s).

**Say (while clicking):**
> "Here's the dashboard. The map shows which markets pay more for carrots, and here are all the models compared.
> The price check: carrots in Colombo are forecast at about 320 rupees for October. If a shop asks 450, KadeMila flags it as 40 percent too high.
> Early warning lists prices expected to jump next month."

## Slide 8 – Innovation + conclusion · 2:40–3:00
**On screen:** 3 bullets: *What's new* · *Limitations* · *Next steps*.

**Say:**
> "What's new is turning ensemble forecasts into a fairness check and early warning for Sri Lankan shoppers, traders and aid programmes.
> Limitations: about two years of full data, and no weather or fuel inputs yet. Next: rainfall, exchange rates, and a Sinhala and Tamil SMS bot. Thank you."

---

## Recording tips
* Record slides and dashboard in one take with OBS Studio or PowerPoint's *Record* feature; export as **MP4 (1080p)**.
* Start the dashboard **before** recording and click through each page once, so the models are already cached and pages load instantly.
* Time yourself with a stopwatch; if you run long, cut the Slide 6 sentence about spikes and the map step in the demo.
* Check the numbers on Slide 5 match your final notebook run.
