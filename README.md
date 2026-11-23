# KadeMila – Next-month food price forecasts for Sri Lankan markets

Ensemble Learning Dashboard assignment: bagging, boosting, stacking and hybrid models on Sri Lankan retail food prices
(WFP / HDX, compiled from the Department of Census and Statistics, HARTI and WFP market surveys).

## How to run
```bash
pip install -r requirements.txt
python src/prepare_data.py                 # clean data + build features (also run inside the notebook)
jupyter notebook KadeMila.ipynb            # Kernel → Restart & Run All (≈ 8–10 min, trains and saves all models)
streamlit run app.py                       # dashboard at http://localhost:8501
python src/make_submission.py              # zip everything for upload (see docs/SUBMISSION_CHECKLIST.md)
```

## Files
| Path | What it is |
|---|---|
| `KadeMila.ipynb` | Main notebook: data, cleaning, EDA, all models, comparison, explainability, research discussion |
| `app.py` | Streamlit dashboard (loads models saved by the notebook) |
| `src/prepare_data.py` | Data cleaning and feature engineering |
| `src/kademila_utils.py` | Shared helpers and custom model classes (CatBoost wrapper, RF + ANN hybrid, blend) |
| `src/nb_models.py`, `src/build_notebook.py` | Source used to generate the notebook |
| `data/raw/` | Original WFP CSVs (downloaded from data.humdata.org, no scraping) |
| `data/processed/` | Cleaned data, modelling table, next-month features, rice series |
| `models/` | Trained models (`.joblib`) – recreated by running the notebook |
| `reports/` | Comparison tables, test predictions, feature importance, figures |
| `docs/` | Video script and upload checklist |
| `video/`, `slides/` | Put the presentation MP4 and exported slides here |

## Data source
WFP *Sri Lanka – Food Prices*, Humanitarian Data Exchange: https://data.humdata.org/dataset/wfp-food-prices-for-sri-lanka (CC BY-IGO).
