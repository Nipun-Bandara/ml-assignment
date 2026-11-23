# KadeMila – upload checklist (due Monday, 23 November 2026)

## Files to upload
| # | Item | File(s) | Status |
|---|---|---|---|
| 1 | Python notebook, run top to bottom with all outputs | `KadeMila.ipynb` | ✅ runs with 0 errors (about 8–10 min) · ☐ fill in name / index / module |
| 2 | 3-minute presentation video (MP4) | `video/KadeMila_presentation.mp4` | ☐ record (script: `docs/video_script.md`) |
| 3 | Slides used in the video (PDF / PPTX) | `slides/KadeMila_slides.pdf` | ☐ add your name on the cover, export from the deck (Share › Export) |
| 4 | Dataset (CSV) | `data/raw/wfp_food_prices_lka.csv`, `data/raw/wfp_markets_lka.csv` (+ `data/processed/`) | ✅ (downloaded from HDX, no scraping script needed) |
| 5 | Data preparation code | `src/prepare_data.py` | ✅ |
| 6 | Dashboard code | `app.py`, `src/kademila_utils.py` | ✅ all 7 pages tested |
| 7 | Requirements | `requirements.txt` | ✅ |
| 8 | Results and figures | `reports/` (comparison tables, predictions, 19 figures) | ✅ |
| 9 | How to run | `README.md` | ✅ |
| 10 | Trained models (optional) | `models/` (~210 MB) | Optional: the notebook recreates them. Include only if there is no size limit |

## Steps on the day
1. Open `KadeMila.ipynb`, replace `[Your name]`, `[Your index number]`, `[Module code / name]` in the first cell, then **Kernel → Restart & Run All** and save.
   (Edit the notebook directly from now on; do not run `src/build_notebook.py` again, it would overwrite your edits.)
2. Fill in the deck's cover, export it as PDF (and PPTX if wanted) into `slides/`.
3. Record the video (about 3:00) into `video/`.
4. Build the zip:
   ```bash
   python src/make_submission.py                 # small zip, no models (~7 MB + video)
   python src/make_submission.py --with-models   # only if large uploads are allowed
   ```
   It prints OK / MISSING for the notebook, app, requirements, dataset, video, slides and your name.
5. Upload `submission/KadeMila_submission.zip` (and the MP4 separately if the LMS wants it separate) before the deadline.

## Before you upload, confirm with the lecturer
- Exact deadline time and upload location (LMS / Drive).
- Individual or group submission.
- Is a separate Streamlit app OK (it is also described in notebook section 14)?
- Any file size or naming rules (e.g. index number in the file name), and whether you need to appear on camera.
