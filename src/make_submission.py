"""
Packs everything to upload into one zip.

    python src/make_submission.py                  # without trained models (~10 MB; the notebook recreates them)
    python src/make_submission.py --with-models    # include models/ (~200 MB) so the dashboard runs without retraining

Put your video in  video/   (e.g. video/KadeMila_presentation.mp4)
Put your slides in slides/  (exported PDF and/or PPTX)
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WITH_MODELS = "--with-models" in sys.argv
OUT = ROOT / "submission" / ("KadeMila_submission_with_models.zip" if WITH_MODELS else "KadeMila_submission.zip")

INCLUDE = ["KadeMila.ipynb", "app.py", "requirements.txt", "requirements-notebook.txt", "README.md",
           "src", "data", "reports", "docs", "video", "slides"] + (["models"] if WITH_MODELS else [])
SKIP_PARTS = {"__pycache__", ".ipynb_checkpoints", "catboost_info", ".claude", "submission"}

files = []
for item in INCLUDE:
    p = ROOT / item
    if p.is_file():
        files.append(p)
    elif p.is_dir():
        files += [f for f in p.rglob("*") if f.is_file() and not SKIP_PARTS & set(f.relative_to(ROOT).parts)]

OUT.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(files):
        z.write(f, Path("KadeMila") / f.relative_to(ROOT))

print(f"Wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1e6:.1f} MB, {len(files)} files)")
checks = {
    "Notebook": (ROOT / "KadeMila.ipynb").exists(),
    "Dashboard app.py": (ROOT / "app.py").exists(),
    "requirements.txt": (ROOT / "requirements.txt").exists(),
    "Dataset CSV": (ROOT / "data" / "raw" / "wfp_food_prices_lka.csv").exists(),
    "Video (video/*.mp4)": any((ROOT / "video").glob("*.mp4")) if (ROOT / "video").exists() else False,
    "Slides (slides/*.pdf or *.pptx)": any(f.suffix.lower() in {".pdf", ".pptx"} for f in (ROOT / "slides").glob("*")) if (ROOT / "slides").exists() else False,
    "Name filled in notebook": "[Your name]" not in (ROOT / "KadeMila.ipynb").read_text(encoding="utf-8"),
}
for k, ok in checks.items():
    print(("  OK      " if ok else "  MISSING ") + k)
