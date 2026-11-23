"""Draws the one-slide method diagram: reports/figures/method_diagram.png"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[1] / "reports" / "figures" / "method_diagram.png"
fig, ax = plt.subplots(figsize=(16, 7.5))
ax.set_xlim(0, 16); ax.set_ylim(0, 7.5); ax.axis("off")


def box(x, y, w, h, title, body="", color="#4C72B0"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", fc=color, ec="none", alpha=0.92))
    ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="top", fontsize=13, weight="bold", color="white")
    if body:
        ax.text(x + w / 2, y + h - 0.85, body, ha="center", va="top", fontsize=10.5, color="white", linespacing=1.4)


def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=2, color="#444"))


box(0.2, 2.6, 2.6, 2.6, "WFP / DCS / HARTI data", "32,907 retail prices\n36 markets · 33 foods\n2004 – Sep 2025", "#555555")
box(3.4, 2.6, 2.8, 2.6, "Features", "last 1–3 prices\nvs national median\nnational momentum\nlast year's move\nmonth · market · item", "#555555")
arrow(2.85, 3.9, 3.35, 3.9)

box(7.0, 5.0, 3.4, 2.1, "BAGGING", "Random Forest\nBagging · Extra Trees", "#4C72B0")
box(7.0, 2.7, 3.4, 2.1, "BOOSTING", "XGBoost · LightGBM\nCatBoost", "#DD8452")
box(7.0, 0.4, 3.4, 2.1, "HYBRID", "RF + ANN (residual / blend)\nRF + ARIMA (rice series)", "#8172B3")
for y in (6.05, 3.75, 1.45):
    arrow(6.25, 3.9, 6.95, y)

box(11.2, 3.0, 2.3, 2.6, "STACKING", "RF + XGB + LGBM\n+ CatBoost\n→ Ridge meta-model", "#2ca02c")
arrow(10.45, 6.0, 11.15, 4.9); arrow(10.45, 3.75, 11.15, 4.1)

box(14.0, 3.0, 1.85, 2.6, "Dashboard", "price check\nearly warning\nmap · SHAP", "#C44E52")
arrow(13.55, 4.3, 13.95, 4.3)

ax.text(8.0, 7.35, "Target: next month's % price change   ·   Train Jun 2023 – Dec 2024   ·   Test Jan – Sep 2025 (no shuffling)",
        ha="center", fontsize=12, style="italic", color="#333")
plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
print("saved", OUT)
