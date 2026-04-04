"""
05_compare_models.py  (optional)
──────────────────────────────────
Reads model_summary.json and produces a publication-ready bar chart
comparing all 4 models on mean macro F1 ± std.
Run AFTER steps 03 and 04.
"""

import _compat  # UTF-8 console fix for Windows
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import config
import utils

# Publication style
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.axisbelow": True,
})

MODEL_COLORS = {
    "XGBoost":      "#e74c3c",
    "PI-XGB":       "#fc8e42",  # Distinct orange for the physics-informed version
    "RandomForest": "#2ecc71",
    "SVM":          "#3498db",
    "LSTM":         "#9b59b6",
    "PhaseNet":     "#e94560",
}


def main():
    utils.ensure_dirs()
    summary_path = os.path.join(config.OUTPUT_DIR, "model_summary.json")
    if not os.path.exists(summary_path):
        print("model_summary.json not found. Run steps 03 and 04 first.")
        return

    with open(summary_path) as f:
        data = json.load(f)

    # Aggregate by model (average across folds if multiple entries)
    agg = {}
    for row in data:
        m = row["model"]
        agg.setdefault(m, []).append((row["mean_macro_f1"], row.get("std_macro_f1", 0)))

    models, means, stds = [], [], []
    for m, vals in agg.items():
        means_arr = [v[0] for v in vals]
        stds_arr  = [v[1] for v in vals]
        models.append(m)
        means.append(np.mean(means_arr))
        stds.append(np.mean(stds_arr))

    # Sort by mean F1
    order = np.argsort(means)[::-1]
    models = [models[i] for i in order]
    means  = [means[i]  for i in order]
    stds   = [stds[i]   for i in order]

    colors = [MODEL_COLORS.get(m, "#95a5a6") for m in models]

    # ── Plot ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color=colors, alpha=0.88, width=0.5,
                  error_kw={"elinewidth": 2, "ecolor": "black"})

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Sounding Rocket Flight Phase Classification\nModel Comparison — Macro F1 (Leave-One-Rocket-Out)",
        fontsize=13, fontweight="bold", pad=12
    )

    # Value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.015,
            f"{mean:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    plt.tight_layout()
    out_path = os.path.join(config.PLOTS_DIR, "model_comparison.png")
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved comparison chart → {out_path}")


if __name__ == "__main__":
    main()
