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
    ax.tick_params(axis='x', labelsize=10)
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
    ax.set_xlabel("Model", fontsize=12)
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
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved comparison chart → {out_path}")


def plot_per_fold_lines(summary_per_fold_path, out_path):
    """
    Per-fold macro F1 line plot — one line per model across the 7 LOROCV
    folds. Addresses reviewer Comment 6: this conveys per-fold behaviour
    (which rockets are hard, which models are stable) that the aggregate
    Table V / bar-chart mean±std cannot show, rather than re-presenting
    the same two summary numbers in a second visual form.
    """
    with open(summary_per_fold_path) as f:
        data = json.load(f)

    by_model = {}
    for row in data:
        by_model.setdefault(row["model"], {})[row["fold"]] = row["macro_f1"]

    models = sorted(by_model.keys())
    all_folds = sorted(
        {fold for rows in by_model.values() for fold in rows},
        key=lambda s: int(s.rsplit("_", 1)[-1])
    )
    fold_labels = [f.replace("fold_rocket_", "Rocket ") for f in all_folds]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X']

    for i, model in enumerate(models):
        f1s = [by_model[model].get(fold, float('nan')) for fold in all_folds]
        ax.plot(fold_labels, f1s, marker=markers[i % len(markers)],
                color=MODEL_COLORS.get(model, colors[i]), label=model,
                linewidth=1.8, markersize=7)

    ax.set_xlabel("Test Rocket (LOROCV Fold)", fontsize=12)
    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_title("Per-Fold Macro F1 Across All Models (7-Fold LOROCV)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved per-fold line plot → {out_path}")


if __name__ == "__main__":
    main()
    perfold_path = os.path.join(config.OUTPUT_DIR, "model_summary_perFold.json")
    if os.path.exists(perfold_path):
        plot_per_fold_lines(
            perfold_path,
            os.path.join(config.PLOTS_DIR, "per_fold_f1_lines.png")
        )
