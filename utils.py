# utils.py — Shared helpers for the pipeline
import matplotlib
matplotlib.use("Agg")
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
import joblib
import config


# ─────────────────────────────────────────────
# DIRECTORY / IO HELPERS
# ─────────────────────────────────────────────

def ensure_dirs():
    for d in [config.OUTPUT_DIR, config.LABELED_DIR,
              config.MODELS_DIR, config.PLOTS_DIR]:
        os.makedirs(d, exist_ok=True)


def load_flight_csvs(data_dir=config.DATA_DIR):
    """Load all CSVs from data_dir. Returns {stem: DataFrame}."""
    flights = {}
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in '{data_dir}/'")
    for f in csv_files:
        path = os.path.join(data_dir, f)
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        flights[os.path.splitext(f)[0]] = df
        print(f"  Loaded: {f}  →  {len(df)} rows, {df.shape[1]} cols")
    return flights


def save_model(obj, name):
    path = os.path.join(config.MODELS_DIR, f"{name}.pkl")
    joblib.dump(obj, path)
    print(f"  Saved model → {path}")


def load_model(name):
    return joblib.load(os.path.join(config.MODELS_DIR, f"{name}.pkl"))


def load_best_model(model_name: str):
    """
    Loads the fold-level .pkl that achieved the highest macro F1
    for the given model type, using model_summary.json as the oracle.

    Falls back to the alphabetically first available fold if the
    summary file is missing or has no matching entry.
    """
    prefix = model_name.lower().replace(" ", "_").replace("-", "_")
    all_pkls = sorted([
        f for f in os.listdir(config.MODELS_DIR)
        if f.startswith(prefix) and f.endswith(".pkl")
    ])
    if not all_pkls:
        raise FileNotFoundError(
            f"No saved model for '{model_name}' in {config.MODELS_DIR}/\n"
            f"Run 03_train_evaluate.py first."
        )

    # Try to find per-fold F1 from summary JSON
    summary_path = os.path.join(config.OUTPUT_DIR, "model_summary_perFold.json")
    best_file = None
    if os.path.exists(summary_path):
        try:
            with open(summary_path) as f:
                data = json.load(f)
            # data: list of {model, fold, macro_f1}
            candidates = [d for d in data if d["model"] == model_name]
            if candidates:
                best = max(candidates, key=lambda d: d["macro_f1"])
                fold_tag = best["fold"]
                matched = [f for f in all_pkls if fold_tag in f]
                if matched:
                    best_file = matched[0]
        except Exception:
            pass

    if best_file is None:
        # Fall back: prefer fold_rocket_1 heuristic, then first alphabetically
        preferred = [f for f in all_pkls if "fold_rocket_1" in f]
        best_file = preferred[0] if preferred else all_pkls[0]

    path = os.path.join(config.MODELS_DIR, best_file)
    print(f"  Loading model: {best_file}")
    return joblib.load(path), best_file


# ─────────────────────────────────────────────
# FEATURE ENGINEERING  (single source of truth)
# Used by: 02_feature_engineering.py, 06_predict.py, 07_realtime_predict.py
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, flight_id: int = -1,
                      ground_alt: float = None) -> pd.DataFrame:
    """
    Computes all derived features in-place on a copy of df.

    Parameters
    ----------
    df          : DataFrame with at minimum altitude and velocity columns
    flight_id   : integer flight identifier (appended as column, -1 for inference)
    ground_alt  : launch-site altitude (m).  If None, uses 2nd-percentile of alt.

    Returns
    -------
    DataFrame with original columns + engineered feature columns.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()
    w = config.ROLLING_WINDOW

    alt = df[config.COL_ALTITUDE].astype(float)
    vel = df[config.COL_VELOCITY].astype(float)

    # ── Diff-based ───────────────────────────────────────────────────
    df["alt_diff"]   = alt.diff().fillna(0)
    df["vel_diff"]   = vel.diff().fillna(0)
    df["acc_proxy"]  = df["alt_diff"].diff().fillna(0)
    df["jerk_proxy"] = df["acc_proxy"].diff().fillna(0)   # NEW: rate of change of accel

    # ── Rolling statistics ───────────────────────────────────────────
    df["alt_rolling_mean"] = alt.rolling(w, min_periods=1).mean()
    df["vel_rolling_mean"] = vel.rolling(w, min_periods=1).mean()
    df["alt_rolling_std"]  = alt.rolling(w, min_periods=1).std().fillna(0)
    df["vel_rolling_std"]  = vel.rolling(w, min_periods=1).std().fillna(0)

    # ── Scalar / event features ──────────────────────────────────────
    df["speed_abs"]      = vel.abs()
    df["is_ascending"]   = (vel > 0).astype(int)

    # NEW: velocity sign change (strong apogee/transition signal)
    vel_sign = np.sign(vel.values)
    sign_shifted = np.roll(vel_sign, 1)
    sign_shifted[0] = vel_sign[0]
    df["vel_sign_change"] = (vel_sign != sign_shifted).astype(int)

    # NEW: mechanical energy proxy (kinetic + potential, per unit mass)
    g_alt = ground_alt if ground_alt is not None else float(np.percentile(alt.values, 2))
    df["energy_proxy"]        = 0.5 * vel**2 + config.GRAVITY * (alt - g_alt)

    # NEW: altitude above launch site (removes elevation bias across rockets)
    df["altitude_from_ground"] = alt - g_alt

    # ── Pyro columns (default 0 if missing) ─────────────────────────
    if config.COL_PYRO1 not in df.columns:
        df[config.COL_PYRO1] = 0
    if config.COL_PYRO2 not in df.columns:
        df[config.COL_PYRO2] = 0

    if flight_id >= 0:
        df["flight_id"] = flight_id

    return df


# ─────────────────────────────────────────────
# POST-PREDICTION TEMPORAL SMOOTHING
# ─────────────────────────────────────────────

def smooth_predictions(predictions: list, window: int = None) -> list:
    """
    Applies a sliding-mode (majority-vote) filter over predictions
    to eliminate isolated mis-classifications.

    Parameters
    ----------
    predictions : list of str  (phase label per sample)
    window      : odd integer window size (default: config.SMOOTHING_WINDOW)

    Returns
    -------
    smoothed list of str
    """
    if window is None:
        window = config.SMOOTHING_WINDOW
    if window < 3:
        return predictions

    preds = np.array(predictions)
    out   = preds.copy()
    half  = window // 2

    for i in range(half, len(preds) - half):
        window_slice = preds[i - half: i + half + 1]
        # Mode — pick most frequent label in window
        values, counts = np.unique(window_slice, return_counts=True)
        out[i] = values[np.argmax(counts)]

    return out.tolist()


# ─────────────────────────────────────────────
# EVALUATION HELPERS
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, model_name, labels=config.PHASE_LABELS):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted Phase")
    ax.set_ylabel("True Phase")
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR,
                        f"cm_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved confusion matrix → {path}")


def plot_roc_curves(y_true, y_score, model_name, labels=config.PHASE_LABELS):
    """One-vs-Rest ROC curves. y_score: (n_samples, n_classes) probabilities."""
    y_bin = label_binarize(y_true, classes=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.tab10.colors

    for i, label in enumerate(labels):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f"{label} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0.0, 1.0]); ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{model_name} — ROC Curves (OvR)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR,
                        f"roc_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved ROC curves → {path}")


def print_classification_report(y_true, y_pred, model_name):
    print(f"\n{'='*55}")
    print(f"  {model_name} — Classification Report")
    print(f"{'='*55}")
    print(classification_report(y_true, y_pred,
                                labels=config.PHASE_LABELS,
                                zero_division=0))


def plot_feature_importance(model, feature_names, model_name, top_n=15):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    n = min(top_n, len(feature_names))
    indices = np.argsort(importances)[::-1][:n]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(n), importances[indices][::-1], color="steelblue")
    ax.set_yticks(range(n))
    ax.set_yticklabels([feature_names[i] for i in indices[::-1]], fontsize=9)
    ax.set_xlabel("Gain Importance (XGBoost)")
    ax.set_ylabel("Feature")
    ax.set_title(f"{model_name} — Top {n} Feature Importances",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR,
                        f"feat_imp_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved feature importance → {path}")


def plot_phase_timeline(df, flight_name):
    """Altitude coloured by phase label for visual verification."""
    phase_colors = {
        "Boost":   "#e74c3c",
        "Coast":   "#f39c12",
        "Apogee":  "#9b59b6",
        "Descent": "#2980b9",
        "Landed":  "#27ae60",
    }
    fig, ax = plt.subplots(figsize=(12, 4))
    for phase, color in phase_colors.items():
        mask = df[config.LABEL_COL] == phase
        ax.scatter(df.loc[mask, config.COL_TIME],
                   df.loc[mask, config.COL_ALTITUDE],
                   c=color, label=phase, s=5, alpha=0.8)
    ax.set_xlabel("Time (deciseconds; 1 ds = 0.1 s)")
    ax.set_ylabel("Altitude (m)")
    ax.set_title(f"Flight: {flight_name} — Phase Labels",
                 fontsize=13, fontweight="bold")
    ax.legend(markerscale=3, fontsize=9)
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, f"timeline_{flight_name}.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"  Saved timeline → {path}")
