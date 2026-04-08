"""
08_apogee_predictor.py
───────────────────────
Direction C — Apogee Altitude Regression.

Given ONLY the Boost phase sensor data of a flight (first ~5 seconds of
powered ascent), predict the peak altitude (apogee) that the rocket will
reach.  This has direct applications in:
    - Pre-flight range safety estimation
    - Parachute deployment decisions
    - Post-flight quick verification

Approach:
    1. Load labeled feature dataset (outputs/features_dataset.csv)
    2. For each flight, extract Boost-phase statistics as input features
    3. Target = max(altitude[m]) for that flight
    4. Train XGBoost Regressor (Leave-One-Rocket-Out CV)
    5. Evaluate: RMSE, RMSE%, and R²
    6. Save best model → outputs/models/apogee_predictor.pkl
    7. Produce plots:
       - Predicted vs Actual scatter
       - Per-rocket bar chart (actual vs predicted apogee)
       - Feature importance

Usage:
    python 08_apogee_predictor.py              # train + evaluate
    python 08_apogee_predictor.py --predict --csv your_flight.csv
"""

import _compat  # UTF-8 console fix for Windows
import argparse
import os
import json
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import xgboost as xgb

import config
import utils


# ─────────────────────────────────────────────────────────────────────
# BOOST FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────

def extract_boost_features(df_flight: pd.DataFrame, flight_label: str) -> Optional[dict]:
    """
    Extracts summary statistics from the Boost phase of a single flight.

    Returns a dict with keys matching config.APOGEE_BOOST_FEATURES,
    plus 'flight_label' and 'actual_apogee' (regression target).
    Returns None if the flight has no Boost samples.
    """
    boost_mask = df_flight[config.LABEL_COL] == "Boost"
    if boost_mask.sum() == 0:
        print(f"    WARNING: {flight_label} — no Boost samples found, skipping.")
        return None

    df_boost = df_flight[boost_mask].copy()
    alt_boost = df_boost[config.COL_ALTITUDE].astype(float).values
    vel_boost = df_boost[config.COL_VELOCITY].astype(float).values

    # acc_proxy is pre-computed in the feature dataset
    if "acc_proxy" in df_boost.columns:
        acc_boost = df_boost["acc_proxy"].astype(float).values
    else:
        acc_boost = np.diff(vel_boost, prepend=vel_boost[0])

    duration       = len(df_boost)
    max_vel        = float(np.max(vel_boost))
    mean_vel       = float(np.mean(vel_boost))
    max_acc        = float(np.max(acc_boost))
    mean_acc       = float(np.mean(acc_boost))
    alt_at_burnout = float(alt_boost[-1])
    first_vel      = float(vel_boost[0])
    vel_rate       = (max_vel - first_vel) / max(duration, 1)

    ground_alt     = float(np.percentile(df_flight[config.COL_ALTITUDE].astype(float).values, 2))
    energy_burnout = 0.5 * vel_boost[-1]**2 + config.GRAVITY * (alt_at_burnout - ground_alt)

    actual_apogee  = float(df_flight[config.COL_ALTITUDE].astype(float).max())

    return {
        "flight_label":          flight_label,
        "boost_max_velocity":    max_vel,
        "boost_mean_velocity":   mean_vel,
        "boost_max_acc":         max_acc,
        "boost_mean_acc":        mean_acc,
        "boost_duration":        float(duration),
        "boost_alt_at_burnout":  alt_at_burnout,
        "boost_vel_rate":        vel_rate,
        "boost_energy_at_burnout": energy_burnout,
        "actual_apogee":         actual_apogee,
    }


# ─────────────────────────────────────────────────────────────────────
# TRAIN + EVALUATE  (Leave-One-Rocket-Out)
# ─────────────────────────────────────────────────────────────────────

def build_regressor():
    return xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )


def train_and_evaluate(records: List[dict]):
    """
    Trains an XGBoost regressor using Leave-One-Rocket-Out CV.

    Parameters
    ----------
    records : list of dicts from extract_boost_features()

    Returns
    -------
    predictions : list of (flight_label, actual, predicted)
    best_model  : trained XGBRegressor on all data
    feature_names : list of feature column names
    """
    feature_names = config.APOGEE_BOOST_FEATURES
    labels   = [r["flight_label"] for r in records]
    X_all    = np.array([[r[f] for f in feature_names] for r in records],
                        dtype=np.float32)
    y_all    = np.array([r["actual_apogee"] for r in records], dtype=np.float32)
    n        = len(records)

    predictions = []
    rmses       = []

    print(f"\n  Leave-One-Rocket-Out CV ({n} rockets):")
    print(f"  {'Rocket':<25} {'Actual':>10} {'Predicted':>12} {'Error%':>8}")
    print(f"  {'-'*60}")

    for i in range(n):
        train_mask = [j for j in range(n) if j != i]
        test_idx   = i

        X_tr = X_all[train_mask]
        y_tr = y_all[train_mask]
        X_te = X_all[[test_idx]]
        y_te = y_all[test_idx]

        scaler   = StandardScaler()
        X_tr_s   = scaler.fit_transform(X_tr)
        X_te_s   = scaler.transform(X_te)

        model    = build_regressor()
        model.fit(X_tr_s, y_tr)

        y_pred   = model.predict(X_te_s)[0]
        error_pct = 100.0 * abs(y_pred - y_te) / max(y_te, 1.0)
        rmses.append((y_pred - y_te)**2)

        predictions.append((labels[i], float(y_te), float(y_pred)))
        print(f"  {labels[i]:<25} {y_te:>10.1f} {y_pred:>12.1f} {error_pct:>7.2f}%")

    overall_rmse = float(np.sqrt(np.mean(rmses)))
    overall_rmse_pct = 100.0 * overall_rmse / float(np.mean(y_all))
    r2 = r2_score(
        [p[1] for p in predictions],
        [p[2] for p in predictions]
    )
    print(f"\n  Overall RMSE     : {overall_rmse:.1f} m")
    print(f"  RMSE%            : {overall_rmse_pct:.2f}%")
    print(f"  R²               : {r2:.4f}")

    # Train final model on all data
    scaler_final = StandardScaler()
    X_all_s      = scaler_final.fit_transform(X_all)
    best_model   = build_regressor()
    best_model.fit(X_all_s, y_all)

    return predictions, best_model, scaler_final, feature_names, {
        "rmse_m":      round(overall_rmse, 2),
        "rmse_pct":    round(overall_rmse_pct, 4),
        "r2":          round(r2, 6),
        "n_rockets":   n,
    }


# ─────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────

def plot_actual_vs_predicted(predictions: list, metrics: dict):
    """Scatter plot of actual vs predicted apogee."""
    labels   = [p[0] for p in predictions]
    actuals  = [p[1] for p in predictions]
    preds    = [p[2] for p in predictions]

    fig, ax = plt.subplots(figsize=(7, 6))

    # Diagonal reference line
    lim = (min(min(actuals), min(preds)) * 0.95,
           max(max(actuals), max(preds)) * 1.05)
    ax.plot(lim, lim, "k--", lw=1, alpha=0.5, label="Perfect prediction")

    scatter = ax.scatter(actuals, preds, s=120, zorder=5,
                         c=range(len(labels)), cmap="tab10", alpha=0.9)

    # Annotate each point with the rocket name
    for label, a, p in zip(labels, actuals, preds):
        ax.annotate(label.split("-")[0].strip(),
                    (a, p), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, alpha=0.8)

    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual Apogee (m)", fontsize=12)
    ax.set_ylabel("Predicted Apogee (m)", fontsize=12)
    ax.set_title(
        f"Apogee Predictor — Actual vs Predicted\n"
        f"RMSE={metrics['rmse_m']:.1f} m  ({metrics['rmse_pct']:.2f}%)  R²={metrics['r2']:.4f}",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9)
    plt.tight_layout()

    path = os.path.join(config.PLOTS_DIR, "apogee_scatter.png")
    plt.savefig(path, dpi=180); plt.close()
    print(f"  Saved scatter → {path}")


def plot_per_rocket_bars(predictions: list):
    """Side-by-side bar chart comparing actual vs predicted apogee per rocket."""
    labels   = [p[0].split("-")[0].strip()[:15] for p in predictions]
    actuals  = [p[1] for p in predictions]
    preds    = [p[2] for p in predictions]

    x     = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.5), 5))
    bars_a = ax.bar(x - width/2, actuals, width, label="Actual Apogee",
                    color="#2980b9", alpha=0.88)
    bars_p = ax.bar(x + width/2, preds,   width, label="Predicted Apogee",
                    color="#e74c3c", alpha=0.88)

    # Value labels
    for bar in list(bars_a) + list(bars_p):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 20,
                f"{bar.get_height():.0f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Altitude (m)")
    ax.set_title("Apogee Predictor — Per-Rocket Comparison",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    plt.tight_layout()

    path = os.path.join(config.PLOTS_DIR, "apogee_per_rocket.png")
    plt.savefig(path, dpi=180); plt.close()
    print(f"  Saved bar chart → {path}")


def plot_feature_importance_reg(model, feature_names: list):
    """Feature importance bar chart for the regression model."""
    imps    = model.feature_importances_
    n       = len(feature_names)
    indices = np.argsort(imps)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(range(n), imps[indices], color="#27ae60", alpha=0.88)
    ax.set_yticks(range(n))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
    ax.set_xlabel("Feature Importance")
    ax.set_title("Apogee Predictor — Feature Importances",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(config.PLOTS_DIR, "apogee_feature_importance.png")
    plt.savefig(path, dpi=180); plt.close()
    print(f"  Saved feature importance → {path}")


# ─────────────────────────────────────────────────────────────────────
# INFERENCE ON NEW FLIGHT
# ─────────────────────────────────────────────────────────────────────

def predict_apogee_from_csv(csv_path: str):
    """
    Given a raw (unlabeled) CSV, uses the saved apogee predictor to
    estimate the peak altitude.  Uses ALL available data as a proxy for
    Boost (since we don't have labels) or, if the file has been labeled,
    uses the Boost rows only.
    """
    if not os.path.exists(config.APOGEE_PREDICTOR_PATH):
        raise FileNotFoundError(
            f"No saved apogee predictor at {config.APOGEE_PREDICTOR_PATH}.\n"
            f"Run 08_apogee_predictor.py (without --predict) first."
        )

    saved    = joblib.load(config.APOGEE_PREDICTOR_PATH)
    model    = saved["model"]
    scaler   = saved["scaler"]
    features = saved["feature_names"]
    metrics  = saved["metrics"]

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    flight_name = os.path.splitext(os.path.basename(csv_path))[0]

    # If labeled, use Boost rows; else use first 20% of data
    if config.LABEL_COL in df.columns:
        boost_mask = df[config.LABEL_COL] == "Boost"
        df_boost   = df[boost_mask] if boost_mask.sum() > 0 else df.head(max(5, len(df)//5))
        print(f"  Using labeled Boost phase ({boost_mask.sum()} rows)")
    else:
        n_boost = max(5, len(df) // 5)
        df_boost = df.head(n_boost)
        print(f"  No labels found — using first {n_boost} rows as proxy for Boost")

    dummy_record = extract_boost_features(
        df.assign(**{config.LABEL_COL: "Boost"}), flight_name
    )
    if dummy_record is None:
        print("  Could not extract Boost features.")
        return

    X_raw = np.array([[dummy_record[f] for f in features]], dtype=np.float32)
    X_s   = scaler.transform(X_raw)
    pred  = model.predict(X_s)[0]

    print(f"\n  +--------------------------------------+")
    print(f"  | Flight   : {flight_name[:28]:<28}|")
    print(f"  | Predicted Apogee: {pred:>7.1f} m           |")
    print(f"  | Model RMSE:       {metrics['rmse_m']:>7.1f} m  ({metrics['rmse_pct']:.2f}%)  |")
    print(f"  +--------------------------------------+\n")
    return pred


# ─────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Apogee altitude predictor from Boost-phase telemetry."
    )
    parser.add_argument("--predict", action="store_true",
                        help="Predict apogee for a new flight (requires --csv)")
    parser.add_argument("--csv",     default=None,
                        help="Path to new flight CSV (used with --predict)")
    args = parser.parse_args()

    utils.ensure_dirs()

    # ── Inference mode ────────────────────────────────────────────────
    if args.predict:
        if args.csv is None:
            parser.error("--predict requires --csv <path>")
        predict_apogee_from_csv(args.csv)
        return

    # ── Training mode ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  STEP 8 — Apogee Altitude Predictor")
    print("="*60)

    if not os.path.exists(config.FEATURES_PATH):
        raise FileNotFoundError(
            f"Feature dataset not found: {config.FEATURES_PATH}\n"
            f"Run 01→02 first."
        )

    df_all = pd.read_csv(config.FEATURES_PATH)
    df_all.columns = df_all.columns.str.strip()
    print(f"\n  Loaded {len(df_all)} samples from {config.FEATURES_PATH}")

    if config.LABEL_COL not in df_all.columns:
        raise ValueError(f"Column '{config.LABEL_COL}' not found. "
                         "Run 01_label_data.py + 02_feature_engineering.py first.")

    if "flight_id" not in df_all.columns:
        raise ValueError("Column 'flight_id' not found. "
                         "Run 02_feature_engineering.py first.")

    # ── Extract one record per flight ─────────────────────────────────
    flight_ids = sorted(df_all["flight_id"].unique())
    records = []
    print(f"\n  Extracting Boost-phase features from {len(flight_ids)} flights:")
    for fid in flight_ids:
        df_flight = df_all[df_all["flight_id"] == fid].copy()
        # Attempt to recover a readable label from first row (for naming)
        flight_label = f"Rocket-{int(fid)}"
        rec = extract_boost_features(df_flight, flight_label)
        if rec is not None:
            records.append(rec)

    if len(records) < 3:
        raise ValueError(
            f"Only {len(records)} flights have Boost data — need at least 3 for CV."
        )

    print(f"\n  Using {len(records)} flights for regression.\n")

    # ── Train & Evaluate ──────────────────────────────────────────────
    predictions, best_model, scaler_final, feature_names, metrics = \
        train_and_evaluate(records)

    # ── Save model bundle ─────────────────────────────────────────────
    bundle = {
        "model":         best_model,
        "scaler":        scaler_final,
        "feature_names": feature_names,
        "metrics":       metrics,
    }
    joblib.dump(bundle, config.APOGEE_PREDICTOR_PATH)
    print(f"\n  Saved predictor → {config.APOGEE_PREDICTOR_PATH}")

    # ── Save metrics JSON ─────────────────────────────────────────────
    metrics_path = os.path.join(config.OUTPUT_DIR, "apogee_predictor_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics   → {metrics_path}")

    # ── Plots ─────────────────────────────────────────────────────────
    plot_actual_vs_predicted(predictions, metrics)
    plot_per_rocket_bars(predictions)
    plot_feature_importance_reg(best_model, feature_names)

    print(f"\n{'='*60}")
    print(f"  ✓ Apogee Predictor complete")
    print(f"  ✓ RMSE  : {metrics['rmse_m']:.1f} m  ({metrics['rmse_pct']:.2f}%)")
    print(f"  ✓ R²    : {metrics['r2']:.4f}")
    print(f"  ✓ Plots → outputs/plots/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
