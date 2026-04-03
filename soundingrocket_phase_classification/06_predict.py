"""
06_predict.py
──────────────
Given a new flight CSV, predicts the phase for every row.

Usage:
    python 06_predict.py --csv your_flight.csv --model xgboost
    python 06_predict.py --csv your_flight.csv --model lstm --no-smooth

Model options: xgboost | randomforest | svm | lstm

Fixes in this version:
  - engineer_features imported from utils.py (single source of truth)
  - LSTM scaler loaded from companion _meta.pkl (no fit-on-test-data bug)
  - Best fold chosen by macro F1 from model_summary_perFold.json
  - Optional temporal smoothing (--no-smooth to disable)
  - Confidence score (max class probability) added to output CSV

Output:
    - Prints phase for each row in the terminal
    - Saves a new CSV with 'predicted_phase' and 'confidence' columns
    - Saves a timeline plot (altitude coloured by predicted phase)
"""

import _compat  # UTF-8 console fix for Windows
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import config
import utils


# ──────────────────────────────────────────────
# LSTM prediction  (uses saved meta for correct scaler)
# ──────────────────────────────────────────────

def predict_lstm(X_raw: np.ndarray, n_samples_original: int):
    """
    Returns (predictions: list[str], confidences: list[float]).

    Loads the best LSTM fold's .keras model together with its _meta.pkl
    (which contains the fitted StandardScaler from training).
    """
    import tensorflow as tf

    # ── Find best lstm _meta.pkl by macro_f1 ────────────────────────
    meta_files = sorted([
        f for f in os.listdir(config.MODELS_DIR)
        if f.startswith("lstm_") and f.endswith("_meta.pkl")
    ])
    if not meta_files:
        raise FileNotFoundError(
            "No LSTM _meta.pkl found. Run 04_lstm_model.py first.\n"
            "(Old models saved without meta need to be retrained.)"
        )
    best_meta = max(
        meta_files,
        key=lambda f: joblib.load(
            os.path.join(config.MODELS_DIR, f)
        ).get("macro_f1", 0.0)
    )
    meta = joblib.load(os.path.join(config.MODELS_DIR, best_meta))
    scaler       = meta["scaler"]
    le           = meta["le"]
    feature_cols = meta["feature_cols"]
    seq_len      = meta["seq_len"]
    print(f"  Using LSTM meta : {best_meta}  (F1={meta.get('macro_f1',0):.4f})")

    # ── Scale using the training scaler ─────────────────────────────
    X_scaled = scaler.transform(X_raw)

    # ── Load corresponding .keras ────────────────────────────────────
    keras_name = best_meta.replace("_meta.pkl", ".keras")
    keras_path = os.path.join(config.MODELS_DIR, keras_name)
    if not os.path.exists(keras_path):
        raise FileNotFoundError(f"LSTM model not found: {keras_path}")
    print(f"  Using LSTM model: {keras_name}")
    model = tf.keras.models.load_model(keras_path)

    # ── Build sequences ──────────────────────────────────────────────
    sequences = []
    for i in range(seq_len, len(X_scaled)):
        sequences.append(X_scaled[i - seq_len:i])
    sequences = np.array(sequences)

    probs       = model.predict(sequences, verbose=0)
    pred_enc    = np.argmax(probs, axis=1)
    pred_labels = [config.PHASE_LABELS[i] for i in pred_enc]
    confs       = probs.max(axis=1).tolist()

    # Pad first seq_len rows with the first predicted label/confidence
    pad_label = pred_labels[0] if pred_labels else config.PHASE_LABELS[0]
    pad_conf  = confs[0] if confs else 1.0
    return ([pad_label] * seq_len + pred_labels,
            [pad_conf]  * seq_len + confs)


# ──────────────────────────────────────────────
# PhaseNet prediction  (uses saved meta for correct scaler)
# ──────────────────────────────────────────────

def predict_phasenet(X_raw: np.ndarray, n_samples_original: int):
    """
    Returns (predictions: list[str], confidences: list[float]).
    Loads the best PhaseNet fold's .keras model + _meta.pkl.
    """
    import tensorflow as tf

    meta_files = sorted([
        f for f in os.listdir(config.MODELS_DIR)
        if f.startswith("phasenet_") and f.endswith("_meta.pkl")
    ])
    if not meta_files:
        raise FileNotFoundError(
            "No PhaseNet _meta.pkl found. Run 09_phasenet_model.py first."
        )
    best_meta = max(
        meta_files,
        key=lambda f: joblib.load(
            os.path.join(config.MODELS_DIR, f)
        ).get("macro_f1", 0.0)
    )
    meta = joblib.load(os.path.join(config.MODELS_DIR, best_meta))
    scaler   = meta["scaler"]
    seq_len  = meta["seq_len"]
    print(f"  Using PhaseNet meta : {best_meta}  (F1={meta.get('macro_f1',0):.4f})")

    X_scaled = scaler.transform(X_raw)

    keras_name = best_meta.replace("_meta.pkl", ".keras")
    keras_path = os.path.join(config.MODELS_DIR, keras_name)
    if not os.path.exists(keras_path):
        raise FileNotFoundError(f"PhaseNet model not found: {keras_path}")
    print(f"  Using PhaseNet model: {keras_name}")
    model = tf.keras.models.load_model(keras_path)

    sequences = []
    for i in range(seq_len, len(X_scaled)):
        sequences.append(X_scaled[i - seq_len:i])
    sequences = np.array(sequences)

    probs       = model.predict(sequences, verbose=0)
    pred_enc    = np.argmax(probs, axis=1)
    pred_labels = [config.PHASE_LABELS[i] for i in pred_enc]
    confs       = probs.max(axis=1).tolist()

    pad_label = pred_labels[0] if pred_labels else config.PHASE_LABELS[0]
    pad_conf  = confs[0] if confs else 1.0
    return ([pad_label] * seq_len + pred_labels,
            [pad_conf]  * seq_len + confs)


# ──────────────────────────────────────────────
# Find best saved classical model (metric-driven)
# ──────────────────────────────────────────────

def find_classical_model(model_name: str):
    """
    Uses utils.load_best_model to pick the fold .pkl with highest F1.
    Returns (saved_bundle_dict, filename_used).
    """
    return utils.load_best_model(model_name)


# ──────────────────────────────────────────────
# Plot predicted phases
# ──────────────────────────────────────────────

def plot_prediction(df, flight_name, model_name):
    phase_colors = {
        "Boost":   "#e74c3c",
        "Coast":   "#f39c12",
        "Apogee":  "#9b59b6",
        "Descent": "#2980b9",
        "Landed":  "#27ae60",
    }
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

    # Top: altitude coloured by phase
    ax = axes[0]
    for phase, color in phase_colors.items():
        mask = df["predicted_phase"] == phase
        ax.scatter(df.loc[mask, config.COL_TIME],
                   df.loc[mask, config.COL_ALTITUDE],
                   c=color, label=phase, s=6, alpha=0.85)
    ax.set_ylabel("Altitude (m)")
    ax.set_title(f"Predicted Phases — {flight_name}  [{model_name}]",
                 fontsize=13, fontweight="bold")
    ax.legend(markerscale=3, fontsize=9)

    # Bottom: confidence
    ax2 = axes[1]
    ax2.plot(df[config.COL_TIME], df["confidence"],
             color="#2c3e50", linewidth=0.8, alpha=0.8)
    ax2.set_ylabel("Confidence")
    ax2.set_xlabel(f"Time ({config.COL_TIME})")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.5,
                label="50% threshold")
    ax2.legend(fontsize=8)

    plt.tight_layout()

    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    plot_path = os.path.join(
        config.PLOTS_DIR,
        f"prediction_{flight_name}_{model_name.lower()}.png"
    )
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n  Plot saved → {plot_path}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Predict flight phases for a new rocket CSV."
    )
    parser.add_argument("--csv",   required=True,
                        help="Path to your flight CSV file")
    parser.add_argument("--model", default="xgboost",
                        choices=["xgboost", "randomforest", "svm", "lstm", "phasenet"],
                        help="Which model to use for prediction (default: xgboost)")
    parser.add_argument("--no-smooth", action="store_true",
                        help="Disable temporal smoothing of predictions")
    args = parser.parse_args()

    # ── Load CSV ─────────────────────────────────────────────────────
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()
    flight_name = os.path.splitext(os.path.basename(args.csv))[0]
    print(f"\n{'='*55}")
    print(f"  Predicting phases for: {flight_name}")
    print(f"  Model: {args.model.upper()}")
    print(f"  Rows : {len(df)}")
    print(f"{'='*55}")

    # ── Feature engineering (imported from utils) ─────────────────
    alt = df[config.COL_ALTITUDE].astype(float)
    ground_alt = float(np.percentile(alt.values, 2))
    df_feat = utils.engineer_features(df, flight_id=-1, ground_alt=ground_alt)
    feature_cols = [c for c in config.FEATURE_COLS if c in df_feat.columns]
    X = df_feat[feature_cols].fillna(0).values

    # ── Predict ──────────────────────────────────────────────────────
    if args.model == "lstm":
        predictions, confidences = predict_lstm(X, len(df))

    elif args.model == "phasenet":
        predictions, confidences = predict_phasenet(X, len(df))

    else:
        saved, fname = find_classical_model(args.model)
        model   = saved["model"]
        scaler  = saved["scaler"]
        le      = saved["le"]

        X_scaled    = scaler.transform(X)
        pred_enc    = model.predict(X_scaled)
        predictions = le.inverse_transform(pred_enc).tolist()

        # Confidence = max class probability
        pred_proba  = model.predict_proba(X_scaled)
        confidences = pred_proba.max(axis=1).tolist()

    # ── Optional temporal smoothing ────────────────────────────────
    if not args.no_smooth:
        predictions = utils.smooth_predictions(predictions)
        print(f"  Temporal smoothing applied (window={config.SMOOTHING_WINDOW})")

    # ── Attach to dataframe ────────────────────────────────────────
    df["predicted_phase"] = predictions
    df["confidence"]      = confidences

    # ── Print summary ──────────────────────────────────────────────
    print(f"\n  Phase distribution:")
    counts = df["predicted_phase"].value_counts()
    for phase in config.PHASE_LABELS:
        n   = counts.get(phase, 0)
        pct = 100 * n / len(df)
        bar = "█" * int(pct / 2)
        print(f"    {phase:10s}: {n:5d} rows  ({pct:5.1f}%)  {bar}")

    mean_conf = np.mean(confidences)
    print(f"\n  Mean confidence : {mean_conf:.3f}")

    # ── Print first rows ───────────────────────────────────────────
    print(f"\n  Sample output (first 10 rows):")
    print(f"  {'Row':<6} {'Time':<12} {'Altitude':>10} {'Velocity':>10} {'Phase':<12} {'Conf':>6}")
    print(f"  {'-'*60}")
    for i, row in df.head(10).iterrows():
        print(f"  {i:<6} {row[config.COL_TIME]:<12} "
              f"{row[config.COL_ALTITUDE]:>10.1f} "
              f"{row[config.COL_VELOCITY]:>10.1f} "
              f"  {row['predicted_phase']:<12} "
              f"{row['confidence']:>6.3f}")

    # ── Save output CSV ────────────────────────────────────────────
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out_csv = os.path.join(
        config.OUTPUT_DIR,
        f"predicted_{flight_name}_{args.model}.csv"
    )
    df.to_csv(out_csv, index=False)
    print(f"\n  Full CSV saved → {out_csv}")

    # ── Plot ───────────────────────────────────────────────────────
    plot_prediction(df, flight_name, args.model)
    print(f"\n✓ Done.")


if __name__ == "__main__":
    main()