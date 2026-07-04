"""
07_realtime_predict.py
───────────────────────
Direction A — Real-time / Online Phase Classifier.

Simulates a live telemetry stream by reading a flight CSV row-by-row
and predicting the flight phase at each decisecond tick, exactly as a
ground station would receive data from the rocket.

Usage:
    python 07_realtime_predict.py --csv your_flight.csv
    python 07_realtime_predict.py --csv your_flight.csv --model lstm --delay 100
    python 07_realtime_predict.py --csv your_flight.csv --no-smooth --no-live

Arguments:
    --csv          Path to flight CSV (required)
    --model        xgboost | randomforest | svm | lstm  (default: xgboost)
    --delay        Simulated inter-sample delay in ms   (default: 0 = as fast as possible)
    --no-smooth    Disable temporal smoothing
    --no-live      Skip live terminal display (useful for scripting)
    --output       Path for output CSV  (default: outputs/realtime_<name>_<model>.csv)

Output:
    - Live phase + confidence display in terminal (colour-coded)
    - Rolling output CSV: [timestamp, altitude, velocity, predicted_phase, confidence]
    - Summary timeline plot at end of flight
    - Phase transition event log printed at end
"""

import _compat  # UTF-8 console fix for Windows
import argparse
import os
import sys
import time
from collections import deque

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

import config
import utils


# ── ANSI colour helpers ─────────────────────────────────────────────
PHASE_COLORS_ANSI = {
    "Boost":   "\033[91m",   # red
    "Coast":   "\033[93m",   # yellow
    "Apogee":  "\033[95m",   # magenta
    "Descent": "\033[94m",   # blue
    "Landed":  "\033[92m",   # green
}
RESET = "\033[0m"


def colorize(text, phase):
    return f"{PHASE_COLORS_ANSI.get(phase, '')}{text}{RESET}"


# ── Model loading ───────────────────────────────────────────────────

class ClassicalPredictor:
    """Wraps a saved sklearn model for single-sample or batch prediction."""

    def __init__(self, model_name: str):
        saved, fname = utils.load_best_model(model_name)
        self.model        = saved["model"]
        self.scaler       = saved["scaler"]
        self.le           = saved["le"]
        self.feature_cols = saved["feature_cols"]
        print(f"  Loaded {model_name} model: {fname}")

    def predict_single(self, x_row: np.ndarray):
        """
        x_row : 1-D array of raw (un-scaled) features, length = n_features
        Returns (phase_str, confidence_float)
        """
        x_scaled   = self.scaler.transform(x_row.reshape(1, -1))
        pred_enc   = self.model.predict(x_scaled)
        pred_proba = self.model.predict_proba(x_scaled)
        phase      = self.le.inverse_transform(pred_enc)[0]
        confidence = float(pred_proba.max())
        return phase, confidence


class LSTMPredictor:
    """
    Wraps a saved Keras LSTM model + its companion _meta.pkl.
    Keeps an internal FIFO buffer; returns None until buffer is full.
    """

    def __init__(self):
        import tensorflow as tf
        self.tf = tf

        # Find best meta by F1
        meta_files = sorted([
            f for f in os.listdir(config.MODELS_DIR)
            if f.startswith("lstm_") and f.endswith("_meta.pkl")
        ])
        if not meta_files:
            raise FileNotFoundError(
                "No LSTM _meta.pkl found in outputs/models/.\n"
                "Run 04_lstm_model.py first."
            )
        best_meta = max(
            meta_files,
            key=lambda f: joblib.load(
                os.path.join(config.MODELS_DIR, f)
            ).get("macro_f1", 0.0)
        )
        meta = joblib.load(os.path.join(config.MODELS_DIR, best_meta))
        self.scaler       = meta["scaler"]
        self.le           = meta["le"]
        self.feature_cols = meta["feature_cols"]
        self.seq_len      = meta["seq_len"]
        print(f"  Loaded LSTM meta : {best_meta}  (F1={meta.get('macro_f1',0):.4f})")

        # Load Keras model
        keras_name = best_meta.replace("_meta.pkl", ".keras")
        keras_path = os.path.join(config.MODELS_DIR, keras_name)
        if not os.path.exists(keras_path):
            raise FileNotFoundError(f"LSTM model not found: {keras_path}")
        self.model = tf.keras.models.load_model(keras_path)
        print(f"  Loaded LSTM model: {keras_name}")

        # FIFO buffer holds scaled feature vectors
        self.buffer = deque(maxlen=self.seq_len)
        self._last_phase = config.PHASE_LABELS[0]
        self._last_conf  = 0.0

    def predict_single(self, x_row: np.ndarray):
        """
        x_row  : 1-D raw feature array
        Returns (phase_str, confidence_float) or last prediction while warming up
        """
        x_scaled = self.scaler.transform(x_row.reshape(1, -1))[0]
        self.buffer.append(x_scaled)

        if len(self.buffer) < self.seq_len:
            # Buffer still warming up — return last known or default
            return self._last_phase, 0.0

        seq = np.array(self.buffer).reshape(1, self.seq_len, -1)
        prob = self.model.predict(seq, verbose=0)[0]
        enc  = int(np.argmax(prob))
        self._last_phase = config.PHASE_LABELS[enc]
        self._last_conf  = float(prob.max())
        return self._last_phase, self._last_conf


class PhaseNetPredictor:
    """
    Wraps a saved PhaseNet .keras model + its companion _meta.pkl.
    Same FIFO buffer approach as LSTMPredictor.
    """

    def __init__(self):
        import tensorflow as tf
        self.tf = tf

        # Find best PhaseNet meta by F1
        meta_files = sorted([
            f for f in os.listdir(config.MODELS_DIR)
            if f.startswith("phasenet_") and f.endswith("_meta.pkl")
        ])
        if not meta_files:
            raise FileNotFoundError(
                "No PhaseNet _meta.pkl found in outputs/models/.\n"
                "Run 09_phasenet_model.py first."
            )
        best_meta = max(
            meta_files,
            key=lambda f: joblib.load(
                os.path.join(config.MODELS_DIR, f)
            ).get("macro_f1", 0.0)
        )
        meta = joblib.load(os.path.join(config.MODELS_DIR, best_meta))
        self.scaler       = meta["scaler"]
        self.le           = meta["le"]
        self.feature_cols = meta["feature_cols"]
        self.seq_len      = meta["seq_len"]
        print(f"  Loaded PhaseNet meta : {best_meta}  (F1={meta.get('macro_f1',0):.4f})")

        # Load Keras model
        keras_name = best_meta.replace("_meta.pkl", ".keras")
        keras_path = os.path.join(config.MODELS_DIR, keras_name)
        if not os.path.exists(keras_path):
            raise FileNotFoundError(f"PhaseNet model not found: {keras_path}")
        self.model = tf.keras.models.load_model(keras_path)
        print(f"  Loaded PhaseNet model: {keras_name}")

        # FIFO buffer
        self.buffer = deque(maxlen=self.seq_len)
        self._last_phase = config.PHASE_LABELS[0]
        self._last_conf  = 0.0

    def predict_single(self, x_row: np.ndarray):
        x_scaled = self.scaler.transform(x_row.reshape(1, -1))[0]
        self.buffer.append(x_scaled)

        if len(self.buffer) < self.seq_len:
            return self._last_phase, 0.0

        seq = np.array(self.buffer).reshape(1, self.seq_len, -1)
        prob = self.model.predict(seq, verbose=0)[0]
        enc  = int(np.argmax(prob))
        self._last_phase = config.PHASE_LABELS[enc]
        self._last_conf  = float(prob.max())
        return self._last_phase, self._last_conf


# ── Single-row feature extraction ──────────────────────────────────

def _make_row_dict(raw_row: pd.Series, prev_row,
                   prev_prev_row,
                   ground_alt: float, rolling_buf: deque,
                   w: int) -> dict:
    """
    Replicates utils.engineer_features for a single incoming row.
    Uses a rolling deque for rolling statistics — O(1) per tick.
    """
    alt = float(raw_row[config.COL_ALTITUDE])
    vel = float(raw_row[config.COL_VELOCITY])

    prev_alt  = float(prev_row[config.COL_ALTITUDE]) if prev_row is not None else alt
    prev_vel  = float(prev_row[config.COL_VELOCITY])  if prev_row is not None else vel

    alt_diff  = alt - prev_alt
    vel_diff  = vel - prev_vel

    if prev_prev_row is not None:
        prev_alt2  = float(prev_prev_row[config.COL_ALTITUDE])
        prev_diff2 = prev_alt - prev_alt2
        acc_proxy  = alt_diff - prev_diff2
    else:
        acc_proxy = 0.0

    # Rolling buffer holds the last w alt/vel values
    rolling_buf.append((alt, vel))
    alts_buf = [r[0] for r in rolling_buf]
    vels_buf = [r[1] for r in rolling_buf]

    alt_rm  = float(np.mean(alts_buf))
    vel_rm  = float(np.mean(vels_buf))
    alt_rs  = float(np.std(alts_buf))
    vel_rs  = float(np.std(vels_buf))

    speed_abs     = abs(vel)
    is_ascending  = int(vel > 0)
    vel_sign_now  = np.sign(vel)
    vel_sign_prev = np.sign(prev_vel)
    vel_sign_change = int(vel_sign_now != vel_sign_prev)
    energy_proxy  = 0.5 * vel**2 + config.GRAVITY * (alt - ground_alt)
    alt_from_gnd  = alt - ground_alt

    pyro1 = float(raw_row.get(config.COL_PYRO1, 0))
    pyro2 = float(raw_row.get(config.COL_PYRO2, 0))
    lat   = float(raw_row.get(config.COL_LAT, 0))
    lon   = float(raw_row.get(config.COL_LON, 0))
    batt  = float(raw_row.get(config.COL_BATTERY, 0))

    return {
        config.COL_ALTITUDE:   alt,
        config.COL_VELOCITY:   vel,
        config.COL_PYRO1:      pyro1,
        config.COL_PYRO2:      pyro2,
        config.COL_LAT:        lat,
        config.COL_LON:        lon,
        config.COL_BATTERY:    batt,
        "alt_diff":            alt_diff,
        "vel_diff":            vel_diff,
        "acc_proxy":           acc_proxy,
        "jerk_proxy":          0.0,   # would require 3 prev rows; set 0 for streaming
        "alt_rolling_mean":    alt_rm,
        "vel_rolling_mean":    vel_rm,
        "alt_rolling_std":     alt_rs,
        "vel_rolling_std":     vel_rs,
        "speed_abs":           speed_abs,
        "is_ascending":        is_ascending,
        "vel_sign_change":     vel_sign_change,
        "energy_proxy":        energy_proxy,
        "altitude_from_ground": alt_from_gnd,
    }


# ── Plot ─────────────────────────────────────────────────────────────

def plot_realtime_result(records: list, flight_name: str, model_name: str):
    """Produces a two-panel summary plot (altitude + confidence) coloured by phase."""
    phase_colors = {
        "Boost":   "#e74c3c",
        "Coast":   "#f39c12",
        "Apogee":  "#9b59b6",
        "Descent": "#2980b9",
        "Landed":  "#27ae60",
    }
    df = pd.DataFrame(records)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax = axes[0]
    for phase, color in phase_colors.items():
        mask = df["predicted_phase"] == phase
        ax.scatter(df.loc[mask, "timestamp"],
                   df.loc[mask, "altitude"],
                   c=color, label=phase, s=7, alpha=0.9)
    ax.set_ylabel("Altitude (m)")
    ax.set_title(
        f"Real-Time Phase Classification — {flight_name}  [{model_name}]\n"
        f"({len(df)} samples processed)",
        fontsize=13, fontweight="bold"
    )
    ax.legend(markerscale=3, fontsize=9)

    ax2 = axes[1]
    ax2.fill_between(df["timestamp"], df["confidence"],
                     alpha=0.3, color="#2c3e50")
    ax2.plot(df["timestamp"], df["confidence"],
             color="#2c3e50", linewidth=0.8)
    ax2.axhline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.6,
                label="50% threshold")
    ax2.set_ylabel("Confidence")
    ax2.set_xlabel(f"Time ({config.COL_TIME})")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)

    plt.tight_layout()
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    path = os.path.join(
        config.PLOTS_DIR,
        f"realtime_{flight_name}_{model_name.lower()}.png"
    )
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"\n  Summary plot saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Real-time (streaming) flight phase classifier."
    )
    parser.add_argument("--csv",      required=True,
                        help="Path to flight CSV (simulates live stream)")
    parser.add_argument("--model",    default=config.REALTIME_DEFAULT_MODEL,
                        choices=["xgboost", "randomforest", "svm", "lstm", "phasenet"],
                        help=f"Model to use (default: {config.REALTIME_DEFAULT_MODEL})")
    parser.add_argument("--delay",    type=int, default=0,
                        help="Simulated inter-sample delay in ms (default: 0)")
    parser.add_argument("--no-smooth", action="store_true",
                        help="Disable temporal smoothing of final predictions")
    parser.add_argument("--no-live",   action="store_true",
                        help="Suppress live terminal output (faster)")
    parser.add_argument("--output",    default=None,
                        help="Output CSV path (default: outputs/realtime_<name>_<model>.csv)")
    args = parser.parse_args()

    # ── Load flight CSV ───────────────────────────────────────────────
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    df_full = pd.read_csv(args.csv)
    df_full.columns = df_full.columns.str.strip()
    flight_name = os.path.splitext(os.path.basename(args.csv))[0]

    # Ground altitude (estimated from full file — simulates pre-launch calibration)
    alt_arr    = df_full[config.COL_ALTITUDE].astype(float).values
    ground_alt = float(np.percentile(alt_arr, 2))

    n_samples = len(df_full)
    print(f"\n{'='*60}")
    print(f"  STEP 7 — Real-Time Phase Prediction")
    print(f"{'='*60}")
    print(f"  Flight  : {flight_name}")
    print(f"  Model   : {args.model.upper()}")
    print(f"  Samples : {n_samples}")
    print(f"  Delay   : {args.delay} ms / sample")
    print(f"  Smoothing: {'OFF' if args.no_smooth else f'ON (window={config.SMOOTHING_WINDOW})'}")
    print(f"{'='*60}\n")

    # ── Load model ────────────────────────────────────────────────────
    if args.model == "lstm":
        predictor = LSTMPredictor()
    elif args.model == "phasenet":
        predictor = PhaseNetPredictor()
    else:
        predictor = ClassicalPredictor(args.model)

    feature_cols = [c for c in config.FEATURE_COLS
                    if c in (predictor.feature_cols
                             if hasattr(predictor, "feature_cols")
                             else config.FEATURE_COLS)]

    # ── Stream rows ───────────────────────────────────────────────────
    records      = []
    prev_row     = None
    prev_prev_row = None
    rolling_buf  = deque(maxlen=config.ROLLING_WINDOW)
    transitions  = []          # list of (idx, ts, old_phase, new_phase)
    last_phase   = None

    print("  [Streaming started]\n")
    if not args.no_live:
        print(f"  {'Tick':<6} {'Time':<12} {'Alt(m)':>9} {'Vel(m/s)':>10} "
              f"{'Phase':<12} {'Conf':>6}")
        print(f"  {'-'*60}")

    t_start = time.time()

    for idx, raw_row in df_full.iterrows():
        # Build feature row for this tick
        feat_dict = _make_row_dict(
            raw_row, prev_row, prev_prev_row,
            ground_alt, rolling_buf, config.ROLLING_WINDOW
        )
        x_row = np.array([feat_dict.get(c, 0.0) for c in feature_cols],
                         dtype=np.float32)

        phase, conf = predictor.predict_single(x_row)

        ts  = raw_row[config.COL_TIME]
        alt = raw_row[config.COL_ALTITUDE]
        vel = raw_row[config.COL_VELOCITY]

        records.append({
            "timestamp":       ts,
            "altitude":        alt,
            "velocity":        vel,
            "predicted_phase": phase,
            "confidence":      conf,
        })

        # Track transitions
        if last_phase is not None and phase != last_phase:
            transitions.append((idx, ts, last_phase, phase))
        last_phase = phase

        # Live display
        if not args.no_live:
            phase_display = colorize(f"{phase:<12}", phase)
            print(f"  {idx:<6} {ts:<12} {alt:>9.1f} {vel:>10.1f} "
                  f"{phase_display} {conf:>6.3f}")

        prev_prev_row = prev_row
        prev_row      = raw_row

        if args.delay > 0:
            time.sleep(args.delay / 1000.0)

    t_elapsed = time.time() - t_start
    print(f"\n  [Streaming done — {n_samples} samples in {t_elapsed:.2f}s "
          f"({n_samples/t_elapsed:.0f} samples/sec)]\n")

    # ── Temporal smoothing ────────────────────────────────────────────
    all_phases = [r["predicted_phase"] for r in records]
    all_confs  = [r["confidence"]      for r in records]

    if not args.no_smooth:
        all_phases = utils.smooth_predictions(all_phases)
        for i, r in enumerate(records):
            r["predicted_phase"] = all_phases[i]
        print(f"  Temporal smoothing applied (window={config.SMOOTHING_WINDOW})")

    # ── Phase distribution summary ────────────────────────────────────
    print(f"\n  Phase distribution:")
    from collections import Counter
    counts = Counter(all_phases)
    for phase in config.PHASE_LABELS:
        n   = counts.get(phase, 0)
        pct = 100 * n / n_samples
        bar = "█" * int(pct / 2)
        print(f"    {phase:10s}: {n:5d} samples  ({pct:5.1f}%)  {bar}")

    print(f"\n  Mean confidence : {np.mean(all_confs):.3f}")

    # ── Phase transition log ──────────────────────────────────────────
    print(f"\n  Phase transitions detected:")
    if transitions:
        for i, (idx, ts, old_p, new_p) in enumerate(transitions, 1):
            print(f"    {i:>2}. t={ts:<10}  {colorize(old_p, old_p):20s} → "
                  f"{colorize(new_p, new_p)}")
    else:
        print("    (none)")

    # ── Save output CSV ───────────────────────────────────────────────
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    out_csv = args.output or os.path.join(
        config.OUTPUT_DIR,
        f"realtime_{flight_name}_{args.model}.csv"
    )
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"\n  Output CSV saved → {out_csv}")

    # ── Summary plot ──────────────────────────────────────────────────
    plot_realtime_result(records, flight_name, args.model)

    print(f"\n✓ Real-time prediction complete.")


if __name__ == "__main__":
    main()
