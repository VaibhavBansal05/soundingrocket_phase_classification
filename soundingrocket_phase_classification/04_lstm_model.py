"""
04_lstm_model.py
─────────────────
Bidirectional LSTM classifier using sliding-window sequences.
Uses the same FEATURE_COLS as the classical models (defined in config.py
and engineered in utils.engineer_features).
Evaluation protocol: Leave-One-Rocket-Out (same as step 03).

FIX: Now saves a companion _meta.pkl file alongside every .keras file,
     containing {scaler, le, feature_cols, seq_len}.
     This is loaded by 06_predict.py and 07_realtime_predict.py so that
     predictions are made with the correct normalisation.
"""

import _compat  # UTF-8 console fix for Windows
import os, json, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import joblib

import config, utils


def build_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def load_dataset():
    df = pd.read_csv(config.FEATURES_PATH)
    df.columns = df.columns.str.strip()
    feature_cols = [c for c in config.FEATURE_COLS if c in df.columns]
    X          = df[feature_cols].values.astype(np.float32)
    y_raw      = df[config.LABEL_COL].values
    flight_ids = df["flight_id"].values if "flight_id" in df.columns else None
    le         = LabelEncoder()
    le.fit(config.PHASE_LABELS)
    y_enc      = le.transform(y_raw).astype(np.int32)
    return X, y_enc, y_raw, flight_ids, feature_cols, le


def build_lstm(seq_len, n_features, n_classes):
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        Bidirectional(LSTM(config.LSTM_UNITS[0], return_sequences=True)),
        Dropout(config.LSTM_DROPOUT),
        Bidirectional(LSTM(config.LSTM_UNITS[1], return_sequences=False)),
        Dropout(config.LSTM_DROPOUT),
        Dense(64, activation="relu"),
        Dropout(config.LSTM_DROPOUT / 2),
        Dense(n_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_fold(X_train, X_test, y_train, y_test,
               y_raw_test, feature_cols, le, fold_name, n_classes):
    seq_len    = config.LSTM_SEQUENCE_LEN
    n_features = X_train.shape[1]

    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_train)
    X_te_s   = scaler.transform(X_test)

    X_tr_seq, y_tr_seq = build_sequences(X_tr_s, y_train, seq_len)
    X_te_seq, y_te_seq = build_sequences(X_te_s, y_test,  seq_len)

    if len(X_te_seq) == 0:
        print(f"  [{fold_name}] Test fold too short — skipping.")
        return None

    y_tr_cat = to_categorical(y_tr_seq, n_classes)

    model = build_lstm(seq_len, n_features, n_classes)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=config.LSTM_PATIENCE,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=5, verbose=0, min_lr=1e-6),
    ]

    print(f"\n  Training LSTM [{fold_name}] ...", end=" ", flush=True)
    history = model.fit(
        X_tr_seq, y_tr_cat,
        validation_split=0.1,
        epochs=config.LSTM_EPOCHS,
        batch_size=config.LSTM_BATCH_SIZE,
        callbacks=callbacks,
        verbose=0,
    )
    print(f"done ({len(history.history['loss'])} epochs).")

    y_pred_prob = model.predict(X_te_seq, verbose=0)
    y_pred_enc  = np.argmax(y_pred_prob, axis=1)
    y_pred_raw  = le.inverse_transform(y_pred_enc)
    y_raw_align = y_raw_test[seq_len:]   # align labels with sequence offset

    utils.print_classification_report(y_raw_align, y_pred_raw,
                                      f"LSTM [{fold_name}]")
    utils.plot_confusion_matrix(y_raw_align, y_pred_raw,
                                f"LSTM_{fold_name}")
    utils.plot_roc_curves(y_raw_align, y_pred_prob,
                          f"LSTM_{fold_name}")
    _plot_learning_curve(history, fold_name)

    macro_f1 = f1_score(y_raw_align, y_pred_raw,
                        labels=config.PHASE_LABELS,
                        average="macro", zero_division=0)

    # ── Save Keras model ────────────────────────────────────────────
    model_path = os.path.join(config.MODELS_DIR, f"lstm_{fold_name}.keras")
    model.save(model_path)
    print(f"  Saved LSTM → {model_path}")

    # ── FIX: Save companion meta (scaler + le + feature info) ───────
    meta_path = os.path.join(config.MODELS_DIR, f"lstm_{fold_name}_meta.pkl")
    joblib.dump({
        "scaler":       scaler,
        "le":           le,
        "feature_cols": feature_cols,
        "seq_len":      seq_len,
        "macro_f1":     macro_f1,
    }, meta_path)
    print(f"  Saved LSTM meta → {meta_path}")

    return macro_f1


def _plot_learning_curve(history, fold_name):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history.history["loss"],     label="Train")
    ax1.plot(history.history["val_loss"], label="Val")
    ax1.set_title(f"LSTM Loss [{fold_name}]"); ax1.legend()
    ax2.plot(history.history["accuracy"],     label="Train")
    ax2.plot(history.history["val_accuracy"], label="Val")
    ax2.set_title(f"LSTM Accuracy [{fold_name}]"); ax2.legend()
    plt.tight_layout()
    path = os.path.join(config.PLOTS_DIR, f"lstm_learning_{fold_name}.png")
    plt.savefig(path, dpi=150); plt.close()


def main():
    utils.ensure_dirs()
    print("\n" + "="*55)
    print("  STEP 4 — LSTM Training & Evaluation")
    print("="*55)

    X, y_enc, y_raw, flight_ids, feature_cols, le = load_dataset()
    n_classes = len(config.PHASE_LABELS)
    print(f"\n  Dataset  : {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Seq len  : {config.LSTM_SEQUENCE_LEN}")

    f1_scores = []
    per_fold  = []

    if config.SPLIT_MODE == "leave_one_out" and flight_ids is not None:
        for fid in np.unique(flight_ids):
            test_mask  = flight_ids == fid
            train_mask = ~test_mask
            fold_name  = f"rocket_{fid}"
            f1 = train_fold(X[train_mask], X[test_mask],
                            y_enc[train_mask], y_enc[test_mask],
                            y_raw[test_mask], feature_cols, le,
                            fold_name, n_classes)
            if f1 is not None:
                f1_scores.append(f1)
                per_fold.append({"model": "LSTM",
                                 "fold":  f"fold_{fold_name}",
                                 "macro_f1": round(f1, 6)})
    else:
        idx = np.arange(len(X))
        tr_idx, te_idx = train_test_split(
            idx, test_size=config.TEST_SIZE,
            stratify=y_enc, random_state=config.RANDOM_STATE
        )
        f1 = train_fold(X[tr_idx], X[te_idx],
                        y_enc[tr_idx], y_enc[te_idx],
                        y_raw[te_idx], feature_cols, le,
                        "random_split", n_classes)
        if f1 is not None:
            f1_scores.append(f1)
            per_fold.append({"model": "LSTM",
                             "fold":  "random_split",
                             "macro_f1": round(f1, 6)})

    if f1_scores:
        mean_f1 = np.mean(f1_scores)
        std_f1  = np.std(f1_scores)
        print(f"\n{'='*55}")
        print(f"  LSTM Mean Macro F1: {mean_f1:.4f} ± {std_f1:.4f}")

        # Update aggregate summary
        summary_path = os.path.join(config.OUTPUT_DIR, "model_summary.json")
        summary = json.load(open(summary_path)) if os.path.exists(summary_path) else []
        # Remove old LSTM entry if present, re-append
        summary = [s for s in summary if s.get("model") != "LSTM"]
        summary.append({"model": "LSTM",
                         "mean_macro_f1": mean_f1,
                         "std_macro_f1":  std_f1})
        json.dump(summary, open(summary_path, "w"), indent=2)
        print(f"  Updated summary → {summary_path}")

        # Update per-fold summary
        pf_path = os.path.join(config.OUTPUT_DIR, "model_summary_perFold.json")
        pf_data = json.load(open(pf_path)) if os.path.exists(pf_path) else []
        pf_data = [d for d in pf_data if d.get("model") != "LSTM"]
        pf_data.extend(per_fold)
        json.dump(pf_data, open(pf_path, "w"), indent=2)
        print(f"  Updated per-fold → {pf_path}")

    print("✓ LSTM complete. Plots → outputs/plots/")


if __name__ == "__main__":
    main()
