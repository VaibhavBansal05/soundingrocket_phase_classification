"""
09_phasenet_model.py
─────────────────────
PhaseNet — Physics-Informed Multi-Scale Temporal Fusion Network.

A CUSTOM architecture for sounding rocket flight phase classification.
Not available in any library — designed specifically for this domain.

Three novelty components:
    1. Multi-Scale Temporal Convolutional Encoder  (parallel TCN branches)
    2. Cross-Scale Attention Fusion                (learned scale weighting)
    3. Physics-Informed Constraint Loss            (energy + transition + kinematics)

Evaluation: Leave-One-Rocket-Out Cross-Validation (same as steps 03 & 04).

Usage:
    python 09_phasenet_model.py                 # train + evaluate
    python 09_phasenet_model.py --ablation      # run ablation study variants

Produces:
    - Per-fold confusion matrices and ROC curves
    - Attention heatmaps (scale importance across flight phases)
    - Learning curves (loss + accuracy)
    - Ablation comparison chart (if --ablation)
    - Updated model_summary.json and model_summary_perFold.json

Author : PhaseNet (Custom architecture)
"""

import _compat  # UTF-8 console fix for Windows
import os, sys, json, argparse, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, LearningRateScheduler,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import joblib

import config, utils
from phasenet_layers import (
    build_phasenet,
    PhysicsConstraintLoss,
    CrossScaleAttentionFusion,
)


# ═══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════

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


def build_sequences(X, y, seq_len):
    """Sliding window sequences for temporal models."""
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i - seq_len:i])
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.int32)


# ═══════════════════════════════════════════════════════════════════════
# FEATURE INDEX HELPERS (for physics loss)
# ═══════════════════════════════════════════════════════════════════════

def get_feature_indices(feature_cols):
    """Find the column indices of velocity and energy features for physics loss."""
    vel_idx = None
    energy_idx = None
    for i, c in enumerate(feature_cols):
        if c == config.COL_VELOCITY:
            vel_idx = i
        elif c == "energy_proxy":
            energy_idx = i
    return vel_idx, energy_idx


# ═══════════════════════════════════════════════════════════════════════
# CUSTOM TRAINING STEP (to integrate physics loss)
# ═══════════════════════════════════════════════════════════════════════

class PhaseNetTrainer:
    """
    Handles the custom training loop that combines:
        - Standard categorical cross-entropy loss
        - Physics-informed constraint loss (energy, transition, kinematics)

    We use a custom training loop instead of model.fit() because the
    physics loss requires access to the raw input sequences AND the
    previous timestep's predictions — which standard Keras losses don't support.
    """

    def __init__(self, model, physics_loss, optimizer, n_classes):
        self.model        = model
        self.physics_loss = physics_loss
        self.optimizer    = optimizer
        self.n_classes    = n_classes
        self.ce_loss_fn   = tf.keras.losses.CategoricalCrossentropy()

    def train_step(self, X_batch, y_batch):
        """Single gradient update step with combined loss (eager mode)."""
        with tf.GradientTape() as tape:
            y_pred = self.model(X_batch, training=True)
            ce_loss = self.ce_loss_fn(y_batch, y_pred)

            # Physics constraint loss
            phys_loss = self.physics_loss(y_pred, X_batch, y_pred_prev=None)

            total_loss = ce_loss + phys_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Compute accuracy from current predictions (avoid second forward pass)
        correct = int(tf.reduce_sum(
            tf.cast(tf.argmax(y_pred, -1) == tf.argmax(y_batch, -1), tf.int32)
        ))
        return float(total_loss), float(ce_loss), float(phys_loss), correct

    def eval_step(self, X_batch, y_batch):
        y_pred = self.model(X_batch, training=False)
        ce_loss = self.ce_loss_fn(y_batch, y_pred)
        correct = int(tf.reduce_sum(
            tf.cast(tf.argmax(y_pred, -1) == tf.argmax(y_batch, -1), tf.int32)
        ))
        return float(ce_loss), correct


    def fit(self, X_train, y_train, X_val, y_val,
            epochs, batch_size, patience, lr_min=1e-6):
        """
        Full training loop with early stopping, LR reduction, and
        history tracking.
        """
        n_train   = len(X_train)
        y_train_cat = to_categorical(y_train, self.n_classes)
        y_val_cat   = to_categorical(y_val, self.n_classes)

        # Convert to tf.data for efficient batching
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_cat))
        train_ds = train_ds.shuffle(n_train).batch(batch_size).prefetch(2)

        val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val_cat))
        val_ds = val_ds.batch(batch_size).prefetch(2)

        history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
        best_val_loss = float("inf")
        best_weights  = None
        wait          = 0
        lr_patience   = max(5, patience // 2)
        lr_wait       = 0

        for epoch in range(epochs):
            # ── Train epoch ──────────────────────────────────────────
            epoch_losses = []
            epoch_correct = 0
            epoch_total   = 0

            for X_b, y_b in train_ds:
                loss, ce, phys, correct = self.train_step(X_b, y_b)
                epoch_losses.append(loss)
                epoch_correct += correct
                epoch_total += len(X_b)

            train_loss = np.mean(epoch_losses)
            train_acc  = epoch_correct / max(epoch_total, 1)
            history["loss"].append(train_loss)
            history["accuracy"].append(train_acc)

            # ── Validation epoch ─────────────────────────────────────
            val_losses = []
            val_correct = 0
            val_total   = 0

            for X_b, y_b in val_ds:
                v_loss, v_correct = self.eval_step(X_b, y_b)
                val_losses.append(v_loss)
                val_correct += v_correct
                val_total += len(X_b)

            val_loss = np.mean(val_losses) if val_losses else float("inf")
            val_acc  = val_correct / max(val_total, 1)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)

            # ── Progress log (every 5 epochs or last) ────────────────
            if epoch % 5 == 0 or epoch == epochs - 1 or wait >= patience - 1:
                marker = "*" if val_loss <= best_val_loss else " "
                print(f"\r    Epoch {epoch+1:3d}/{epochs}  "
                      f"loss={train_loss:.4f}  acc={train_acc:.3f}  "
                      f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f} {marker}",
                      flush=True)

            # ── Early stopping ───────────────────────────────────────
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights  = self.model.get_weights()
                wait = 0
                lr_wait = 0
            else:
                wait += 1
                lr_wait += 1

            # ── Learning rate reduction ──────────────────────────────
            if lr_wait >= lr_patience:
                current_lr = float(self.optimizer.learning_rate)
                new_lr = max(current_lr * 0.5, lr_min)
                self.optimizer.learning_rate.assign(new_lr)
                lr_wait = 0

            if wait >= patience:
                break

        # Restore best weights
        if best_weights is not None:
            self.model.set_weights(best_weights)

        n_epochs = len(history["loss"])
        return history, n_epochs


# ═══════════════════════════════════════════════════════════════════════
# SINGLE FOLD TRAIN + EVALUATE
# ═══════════════════════════════════════════════════════════════════════

def train_fold(X_train, X_test, y_train, y_test,
               y_raw_test, feature_cols, le, fold_name, n_classes,
               use_physics_loss=True, use_multiscale=True,
               use_attention=True, variant_name="PhaseNet"):
    """
    Trains and evaluates PhaseNet on a single fold.

    The use_physics_loss, use_multiscale, and use_attention flags
    control ablation:
        Full PhaseNet:   all True
        No-physics:      use_physics_loss=False
        Single-scale:    use_multiscale=False (uses only k=7)
        No-attention:    use_attention=False (uses concatenation only)
    """
    seq_len    = config.PHASENET_SEQUENCE_LEN
    n_features = X_train.shape[1]

    # Scale features
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    # Build sequences
    X_tr_seq, y_tr_seq = build_sequences(X_tr_s, y_train, seq_len)
    X_te_seq, y_te_seq = build_sequences(X_te_s, y_test, seq_len)

    if len(X_te_seq) == 0:
        print(f"  [{fold_name}] Test fold too short — skipping.")
        return None

    # Validation split from training data
    val_split  = 0.1
    n_val      = max(1, int(len(X_tr_seq) * val_split))
    X_val_seq  = X_tr_seq[-n_val:]
    y_val_seq  = y_tr_seq[-n_val:]
    X_tr_seq   = X_tr_seq[:-n_val]
    y_tr_seq   = y_tr_seq[:-n_val]

    # ── Configure scales (ablation control) ──────────────────────────
    if use_multiscale:
        scales_config = config.PHASENET_TCN_DILATIONS
    else:
        # Single-scale ablation: only medium kernel
        scales_config = {7: config.PHASENET_TCN_DILATIONS[7]}

    # ── Build model ──────────────────────────────────────────────────
    model = build_phasenet(
        seq_len         = seq_len,
        n_features      = n_features,
        n_classes       = n_classes,
        scales_config   = scales_config,
        filters         = config.PHASENET_TCN_FILTERS,
        attention_heads = config.PHASENET_ATTENTION_HEADS if use_attention else 1,
        attention_key_dim = config.PHASENET_ATTENTION_KEY_DIM if use_attention else 16,
        dense_units     = config.PHASENET_DENSE_UNITS,
        dropout         = config.PHASENET_DROPOUT,
    )

    # ── Physics constraint loss ──────────────────────────────────────
    vel_idx, energy_idx = get_feature_indices(feature_cols)

    if use_physics_loss:
        physics_loss = PhysicsConstraintLoss(
            lambda_energy     = config.PHASENET_LAMBDA_ENERGY,
            lambda_transition = config.PHASENET_LAMBDA_TRANSITION,
            lambda_kinematics = config.PHASENET_LAMBDA_KINEMATICS,
            legal_transitions = config.PHASENET_LEGAL_TRANSITIONS,
            velocity_feature_idx = vel_idx,
            energy_feature_idx   = energy_idx,
        )
    else:
        # Ablation: no physics loss → use zero loss
        physics_loss = PhysicsConstraintLoss(
            lambda_energy=0.0, lambda_transition=0.0, lambda_kinematics=0.0,
            legal_transitions=config.PHASENET_LEGAL_TRANSITIONS,
            velocity_feature_idx=vel_idx, energy_feature_idx=energy_idx,
        )

    optimizer = tf.keras.optimizers.Adam(config.PHASENET_LR_INIT)

    trainer = PhaseNetTrainer(model, physics_loss, optimizer, n_classes)

    # ── Train ────────────────────────────────────────────────────────
    print(f"\n  Training {variant_name} [{fold_name}] ...", end=" ", flush=True)
    history, n_epochs = trainer.fit(
        X_tr_seq, y_tr_seq,
        X_val_seq, y_val_seq,
        epochs     = config.PHASENET_EPOCHS,
        batch_size = config.PHASENET_BATCH_SIZE,
        patience   = config.PHASENET_PATIENCE,
        lr_min     = config.PHASENET_LR_MIN,
    )
    print(f"done ({n_epochs} epochs).")

    # ── Evaluate ─────────────────────────────────────────────────────
    y_pred_prob = model.predict(X_te_seq, verbose=0)
    y_pred_enc  = np.argmax(y_pred_prob, axis=1)
    y_pred_raw  = le.inverse_transform(y_pred_enc)
    y_raw_align = y_raw_test[seq_len:]

    utils.print_classification_report(y_raw_align, y_pred_raw,
                                      f"{variant_name} [{fold_name}]")
    utils.plot_confusion_matrix(y_raw_align, y_pred_raw,
                                f"{variant_name}_{fold_name}")
    utils.plot_roc_curves(y_raw_align, y_pred_prob,
                          f"{variant_name}_{fold_name}")

    # Learning curve
    _plot_learning_curve(history, fold_name, variant_name)

    macro_f1 = f1_score(y_raw_align, y_pred_raw,
                        labels=config.PHASE_LABELS,
                        average="macro", zero_division=0)

    # ── Save model + meta ────────────────────────────────────────────
    safe_variant = variant_name.lower().replace(" ", "_").replace("-", "_")
    model_path = os.path.join(config.MODELS_DIR,
                              f"{safe_variant}_{fold_name}.keras")
    model.save(model_path)
    print(f"  Saved {variant_name} → {model_path}")

    meta_path = os.path.join(config.MODELS_DIR,
                             f"{safe_variant}_{fold_name}_meta.pkl")
    joblib.dump({
        "scaler":       scaler,
        "le":           le,
        "feature_cols": feature_cols,
        "seq_len":      seq_len,
        "macro_f1":     macro_f1,
        "variant":      variant_name,
        "n_scales":     len(scales_config),
        "physics_loss": use_physics_loss,
    }, meta_path)
    print(f"  Saved meta → {meta_path}")

    return macro_f1


# ═══════════════════════════════════════════════════════════════════════
# ATTENTION HEATMAP VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════

def plot_attention_heatmap(model, X_sample, y_sample_raw, feature_cols,
                           le, fold_name):
    """
    Extracts and visualizes the cross-scale attention weights.
    Shows which temporal scale the model relied on most at each timestep.

    This is a key publishable figure for the research paper.
    """
    # Get the attention layer
    attention_layer = None
    for layer in model.layers:
        if isinstance(layer, CrossScaleAttentionFusion):
            attention_layer = layer
            break

    if attention_layer is None:
        print("  Warning: no CrossScaleAttentionFusion layer found for heatmap.")
        return

    # Build a sub-model that outputs attention weights
    encoder_output = None
    for layer in model.layers:
        if "multi_scale_encoder" in layer.name:
            encoder_output = layer.output
            break

    if encoder_output is None:
        return

    # Forward pass through encoder → then attention with return_attention=True
    # Simpler approach: direct call
    try:
        # Get encoder output
        encoder_model = tf.keras.Model(
            inputs=model.input,
            outputs=encoder_output,
            name="encoder_submodel"
        )
        encoded = encoder_model.predict(X_sample[:50], verbose=0)

        # Get attention weights
        _, attn_weights = attention_layer(
            tf.constant(encoded, dtype=tf.float32),
            training=False,
            return_attention=True,
        )

        if attn_weights is None:
            return

        # Average attention across heads → (batch, seq_len, seq_len)
        attn_avg = tf.reduce_mean(attn_weights, axis=1).numpy()
        # Average across batch → (seq_len, seq_len)
        attn_avg = np.mean(attn_avg, axis=0)

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(attn_avg, cmap="magma", aspect="auto")
        ax.set_xlabel("Key Position (timestep)", fontsize=11)
        ax.set_ylabel("Query Position (timestep)", fontsize=11)
        ax.set_title(
            f"PhaseNet — Cross-Scale Attention Heatmap [{fold_name}]\n"
            f"(Averaged across {min(50, len(X_sample))} samples, "
            f"{attn_weights.shape[1]} heads)",
            fontsize=12, fontweight="bold"
        )
        plt.colorbar(im, ax=ax, label="Attention Weight")
        plt.tight_layout()

        path = os.path.join(config.PLOTS_DIR,
                            f"phasenet_attention_{fold_name}.png")
        plt.savefig(path, dpi=180)
        plt.close()
        print(f"  Saved attention heatmap → {path}")
    except Exception as e:
        print(f"  Warning: Could not generate attention heatmap: {e}")


# ═══════════════════════════════════════════════════════════════════════
# LEARNING CURVE PLOT
# ═══════════════════════════════════════════════════════════════════════

def _plot_learning_curve(history, fold_name, variant_name="PhaseNet"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["loss"],     label="Train", linewidth=2)
    ax1.plot(history["val_loss"], label="Val",   linewidth=2)
    ax1.set_title(f"{variant_name} Loss [{fold_name}]",
                  fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["accuracy"],     label="Train", linewidth=2)
    ax2.plot(history["val_accuracy"], label="Val",   linewidth=2)
    ax2.set_title(f"{variant_name} Accuracy [{fold_name}]",
                  fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    safe_name = variant_name.lower().replace(" ", "_").replace("-", "_")
    path = os.path.join(config.PLOTS_DIR,
                        f"{safe_name}_learning_{fold_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# ABLATION STUDY
# ═══════════════════════════════════════════════════════════════════════

def run_ablation(X, y_enc, y_raw, flight_ids, feature_cols, le, n_classes):
    """
    Runs four model variants for the ablation study:
        1. PhaseNet (full)
        2. PhaseNet-NoPhysics   (physics loss disabled)
        3. PhaseNet-SingleScale (only k=7 branch)
        4. PhaseNet-NoAttention (1-head, minimal attention)

    Produces a comparison chart.
    """
    variants = [
        ("PhaseNet",             True,  True,  True),
        ("PhaseNet-NoPhysics",   False, True,  True),
        ("PhaseNet-SingleScale", True,  False, True),
        ("PhaseNet-NoAttention", True,  True,  False),
    ]

    results = {}

    for variant_name, use_phys, use_multi, use_attn in variants:
        print(f"\n{'═'*60}")
        print(f"  ABLATION — {variant_name}")
        print(f"{'═'*60}")

        f1_scores = []

        if config.SPLIT_MODE == "leave_one_out" and flight_ids is not None:
            for fid in np.unique(flight_ids):
                test_mask  = flight_ids == fid
                train_mask = ~test_mask
                fold_name  = f"rocket_{int(fid)}"
                f1 = train_fold(
                    X[train_mask], X[test_mask],
                    y_enc[train_mask], y_enc[test_mask],
                    y_raw[test_mask], feature_cols, le,
                    fold_name, n_classes,
                    use_physics_loss=use_phys,
                    use_multiscale=use_multi,
                    use_attention=use_attn,
                    variant_name=variant_name,
                )
                if f1 is not None:
                    f1_scores.append(f1)
        else:
            idx = np.arange(len(X))
            tr_idx, te_idx = train_test_split(
                idx, test_size=config.TEST_SIZE,
                stratify=y_enc, random_state=config.RANDOM_STATE,
            )
            f1 = train_fold(
                X[tr_idx], X[te_idx],
                y_enc[tr_idx], y_enc[te_idx],
                y_raw[te_idx], feature_cols, le,
                "random_split", n_classes,
                use_physics_loss=use_phys,
                use_multiscale=use_multi,
                use_attention=use_attn,
                variant_name=variant_name,
            )
            if f1 is not None:
                f1_scores.append(f1)

        if f1_scores:
            results[variant_name] = {
                "mean_f1": np.mean(f1_scores),
                "std_f1":  np.std(f1_scores),
            }
            print(f"\n  {variant_name} — Mean Macro F1: "
                  f"{results[variant_name]['mean_f1']:.4f} "
                  f"± {results[variant_name]['std_f1']:.4f}")

    # ── Plot ablation comparison ─────────────────────────────────────
    if results:
        _plot_ablation(results)

    return results


def _plot_ablation(results):
    """Bar chart comparing ablation variants."""
    names = list(results.keys())
    means = [results[n]["mean_f1"] for n in names]
    stds  = [results[n]["std_f1"]  for n in names]

    colors = ["#e94560", "#0f3460", "#16213e", "#533483"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(names))
    bars = ax.bar(x, means, yerr=stds, capsize=6,
                  color=colors[:len(names)], alpha=0.88, width=0.55,
                  error_kw={"elinewidth": 2, "ecolor": "black"})

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, rotation=15, ha="right")
    ax.set_ylabel("Macro F1 Score", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "PhaseNet Ablation Study\n"
        "Contribution of Each Novelty Component",
        fontsize=13, fontweight="bold", pad=12,
    )

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.015,
                f"{mean:.3f}", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    path = os.path.join(config.PLOTS_DIR, "phasenet_ablation.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\n✓ Saved ablation chart → {path}")

    # Save ablation results JSON
    json_path = os.path.join(config.OUTPUT_DIR, "phasenet_ablation.json")
    with open(json_path, "w") as f:
        json.dump({k: {kk: round(vv, 6) for kk, vv in v.items()}
                   for k, v in results.items()}, f, indent=2)
    print(f"✓ Saved ablation results → {json_path}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PhaseNet — Physics-Informed Multi-Scale Temporal Fusion Network"
    )
    parser.add_argument("--ablation", action="store_true",
                        help="Run ablation study (trains 4 variants × N folds)")
    args = parser.parse_args()

    utils.ensure_dirs()
    print("\n" + "═"*60)
    print("  STEP 9 — PhaseNet Training & Evaluation")
    print("  Physics-Informed Multi-Scale Temporal Fusion Network")
    print("═"*60)

    X, y_enc, y_raw, flight_ids, feature_cols, le = load_dataset()
    n_classes = len(config.PHASE_LABELS)
    print(f"\n  Dataset   : {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Seq len   : {config.PHASENET_SEQUENCE_LEN}")
    print(f"  TCN scales: {config.PHASENET_TCN_SCALES}")
    print(f"  Attention : {config.PHASENET_ATTENTION_HEADS} heads, "
          f"key_dim={config.PHASENET_ATTENTION_KEY_DIM}")
    print(f"  Physics λ : energy={config.PHASENET_LAMBDA_ENERGY}, "
          f"transition={config.PHASENET_LAMBDA_TRANSITION}, "
          f"kinematics={config.PHASENET_LAMBDA_KINEMATICS}")

    # ── Ablation mode ────────────────────────────────────────────────
    if args.ablation:
        run_ablation(X, y_enc, y_raw, flight_ids, feature_cols, le, n_classes)
        print("\n✓ Ablation study complete.")
        return

    # ── Standard training ────────────────────────────────────────────
    f1_scores = []
    per_fold  = []

    if config.SPLIT_MODE == "leave_one_out" and flight_ids is not None:
        for fid in np.unique(flight_ids):
            test_mask  = flight_ids == fid
            train_mask = ~test_mask
            fold_name  = f"rocket_{int(fid)}"

            f1 = train_fold(
                X[train_mask], X[test_mask],
                y_enc[train_mask], y_enc[test_mask],
                y_raw[test_mask], feature_cols, le,
                fold_name, n_classes,
            )
            if f1 is not None:
                f1_scores.append(f1)
                per_fold.append({
                    "model":    "PhaseNet",
                    "fold":     f"fold_{fold_name}",
                    "macro_f1": round(f1, 6),
                })

            # Generate attention heatmap for last fold
            try:
                safe_variant = "phasenet"
                model_path = os.path.join(config.MODELS_DIR,
                                          f"{safe_variant}_{fold_name}.keras")
                if os.path.exists(model_path):
                    loaded_model = tf.keras.models.load_model(model_path)
                    scaler = StandardScaler()
                    scaler.fit(X[train_mask])
                    X_te_s = scaler.transform(X[test_mask])
                    X_te_seq, _ = build_sequences(X_te_s, y_enc[test_mask],
                                                   config.PHASENET_SEQUENCE_LEN)
                    if len(X_te_seq) > 0:
                        plot_attention_heatmap(
                            loaded_model, X_te_seq,
                            y_raw[test_mask],
                            feature_cols, le, fold_name,
                        )
            except Exception as e:
                print(f"  Warning: attention heatmap skipped: {e}")

    else:
        idx = np.arange(len(X))
        tr_idx, te_idx = train_test_split(
            idx, test_size=config.TEST_SIZE,
            stratify=y_enc, random_state=config.RANDOM_STATE,
        )
        f1 = train_fold(
            X[tr_idx], X[te_idx],
            y_enc[tr_idx], y_enc[te_idx],
            y_raw[te_idx], feature_cols, le,
            "random_split", n_classes,
        )
        if f1 is not None:
            f1_scores.append(f1)
            per_fold.append({
                "model":    "PhaseNet",
                "fold":     "random_split",
                "macro_f1": round(f1, 6),
            })

    # ── Summary ──────────────────────────────────────────────────────
    if f1_scores:
        mean_f1 = np.mean(f1_scores)
        std_f1  = np.std(f1_scores)
        print(f"\n{'═'*60}")
        print(f"  PhaseNet Mean Macro F1: {mean_f1:.4f} ± {std_f1:.4f}")
        print(f"{'═'*60}")

        # Print model architecture summary
        dummy_model = build_phasenet(
            config.PHASENET_SEQUENCE_LEN, X.shape[1], n_classes,
            config.PHASENET_TCN_DILATIONS, config.PHASENET_TCN_FILTERS,
            config.PHASENET_ATTENTION_HEADS, config.PHASENET_ATTENTION_KEY_DIM,
            config.PHASENET_DENSE_UNITS, config.PHASENET_DROPOUT,
        )
        total_params = dummy_model.count_params()
        print(f"  Total parameters: {total_params:,}")

        # Update aggregate summary
        summary_path = os.path.join(config.OUTPUT_DIR, "model_summary.json")
        summary = json.load(open(summary_path)) if os.path.exists(summary_path) else []
        summary = [s for s in summary if s.get("model") != "PhaseNet"]
        summary.append({
            "model":         "PhaseNet",
            "mean_macro_f1": mean_f1,
            "std_macro_f1":  std_f1,
        })
        json.dump(summary, open(summary_path, "w"), indent=2)
        print(f"  Updated summary  → {summary_path}")

        # Update per-fold summary
        pf_path = os.path.join(config.OUTPUT_DIR, "model_summary_perFold.json")
        pf_data = json.load(open(pf_path)) if os.path.exists(pf_path) else []
        pf_data = [d for d in pf_data if d.get("model") != "PhaseNet"]
        pf_data.extend(per_fold)
        json.dump(pf_data, open(pf_path, "w"), indent=2)
        print(f"  Updated per-fold → {pf_path}")

    print("\n✓ PhaseNet training complete.")
    print("  Plots     → outputs/plots/")
    print("  Models    → outputs/models/")


if __name__ == "__main__":
    main()
