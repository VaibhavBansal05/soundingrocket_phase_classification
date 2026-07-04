"""
phasenet_layers.py
───────────────────
Custom Keras layers for PhaseNet — Physics-Informed Multi-Scale
Temporal Fusion Network.

Components:
    1. DilatedCausalConvBlock  — single TCN residual block
    2. TCNBranch               — stacked dilated causal convolutions at one scale
    3. MultiScaleTemporalEncoder — parallel TCN branches at multiple kernel sizes
    4. CrossScaleAttentionFusion — multi-head attention across temporal scales
    5. PhysicsConstraintLoss     — physics-informed auxiliary loss

Author : PhaseNet (Custom architecture — not available in any library)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
import keras


# ═══════════════════════════════════════════════════════════════════════
# 1.  DILATED CAUSAL CONV BLOCK  (TCN building block)
# ═══════════════════════════════════════════════════════════════════════

@keras.saving.register_keras_serializable()
class DilatedCausalConvBlock(layers.Layer):
    """
    Single residual block:  CausalConv1D → LayerNorm → GELU → Dropout
                          + CausalConv1D → LayerNorm → GELU → Dropout
                          + Residual skip connection

    Causal padding ensures no future information leaks — critical for
    real-time streaming prediction.
    """

    def __init__(self, filters, kernel_size, dilation_rate, dropout=0.2,
                 **kwargs):
        super().__init__(**kwargs)
        self.filters       = int(filters)
        self.kernel_size   = int(kernel_size)
        self.dilation_rate = int(dilation_rate)
        self.dropout_rate  = dropout

    def build(self, input_shape):
        # Two causal conv layers per block (standard TCN design)
        self.conv1 = layers.Conv1D(
            self.filters, self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding="causal",
            kernel_initializer="he_normal",
        )
        self.ln1   = layers.LayerNormalization()
        self.drop1 = layers.Dropout(self.dropout_rate)

        self.conv2 = layers.Conv1D(
            self.filters, self.kernel_size,
            dilation_rate=self.dilation_rate,
            padding="causal",
            kernel_initializer="he_normal",
        )
        self.ln2   = layers.LayerNormalization()
        self.drop2 = layers.Dropout(self.dropout_rate)

        # 1×1 conv for residual dimension matching
        in_channels = input_shape[-1]
        if in_channels != self.filters:
            self.skip_conv = layers.Conv1D(self.filters, 1)
        else:
            self.skip_conv = None

        super().build(input_shape)

    def call(self, x, training=False):
        residual = x

        # First conv + norm + activation + dropout
        out = self.conv1(x)
        out = self.ln1(out)
        out = tf.nn.gelu(out)
        out = self.drop1(out, training=training)

        # Second conv + norm + activation + dropout
        out = self.conv2(out)
        out = self.ln2(out)
        out = tf.nn.gelu(out)
        out = self.drop2(out, training=training)

        # Residual connection
        if self.skip_conv is not None:
            residual = self.skip_conv(residual)

        return out + residual

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "filters":       self.filters,
            "kernel_size":   self.kernel_size,
            "dilation_rate": self.dilation_rate,
            "dropout":       self.dropout_rate,
        })
        return cfg


# ═══════════════════════════════════════════════════════════════════════
# 2.  TCN BRANCH  (stack of dilated causal blocks at one scale)
# ═══════════════════════════════════════════════════════════════════════

@keras.saving.register_keras_serializable()
class TCNBranch(layers.Layer):
    """
    A single-scale temporal branch:  stacks DilatedCausalConvBlocks
    with exponentially increasing dilation rates.

    Parameters
    ----------
    filters      : int    — number of filters per conv layer
    kernel_size  : int    — kernel size defining temporal scale (e.g., 3, 7, 15)
    dilations    : list   — dilation rates per block (e.g., [1, 2, 4, 8])
    dropout      : float  — dropout rate within each block
    """

    def __init__(self, filters, kernel_size, dilations, dropout=0.2,
                 **kwargs):
        super().__init__(**kwargs)
        self.filters     = int(filters)
        self.kernel_size = int(kernel_size)
        self.dilations   = dilations
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.blocks = []
        for d in self.dilations:
            self.blocks.append(
                DilatedCausalConvBlock(
                    self.filters, self.kernel_size, d,
                    dropout=self.dropout_rate,
                    name=f"tcn_k{self.kernel_size}_d{d}"
                )
            )
        super().build(input_shape)

    def call(self, x, training=False):
        out = x
        for block in self.blocks:
            out = block(out, training=training)
        return out

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "filters":     self.filters,
            "kernel_size": self.kernel_size,
            "dilations":   self.dilations,
            "dropout":     self.dropout_rate,
        })
        return cfg


# ═══════════════════════════════════════════════════════════════════════
# 3.  MULTI-SCALE TEMPORAL ENCODER  (parallel TCN branches)
# ═══════════════════════════════════════════════════════════════════════

@keras.saving.register_keras_serializable()
class MultiScaleTemporalEncoder(layers.Layer):
    """
    Runs the input through N parallel TCN branches, each with a
    different kernel size (temporal scale).

    Output: concatenated feature maps from all branches.
            shape = (batch, seq_len, filters × n_scales)

    This is the FIRST novelty component:
        - Fine-grain (k=3):  captures rapid transitions (ignition, deploy)
        - Medium-grain (k=7): captures phase-level dynamics
        - Coarse-grain (k=15): captures flight-level trajectory shape
    """

    def __init__(self, scales_config, filters, dropout=0.2, **kwargs):
        """
        scales_config : dict  — {kernel_size: [dilation_rates]}
                        e.g., {3: [1,2,4,8], 7: [1,2,4], 15: [1,2]}
        """
        super().__init__(**kwargs)
        self.scales_config = {int(k): v for k, v in scales_config.items()}
        self.filters       = filters
        self.dropout_rate  = dropout

    def build(self, input_shape):
        self.branches = {}
        for kernel_size, dilations in self.scales_config.items():
            self.branches[kernel_size] = TCNBranch(
                self.filters, kernel_size, dilations,
                dropout=self.dropout_rate,
                name=f"branch_k{kernel_size}",
            )
        super().build(input_shape)

    def call(self, x, training=False):
        outputs = []
        for kernel_size in sorted(self.branches.keys()):
            branch_out = self.branches[kernel_size](x, training=training)
            outputs.append(branch_out)
        # Concatenate along feature axis → (batch, seq_len, filters * n_scales)
        return tf.concat(outputs, axis=-1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "scales_config": {int(k): v for k, v in self.scales_config.items()},
            "filters":       self.filters,
            "dropout":       self.dropout_rate,
        })
        return cfg


# ═══════════════════════════════════════════════════════════════════════
# 4.  CROSS-SCALE ATTENTION FUSION
# ═══════════════════════════════════════════════════════════════════════

@keras.saving.register_keras_serializable()
class CrossScaleAttentionFusion(layers.Layer):
    """
    Multi-Head Self-Attention across the concatenated multi-scale features.

    This is the SECOND novelty component:
        Instead of simply concatenating the TCN branch outputs, attention
        learns *which temporal scale* is most informative at each time step.

        - During Boost:   fine-grain features dominate (rapid changes)
        - During Coast:   coarse-grain features dominate (smooth trajectory)
        - At transitions: medium-grain provides the strongest signal

    The attention weights can be extracted for interpretability plots
    (publishable figures showing scale importance per flight phase).
    """

    def __init__(self, num_heads=4, key_dim=48, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_heads    = num_heads
        self.key_dim      = key_dim
        self.dropout_rate = dropout

    def build(self, input_shape):
        self.mha = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            dropout=self.dropout_rate,
            name="cross_scale_mha",
        )
        self.ln1     = layers.LayerNormalization()
        self.ff      = layers.Dense(input_shape[-1], activation="gelu",
                                    kernel_initializer="he_normal")
        self.ln2     = layers.LayerNormalization()
        self.ff_drop = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, x, training=False, return_attention=False):
        # Self-attention: each timestep attends to all others
        if return_attention:
            attn_out, attn_weights = self.mha(
                query=x, value=x, key=x,
                training=training,
                return_attention_scores=True,
            )
        else:
            attn_out = self.mha(query=x, value=x, key=x, training=training)
            attn_weights = None

        # Residual + LayerNorm
        x = self.ln1(x + attn_out)

        # Feed-forward + Residual + LayerNorm
        ff_out = self.ff(x)
        ff_out = self.ff_drop(ff_out, training=training)
        x = self.ln2(x + ff_out)

        if return_attention:
            return x, attn_weights
        return x

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "num_heads": self.num_heads,
            "key_dim":   self.key_dim,
            "dropout":   self.dropout_rate,
        })
        return cfg


# ═══════════════════════════════════════════════════════════════════════
# 5.  PHYSICS-INFORMED CONSTRAINT LOSS
# ═══════════════════════════════════════════════════════════════════════

class PhysicsConstraintLoss:
    """
    This is the THIRD novelty component:
        Embeds domain knowledge directly into the training objective.
        The total loss = L_CE + λ₁·L_energy + λ₂·L_transition + λ₃·L_kinematics

    Three physics constraints:

    1. ENERGY MONOTONICITY (L_energy):
       After apogee, total mechanical energy must decrease (drag + gravity).
       Penalizes if model predicts Descent/Landed but the energy proxy in the
       input window is increasing.

    2. LEGAL PHASE TRANSITIONS (L_transition):
       Phases must follow the physical order: Boost→Coast→Apogee→Descent→Landed.
       Penalizes the model's predicted probability on classes that would
       require an illegal transition from the previous timestep.

    3. KINEMATIC CONSISTENCY (L_kinematics):
       Velocity sign must be consistent with phase:
         - Boost/Coast → velocity should be positive (ascending)
         - Descent     → velocity should be negative (descending)
       Penalizes contradiction between predicted phase and velocity sign
       embedded in the input features.

    Usage:
        physics_loss = PhysicsConstraintLoss(...)
        # Inside training step:
        total_loss = ce_loss + physics_loss(y_pred_probs, X_batch)
    """

    def __init__(self, lambda_energy=0.1, lambda_transition=0.2,
                 lambda_kinematics=0.1, legal_transitions=None,
                 velocity_feature_idx=None, energy_feature_idx=None):
        self.lambda_energy     = lambda_energy
        self.lambda_transition = lambda_transition
        self.lambda_kinematics = lambda_kinematics
        self.legal_transitions = legal_transitions or {}
        self.vel_idx           = velocity_feature_idx
        self.energy_idx        = energy_feature_idx

        # Pre-compute illegal transition mask  (n_classes × n_classes)
        # illegal_mask[i, j] = 1.0 if transition from class i to class j is illegal
        n_classes = 5
        self._illegal_mask = np.ones((n_classes, n_classes), dtype=np.float32)
        for src, dsts in self.legal_transitions.items():
            for dst in dsts:
                self._illegal_mask[src, dst] = 0.0
        self._illegal_mask_tf = tf.constant(self._illegal_mask)

    def energy_loss(self, y_pred, X_seq):
        """
        Penalizes Descent/Landed predictions when energy is increasing.
        Uses the last timestep of the input sequence for energy proxy.
        """
        if self.energy_idx is None:
            return 0.0

        # Energy at last timestep in the sequence
        energy_last = X_seq[:, -1, self.energy_idx]      # (batch,)
        energy_prev = X_seq[:, -2, self.energy_idx]      # (batch,)
        energy_increasing = tf.nn.relu(energy_last - energy_prev)  # > 0 if increasing

        # Probability assigned to Descent (idx=3) + Landed (idx=4)
        p_descent_landed = y_pred[:, 3] + y_pred[:, 4]

        # Loss: penalize when energy is rising AND model says Descent/Landed
        loss = tf.reduce_mean(p_descent_landed * energy_increasing)
        return loss

    def transition_loss(self, y_pred, y_pred_prev):
        """
        Penalizes predicted probability mass on classes that would
        require an illegal transition from the previous prediction.

        y_pred      : (batch, n_classes) — current timestep probabilities
        y_pred_prev : (batch, n_classes) — previous timestep probabilities
        """
        if y_pred_prev is None:
            return 0.0

        # Guard: if batch sizes differ (e.g. last mini-batch is smaller),
        # trim y_pred_prev to the current batch size so shapes are compatible.
        current_bs = tf.shape(y_pred)[0]
        y_pred_prev = y_pred_prev[:current_bs]

        # Previous most-likely class  → one-hot → which next classes are illegal
        prev_class = tf.argmax(y_pred_prev, axis=-1)  # (batch,)

        # Gather the illegal mask row for each sample's previous class
        # illegal_for_sample[b, j] = 1 if transition prev_class[b] → j is illegal
        illegal_for_sample = tf.gather(self._illegal_mask_tf, prev_class)  # (batch, n_classes)

        # Penalize: probability assigned to illegal next-classes
        illegal_prob = tf.reduce_sum(y_pred * illegal_for_sample, axis=-1)  # (batch,)
        return tf.reduce_mean(illegal_prob)

    def kinematics_loss(self, y_pred, X_seq):
        """
        Penalizes when velocity sign contradicts predicted phase.
        velocity > 0 but predicting Descent, or velocity < 0 but predicting Boost/Coast.
        """
        if self.vel_idx is None:
            return 0.0

        vel_last = X_seq[:, -1, self.vel_idx]  # (batch,)

        # Ascending signal: velocity > 0
        is_ascending = tf.cast(vel_last > 0.0, tf.float32)     # 1 if going up
        is_descending = tf.cast(vel_last < 0.0, tf.float32)    # 1 if going down

        # Probability on ascending phases (Boost=0, Coast=1)
        p_ascending = y_pred[:, 0] + y_pred[:, 1]
        # Probability on descending phases (Descent=3)
        p_descending = y_pred[:, 3]

        # Penalize: descending velocity but predicting ascending phase
        loss_asc = tf.reduce_mean(is_descending * p_ascending)
        # Penalize: ascending velocity but predicting Descent
        loss_desc = tf.reduce_mean(is_ascending * p_descending)

        return loss_asc + loss_desc

    def __call__(self, y_pred, X_seq, y_pred_prev=None):
        """
        Compute total physics constraint loss.

        Parameters
        ----------
        y_pred      : (batch, n_classes) predicted probabilities
        X_seq       : (batch, seq_len, n_features) input sequences
        y_pred_prev : (batch, n_classes) previous step predictions (optional)

        Returns
        -------
        Scalar loss (to be added to cross-entropy loss)
        """
        L_e = self.lambda_energy     * self.energy_loss(y_pred, X_seq)
        L_t = self.lambda_transition * self.transition_loss(y_pred, y_pred_prev)
        L_k = self.lambda_kinematics * self.kinematics_loss(y_pred, X_seq)
        return L_e + L_t + L_k


# ═══════════════════════════════════════════════════════════════════════
# 6.  FULL PHASENET MODEL
# ═══════════════════════════════════════════════════════════════════════

def build_phasenet(seq_len, n_features, n_classes,
                   scales_config, filters,
                   attention_heads, attention_key_dim,
                   dense_units, dropout):
    """
    Builds the complete PhaseNet model as a Keras functional model.

    Architecture:
        Input → Multi-Scale TCN Encoder → Cross-Scale Attention Fusion
              → Global Avg Pooling → Dense Head → Softmax

    Returns
    -------
    tf.keras.Model
    """
    inp = layers.Input(shape=(seq_len, n_features), name="input_sequence")

    # ── Multi-Scale Temporal Encoder ─────────────────────────────────
    ms_encoder = MultiScaleTemporalEncoder(
        scales_config=scales_config,
        filters=filters,
        dropout=dropout,
        name="multi_scale_encoder",
    )
    encoded = ms_encoder(inp)  # (batch, seq_len, filters * n_scales)

    # ── Cross-Scale Attention Fusion ─────────────────────────────────
    attention = CrossScaleAttentionFusion(
        num_heads=attention_heads,
        key_dim=attention_key_dim,
        dropout=dropout * 0.5,
        name="cross_scale_attention",
    )
    fused = attention(encoded)  # (batch, seq_len, filters * n_scales)

    # ── Temporal Aggregation ─────────────────────────────────────────
    pooled = layers.GlobalAveragePooling1D(name="temporal_pool")(fused)

    # ── Classification Head ──────────────────────────────────────────
    x = pooled
    for i, units in enumerate(dense_units):
        x = layers.Dense(units, activation="gelu",
                         kernel_initializer="he_normal",
                         name=f"dense_{i}")(x)
        x = layers.Dropout(dropout if i == 0 else dropout * 0.66,
                           name=f"head_dropout_{i}")(x)

    output = layers.Dense(n_classes, activation="softmax",
                          name="phase_output")(x)

    model = Model(inputs=inp, outputs=output, name="PhaseNet")
    return model
