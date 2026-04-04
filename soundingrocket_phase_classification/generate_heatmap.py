import os
import tensorflow as tf
import numpy as np
import joblib
import config
from sklearn.preprocessing import StandardScaler
from phasenet_layers import MultiScaleTemporalEncoder, CrossScaleAttentionFusion, PhysicsConstraintLoss

# 1. Load the data again
print("Loading data...")
import pandas as pd
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(config.FEATURES_PATH)
df.columns = df.columns.str.strip()
feature_cols = [c for c in config.FEATURE_COLS if c in df.columns]
X = df[feature_cols].values.astype(np.float32)
y_raw = df[config.LABEL_COL].values
flight_ids = df["flight_id"].values
le = LabelEncoder()
y_enc = le.fit_transform(y_raw).astype(np.int32)

# Pick fold 1 (the best fold usually)
fid = 1 
test_mask = flight_ids == fid
train_mask = ~test_mask
fold_name = f"rocket_{int(fid)}"

scaler = StandardScaler()
scaler.fit(X[train_mask])
X_te_s = scaler.transform(X[test_mask])

# Build sequences for test
seq_len = config.PHASENET_SEQUENCE_LEN
Xs = []
for i in range(seq_len, len(X_te_s)):
    Xs.append(X_te_s[i - seq_len:i])
X_te_seq = np.array(Xs, dtype=np.float32)

# 2. Load the model using custom_objects
print(f"Loading model for {fold_name}...")
model_path = os.path.join(config.MODELS_DIR, f"phasenet_{fold_name}.keras")

custom_objects = {
    'MultiScaleTemporalEncoder': MultiScaleTemporalEncoder,
    'CrossScaleAttentionFusion': CrossScaleAttentionFusion,
    'PhysicsConstraintLoss': PhysicsConstraintLoss
}

loaded_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

# 3. Use the function from 09_phasenet_model to plot
from importlib import import_module
phasenet_script = import_module("09_phasenet_model")

print("Generating heatmap...")
phasenet_script.plot_attention_heatmap(
    loaded_model, 
    X_te_seq, 
    y_raw[test_mask], 
    feature_cols, 
    le, 
    fold_name
)
print("Done! You now have Figure 13.")
