import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import config
import utils
from phasenet_layers import (
    DilatedCausalConvBlock,
    TCNBranch,
    MultiScaleTemporalEncoder,
    CrossScaleAttentionFusion
)
import importlib
phasenet_module = importlib.import_module("09_phasenet_model")
load_dataset = phasenet_module.load_dataset
plot_attention_heatmap = phasenet_module.plot_attention_heatmap
build_sequences = phasenet_module.build_sequences

def generate_all_heatmaps():
    print("═"*60)
    print("  Generating Attention Heatmaps for Saved PhaseNet Models")
    print("═"*60)

    # 1. Load data
    X, y_enc, y_raw, flight_ids, feature_cols, le = load_dataset()
    
    if flight_ids is None or config.SPLIT_MODE != "leave_one_out":
        print("This script is designed for leave_one_out cross-validation.")
        return

    # 2. Iterate through folds
    for fid in np.unique(flight_ids):
        fold_name  = f"rocket_{int(fid)}"
        model_path = os.path.join(config.MODELS_DIR, f"phasenet_{fold_name}.keras")
        
        if not os.path.exists(model_path):
            print(f"Skipping {fold_name} - model not found at {model_path}")
            continue
            
        print(f"\nProcessing {fold_name}...")
        
        test_mask  = flight_ids == fid
        train_mask = ~test_mask
        
        try:
            # Load the model with custom_objects
            custom_objects = {
                "DilatedCausalConvBlock": DilatedCausalConvBlock,
                "TCNBranch": TCNBranch,
                "MultiScaleTemporalEncoder": MultiScaleTemporalEncoder,
                "CrossScaleAttentionFusion": CrossScaleAttentionFusion,
                "physics_loss": None  # or whatever if physics loss was saved, but probably not as layer
            }
            loaded_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, safe_mode=False)
            
            # Standard scaler (fitted on training data)
            scaler = StandardScaler()
            scaler.fit(X[train_mask])
            X_te_s = scaler.transform(X[test_mask])
            
            # Build sequences for this fold
            X_te_seq, _ = build_sequences(X_te_s, y_enc[test_mask], config.PHASENET_SEQUENCE_LEN)
            
            if len(X_te_seq) > 0:
                plot_attention_heatmap(
                    loaded_model, X_te_seq,
                    y_raw[test_mask],
                    feature_cols, le, fold_name,
                )
                print(f"✓ Heatmap generated for {fold_name}")
            else:
                print(f"! No valid sequences found for {fold_name}")
                
        except Exception as e:
            print(f"✗ Failed to generate heatmap for {fold_name}: {e}")

if __name__ == "__main__":
    utils.ensure_dirs()
    generate_all_heatmaps()
