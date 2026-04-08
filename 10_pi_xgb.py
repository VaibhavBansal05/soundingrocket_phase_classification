"""
10_pi_xgb.py
─────────────
Physics-Informed XGBoost (PI-XGB)

This model uses a custom mathematical objective function for XGBoost.
Standard Gradient Boosting minimizes Categorical Cross-Entropy. PI-XGB minimizes:
  Total Loss = Cross-Entropy + λ·Physics_Penalty

The Physics Penalty matrix C penalizes the model for predicting states that
violate the known kinematics of the rocket based on the tabular features.
"""

import _compat
import os, json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

import config
import utils


def build_physics_objective(X_train_raw, y_train_enc, feature_cols):
    """
    Closure that creates a custom Multi-class objective function for XGBoost.
    It computes exact gradients (1st derivative) and approximate Hessians (2nd derivative)
    for a Softmax objective augmented with tabular Physics Constraints.
    """
    n_classes = len(config.PHASE_LABELS)
    vel_idx = feature_cols.index(config.COL_VELOCITY)
    alt_idx = feature_cols.index("altitude_from_ground")
    
    # Pre-compute penalty matrix C of shape (N, n_classes)
    # C[i, k] > 0 means class k is physically penalized for row i
    N = len(X_train_raw)
    C = np.zeros((N, n_classes), dtype=np.float32)
    
    vel_arr = X_train_raw[:, vel_idx]
    alt_arr = X_train_raw[:, alt_idx]
    
    lamb_kin = config.PI_XGB_LAMBDA_KINEMATICS
    lamb_alt = 5.0  # Height penalty
    
    for i in range(N):
        v = vel_arr[i]
        a = alt_arr[i]
        
        # Kinematics constraints
        if v < -5.0:
            C[i, 0] += lamb_kin  # Boost impossible if falling fast
            C[i, 1] += lamb_kin  # Coast impossible if falling fast
        if v > +5.0:
            C[i, 3] += lamb_kin  # Descent impossible if going up fast
            C[i, 4] += lamb_kin  # Landed impossible if going up fast
            
        # Altitude constraints
        if a > 500.0:
            C[i, 4] += lamb_alt  # Cannot be landed if 500m in the air

    # The custom objective function required by XGBoost
    def pi_xgb_objective(preds, dtrain):
        labels = dtrain.get_label().astype(int)
        
        # Convert logits to probabilities (Softmax)
        preds_2d = preds.reshape(N, n_classes)
        preds_max = np.max(preds_2d, axis=1, keepdims=True)
        exp_preds = np.exp(preds_2d - preds_max)
        probs = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
        
        # One-hot encode true labels
        y_onehot = np.zeros_like(probs)
        y_onehot[np.arange(N), labels] = 1.0
        
        # Calculate penalty expected value: E[C] = sum_k (P_k * C_k)
        E_C = np.sum(probs * C, axis=1, keepdims=True)
        
        # Gradient = (P - Y) + P * (C - E[C])
        grad = (probs - y_onehot) + probs * (C - E_C)
        
        # Hessian (approximate standard softmax hessian to ensure stable tree growth)
        hess = probs * (1.0 - probs)
        
        # To avoid zero-hessian math errors in XGBoost
        hess = np.maximum(hess, 1e-16)
        
        return grad, hess
        
    return pi_xgb_objective


class PIXGBWrapper:
    """Wraps the XGBoost Booster to provide sklearn-like predict() and predict_proba()."""
    def __init__(self, booster, num_classes):
        self._Booster = booster
        self.num_classes = num_classes
        self.classes_ = np.arange(num_classes)
        
    def predict_proba(self, X):
        dmat = xgb.DMatrix(X)
        return self._Booster.predict(dmat)
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

def train_pi_xgb_fold(X_train, X_test, y_train, y_test, y_raw_test, 
                      feature_cols, le, fold_name):
    # Scale features
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    # Note: SMOTE messes up physics matching between X_raw and X_scaled because it generates synthetic rows.
    # Therefore, we will only apply the physics penalty to the ORIGINAL dataset rows. 
    # Or, simpler: we skip SMOTE for PI-XGB to preserve pure physical telemetry,
    # OR we compute the penalty matrix *after* SMOTE by inverse-transforming.
    
    smote = SMOTE(random_state=config.RANDOM_STATE)
    X_tr_s_smote, y_train_smote = smote.fit_resample(X_tr_s, y_train)
    
    # Inverse transform to get raw features back for the penalty matrix logic
    X_train_raw_smote = scaler.inverse_transform(X_tr_s_smote)
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_tr_s_smote, label=y_train_smote)
    dtest = xgb.DMatrix(X_te_s, label=y_test)
    
    # Build Custom Objective Closure using the raw features
    obj_fn = build_physics_objective(X_train_raw_smote, y_train_smote, feature_cols)
    
    # Params
    params = config.PI_XGB_PARAMS.copy()
    params["objective"] = "multi:softprob"
    params["num_class"] = len(config.PHASE_LABELS)
    params["disable_default_eval_metric"] = 1  # Required when using custom objective
    
    print(f"\n  Training PI-XGB [{fold_name}] with Physics Object...", end=" ", flush=True)
    
    bst = xgb.train(
        params, 
        dtrain, 
        num_boost_round=params.get("n_estimators", 300), 
        obj=obj_fn,
        verbose_eval=False
    )
    print("done.")
    
    # Predict
    preds_prob = bst.predict(dtest)  # shape (N, n_classes)
    y_pred = np.argmax(preds_prob, axis=1)
    y_pred_raw = le.inverse_transform(y_pred)
    
    utils.print_classification_report(y_raw_test, y_pred_raw, f"PI-XGB [{fold_name}]")
    utils.plot_confusion_matrix(y_raw_test, y_pred_raw, f"PI-XGB_{fold_name}")
    utils.plot_roc_curves(y_raw_test, preds_prob, f"PI-XGB_{fold_name}")
    
    macro_f1 = f1_score(y_raw_test, y_pred_raw, 
                        labels=config.PHASE_LABELS, average="macro", zero_division=0)
    
    model_wrapper = PIXGBWrapper(bst, len(config.PHASE_LABELS))
    
    utils.save_model({"model": model_wrapper, "scaler": scaler, "le": le, "feature_cols": feature_cols}, 
                     f"pi_xgb_{fold_name}")
    return macro_f1


def main():
    utils.ensure_dirs()
    print("\n" + "="*55)
    print("  STEP 10 — Physics-Informed XGBoost (PI-XGB)")
    print("="*55)

    df = pd.read_csv(config.FEATURES_PATH)
    df.columns = df.columns.str.strip()
    feature_cols = [c for c in config.FEATURE_COLS if c in df.columns]
    X = df[feature_cols].values
    y_raw = df[config.LABEL_COL].values
    flight_ids = df["flight_id"].values if "flight_id" in df.columns else None

    le = LabelEncoder()
    le.fit(config.PHASE_LABELS)
    y_enc = le.transform(y_raw)

    print(f"\n  Dataset : {X.shape[0]} samples × {X.shape[1]} features")

    splits = []
    if config.SPLIT_MODE == "leave_one_out" and flight_ids is not None:
        for fid in np.unique(flight_ids):
            test_mask = flight_ids == fid
            train_mask = ~test_mask
            splits.append((X[train_mask], X[test_mask],
                           y_enc[train_mask], y_enc[test_mask],
                           y_raw[test_mask], f"fold_rocket_{fid}"))
    else:
        X_tr, X_te, y_tr, y_te, yr_tr, yr_te = train_test_split(
            X, y_enc, y_raw, test_size=config.TEST_SIZE, stratify=y_enc, random_state=config.RANDOM_STATE
        )
        splits.append((X_tr, X_te, y_tr, y_te, yr_te, "random_split"))

    f1_scores = []
    per_fold = []

    for fold_data in splits:
        X_train, X_test, y_train, y_test, y_raw_test, fold_name = fold_data
        f1 = train_pi_xgb_fold(X_train, X_test, y_train, y_test, y_raw_test,
                               feature_cols, le, fold_name)
        f1_scores.append(f1)
        per_fold.append({"model": "PI-XGB", "fold": fold_name, "macro_f1": round(f1, 6)})

    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    print("\n" + "="*55)
    print(f"  PI-XGB Mean Macro F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print("="*55)

    # Update summary
    out = os.path.join(config.OUTPUT_DIR, "model_summary.json")
    if os.path.exists(out):
        with open(out) as f:
            summary = json.load(f)
    else:
        summary = []
        
    summary = [s for s in summary if s.get("model") != "PI-XGB"]
    summary.append({"model": "PI-XGB", "mean_macro_f1": mean_f1, "std_macro_f1": std_f1})
    
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    # Update per-fold summary
    pf_path = os.path.join(config.OUTPUT_DIR, "model_summary_perFold.json")
    if os.path.exists(pf_path):
        with open(pf_path) as f:
            pf_data = json.load(f)
    else:
        pf_data = []
        
    pf_data = [d for d in pf_data if d.get("model") != "PI-XGB"]
    pf_data.extend(per_fold)
    
    with open(pf_path, "w") as f:
        json.dump(pf_data, f, indent=2)

    print("✓ Added PI-XGB to summary files.")

if __name__ == "__main__":
    main()
