"""
03_train_evaluate.py
─────────────────────
Trains XGBoost, Random Forest, SVM.
Evaluation: confusion matrix, per-class F1, ROC/AUC, feature importance.
Split: Leave-One-Rocket-Out (or random — set in config.py).

Also writes outputs/model_summary_perFold.json for metric-driven
best-fold selection used by 06_predict.py.
"""

import _compat  # UTF-8 console fix for Windows
import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb

import config
import utils


def load_dataset():
    df = pd.read_csv(config.FEATURES_PATH)
    df.columns = df.columns.str.strip()

    feature_cols = [c for c in config.FEATURE_COLS if c in df.columns]
    X          = df[feature_cols].values
    y_raw      = df[config.LABEL_COL].values
    flight_ids = df["flight_id"].values if "flight_id" in df.columns else None

    le    = LabelEncoder()
    le.fit(config.PHASE_LABELS)
    y_enc = le.transform(y_raw)

    return X, y_enc, y_raw, flight_ids, feature_cols, le


def get_splits(X, y_enc, y_raw, flight_ids):
    splits = []
    if config.SPLIT_MODE == "leave_one_out" and flight_ids is not None:
        for fid in np.unique(flight_ids):
            test_mask  = flight_ids == fid
            train_mask = ~test_mask
            splits.append((X[train_mask], X[test_mask],
                           y_enc[train_mask], y_enc[test_mask],
                           y_raw[test_mask], f"fold_rocket_{fid}"))
    else:
        X_tr, X_te, y_tr, y_te, yr_tr, yr_te = train_test_split(
            X, y_enc, y_raw,
            test_size=config.TEST_SIZE,
            stratify=y_enc,
            random_state=config.RANDOM_STATE
        )
        splits.append((X_tr, X_te, y_tr, y_te, yr_te, "random_split"))
    return splits


def build_models():
    return {
        "XGBoost":      xgb.XGBClassifier(**config.XGB_PARAMS),
        "RandomForest": RandomForestClassifier(**config.RF_PARAMS),
        "SVM":          SVC(**config.SVM_PARAMS, probability=True),
    }


def train_evaluate(model, model_name, X_train, X_test,
                   y_train, y_test, y_raw_test, feature_cols, le, fold_name):
    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_train)
    X_te_s   = scaler.transform(X_test)

    smote = SMOTE(random_state=config.RANDOM_STATE)
    X_tr_s, y_train = smote.fit_resample(X_tr_s, y_train)

    print(f"\n  Training {model_name} [{fold_name}] ...", end=" ", flush=True)
    model.fit(X_tr_s, y_train)
    print("done.")

    y_pred      = model.predict(X_te_s)
    y_pred_prob = model.predict_proba(X_te_s)
    y_pred_raw  = le.inverse_transform(y_pred)

    utils.print_classification_report(y_raw_test, y_pred_raw,
                                      f"{model_name} [{fold_name}]")
    utils.plot_confusion_matrix(y_raw_test, y_pred_raw,
                                f"{model_name}_{fold_name}")
    utils.plot_roc_curves(y_raw_test, y_pred_prob,
                          f"{model_name}_{fold_name}")
    utils.plot_feature_importance(model, feature_cols,
                                  f"{model_name}_{fold_name}")

    macro_f1 = f1_score(y_raw_test, y_pred_raw,
                        labels=config.PHASE_LABELS,
                        average="macro", zero_division=0)
    utils.save_model({"model": model, "scaler": scaler,
                      "le": le, "feature_cols": feature_cols},
                     f"{model_name.lower().replace(' ','_')}_{fold_name}")
    return macro_f1


def main():
    utils.ensure_dirs()
    print("\n" + "="*55)
    print("  STEP 3 — Training & Evaluation")
    print("="*55)

    X, y_enc, y_raw, flight_ids, feature_cols, le = load_dataset()
    print(f"\n  Dataset : {X.shape[0]} samples × {X.shape[1]} features")
    print(f"  Features: {feature_cols}")
    print(f"  Split   : {config.SPLIT_MODE}")

    splits   = get_splits(X, y_enc, y_raw, flight_ids)
    summary  = {}
    per_fold = []          # for metric-driven best-fold selection

    for fold_data in splits:
        X_train, X_test, y_train, y_test, y_raw_test, fold_name = fold_data
        for model_name, model in build_models().items():
            f1 = train_evaluate(model, model_name, X_train, X_test,
                                y_train, y_test, y_raw_test,
                                feature_cols, le, fold_name)
            summary.setdefault(model_name, []).append(f1)
            per_fold.append({"model": model_name,
                             "fold":  fold_name,
                             "macro_f1": round(f1, 6)})

    # ── Aggregate summary ─────────────────────────────────────────────
    print("\n" + "="*55)
    print("  SUMMARY — Mean Macro F1 across folds")
    print("="*55)
    rows = []
    for mname, scores in summary.items():
        mean_f1 = np.mean(scores)
        std_f1  = np.std(scores)
        print(f"  {mname:15s}: {mean_f1:.4f} ± {std_f1:.4f}")
        rows.append({"model": mname,
                     "mean_macro_f1": mean_f1,
                     "std_macro_f1":  std_f1})

    # Save aggregate summary
    out = os.path.join(config.OUTPUT_DIR, "model_summary.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)

    # Save per-fold summary (used by load_best_model in utils.py)
    pf_out = os.path.join(config.OUTPUT_DIR, "model_summary_perFold.json")
    with open(pf_out, "w") as f:
        json.dump(per_fold, f, indent=2)
    print(f"\n✓ Summary        → {out}")
    print(f"✓ Per-fold JSON  → {pf_out}")
    print("✓ Plots          → outputs/plots/")
    print("✓ Models         → outputs/models/")


if __name__ == "__main__":
    main()
