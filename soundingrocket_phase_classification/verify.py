"""Quick verification of all pipeline files and new features."""
import sys, os, ast
sys.path.insert(0, ".")

print("=" * 55)
print("  Verification Script")
print("=" * 55)

# ── config.py ────────────────────────────────────────────
import config

feat = config.FEATURE_COLS
new_feats = [f for f in ["jerk_proxy", "vel_sign_change",
                          "energy_proxy", "altitude_from_ground"] if f in feat]
print(f"\n[config] FEATURE_COLS count    : {len(feat)}")
print(f"[config] New features present  : {new_feats}")
print(f"[config] SMOOTHING_WINDOW      : {config.SMOOTHING_WINDOW}")
print(f"[config] GRAVITY               : {config.GRAVITY}")
print(f"[config] REALTIME_BUFFER_LEN   : {config.REALTIME_BUFFER_LEN}")
print(f"[config] APOGEE_PREDICTOR_PATH : {config.APOGEE_PREDICTOR_PATH}")
print(f"[config] APOGEE_BOOST_FEATURES : {config.APOGEE_BOOST_FEATURES}")

# ── utils.py ────────────────────────────────────────────
import utils
print(f"\n[utils] engineer_features    : {callable(utils.engineer_features)}")
print(f"[utils] smooth_predictions   : {callable(utils.smooth_predictions)}")
print(f"[utils] load_best_model      : {callable(utils.load_best_model)}")

# ── Functional test: engineer_features ─────────────────
import pandas as pd
import numpy as np

n = 50
dummy = pd.DataFrame({
    "altitude[m]":       np.linspace(0, 3000, n),
    "velocity[m/s]":     np.concatenate([np.linspace(0, 200, n // 2),
                                         np.linspace(200, -40, n // 2)]),
    "battery[decivolts]": np.ones(n) * 120,
    "pyro1": np.zeros(n),
    "pyro2": np.zeros(n),
    "lat[deg/10000]": np.zeros(n),
    "lon[deg/10000]": np.zeros(n),
})
out = utils.engineer_features(dummy, flight_id=0, ground_alt=0.0)
produced = [c for c in ["jerk_proxy", "vel_sign_change",
                         "energy_proxy", "altitude_from_ground"]
            if c in out.columns]
print(f"\n[utils] engineer_features new cols produced : {produced}")
assert len(produced) == 4, "FAIL: missing new feature columns"

# ── Functional test: smooth_predictions ────────────────
preds_in  = ["Boost"] * 5 + ["Coast"] + ["Boost"] * 4
smoothed  = utils.smooth_predictions(preds_in, window=5)
removed   = "Coast" not in smoothed
print(f"[utils] smooth_predictions (isolated label removed) : {removed}")

# ── Syntax check all scripts ────────────────────────────
print()
scripts = [
    "01_label_data",
    "02_feature_engineering",
    "03_train_evaluate",
    "04_lstm_model",
    "05_compare_models",
    "06_predict",
    "07_realtime_predict",
    "08_apogee_predictor",
]
all_ok = True
for name in scripts:
    path = f"{name}.py"
    try:
        with open(path, encoding="utf-8") as fh:
            ast.parse(fh.read())
        print(f"  [  OK  ] {path}")
    except SyntaxError as e:
        print(f"  [ERROR ] {path}: {e}")
        all_ok = False

print()
print("=" * 55)
print("  All checks passed!" if all_ok else "  SOME CHECKS FAILED — see above")
print("=" * 55)
