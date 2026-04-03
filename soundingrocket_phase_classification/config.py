# config.py — Central configuration for the pipeline

import os

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DATA_DIR      = "data"
OUTPUT_DIR    = "outputs"
LABELED_DIR           = os.path.join(OUTPUT_DIR, "labeled")
FEATURES_PATH         = os.path.join(OUTPUT_DIR, "features_dataset.csv")
MODELS_DIR            = os.path.join(OUTPUT_DIR, "models")
PLOTS_DIR             = os.path.join(OUTPUT_DIR, "plots")
APOGEE_PREDICTOR_PATH = os.path.join(MODELS_DIR,  "apogee_predictor.pkl")

# ─────────────────────────────────────────────
# COLUMN NAMES  — exact headers from your CSVs
# ─────────────────────────────────────────────
COL_LINK      = "link"               # Telemetry link ID (1 or 2)
COL_TIME      = "ts[deciseconds]"    # Timestamp in deciseconds
COL_RAW_STATE = "state"              # Numeric state from flight computer
COL_ERRORS    = "errors"
COL_LAT       = "lat[deg/10000]"
COL_LON       = "lon[deg/10000]"
COL_ALTITUDE  = "altitude[m]"        # Altitude in metres
COL_VELOCITY  = "velocity[m/s]"      # Vertical velocity m/s
COL_BATTERY   = "battery[decivolts]"
COL_PYRO1     = "pyro1"              # 0/1 ejection charge fired
COL_PYRO2     = "pyro2"              # 0/1 ejection charge fired
# NOTE: No pressure sensor — not used anywhere in this pipeline

# ─────────────────────────────────────────────
# NUMERIC STATE → PHASE MAPPING
# Flight computer codes observed in data:
#   3 = Boost  |  4 = Coast (will be split into Coast/Apogee by labeler)
#   5 = Descent  |  6 = Landed
# ─────────────────────────────────────────────
RAW_STATE_MAP = {
    3: "Boost",
    4: "Coast",
    5: "Descent",
    6: "Landed",
}

# Feature columns used for ML (time, raw state, errors, link excluded)
FEATURE_COLS = [
    COL_ALTITUDE,       # altitude[m]
    COL_VELOCITY,       # velocity[m/s]
    COL_PYRO1,          # pyro1
    COL_PYRO2,          # pyro2
    COL_LAT,            # lat[deg/10000]
    COL_LON,            # lon[deg/10000]
    COL_BATTERY,        # battery[decivolts]
    # Engineered features (added by utils.engineer_features):
    "alt_diff",
    "vel_diff",
    "acc_proxy",
    "jerk_proxy",
    "alt_rolling_mean",
    "vel_rolling_mean",
    "alt_rolling_std",
    "vel_rolling_std",
    "speed_abs",
    "is_ascending",
    "vel_sign_change",
    "energy_proxy",
    "altitude_from_ground",
]

# ─────────────────────────────────────────────
# LABELING THRESHOLDS
# Tuned to actual data: vel -39→+229 m/s, alt 0→3466 m
# ─────────────────────────────────────────────
APOGEE_VEL_THRESHOLD = 10.0
LANDED_ALT_THRESHOLD = 20.0
LANDED_VEL_THRESHOLD =  5.0
BOOST_VEL_MIN        = 30.0
APOGEE_WINDOW_FRAC   =  0.04

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
ROLLING_WINDOW        = 5
SMOOTHING_WINDOW      = 7    # post-prediction temporal smoothing (odd number)
GRAVITY               = 9.81  # m/s² for energy_proxy calculation

# ─────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
SPLIT_MODE   = "leave_one_out"
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ─────────────────────────────────────────────
# MODEL HYPERPARAMETERS
# ─────────────────────────────────────────────
XGB_PARAMS = {
    "n_estimators":      300,
    "max_depth":           6,
    "learning_rate":    0.05,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "eval_metric":    "mlogloss",
    "random_state":  RANDOM_STATE,
    "n_jobs":             -1,
}

RF_PARAMS = {
    "n_estimators":      300,
    "max_depth":        None,
    "min_samples_split":   5,
    "random_state":  RANDOM_STATE,
    "n_jobs":             -1,
}

SVM_PARAMS = {
    "kernel":  "rbf",
    "C":        10.0,
    "gamma":  "scale",
    "decision_function_shape": "ovr",
}

# ─────────────────────────────────────────────
# LSTM HYPERPARAMETERS
# ─────────────────────────────────────────────
LSTM_SEQUENCE_LEN = 20
LSTM_UNITS        = [64, 32]
LSTM_DROPOUT      = 0.3
LSTM_BATCH_SIZE   = 32
LSTM_EPOCHS       = 50
LSTM_PATIENCE     = 10

# ─────────────────────────────────────────────
# PHASENET HYPERPARAMETERS  (Step 09)
# Physics-Informed Multi-Scale Temporal Fusion Network
# ─────────────────────────────────────────────
PHASENET_SEQUENCE_LEN   = 20          # same as LSTM for fair comparison

# Multi-Scale TCN Encoder — kernel sizes for each branch
PHASENET_TCN_SCALES     = [3, 7, 15]  # [fine, medium, coarse]
PHASENET_TCN_FILTERS    = 48          # filters per conv layer per branch
PHASENET_TCN_DILATIONS  = {
    3:  [1, 2, 4, 8],                 # fine-grain: 4 layers
    7:  [1, 2, 4],                    # medium-grain: 3 layers
    15: [1, 2],                       # coarse-grain: 2 layers
}

# Cross-Scale Attention Fusion
PHASENET_ATTENTION_HEADS   = 4
PHASENET_ATTENTION_KEY_DIM = 36

# Classification head
PHASENET_DENSE_UNITS = [96, 48]
PHASENET_DROPOUT     = 0.3

# Physics-Informed Loss weights  (λ values)
PHASENET_LAMBDA_ENERGY     = 0.1      # energy monotonicity constraint
PHASENET_LAMBDA_TRANSITION = 0.2      # legal phase ordering constraint
PHASENET_LAMBDA_KINEMATICS = 0.1      # velocity-phase consistency

# Training
PHASENET_BATCH_SIZE = 64
PHASENET_EPOCHS     = 60
PHASENET_PATIENCE   = 15
PHASENET_LR_INIT    = 1e-3
PHASENET_LR_MIN     = 1e-6

# Legal phase transitions (used by physics loss)
# Index mapping: Boost=0, Coast=1, Apogee=2, Descent=3, Landed=4
PHASENET_LEGAL_TRANSITIONS = {
    0: [0, 1],        # Boost   → Boost, Coast
    1: [1, 2],        # Coast   → Coast, Apogee
    2: [2, 3],        # Apogee  → Apogee, Descent
    3: [3, 4],        # Descent → Descent, Landed
    4: [4],           # Landed  → Landed
}

# ─────────────────────────────────────────────
# CLASS LABELS
# ─────────────────────────────────────────────
PHASE_LABELS = ["Boost", "Coast", "Apogee", "Descent", "Landed"]
LABEL_COL    = "phase"

# ─────────────────────────────────────────────
# APOGEE ALTITUDE PREDICTOR (Step 08)
# ─────────────────────────────────────────────
# Features extracted from the Boost phase only
APOGEE_BOOST_FEATURES = [
    "boost_max_velocity",      # peak velocity during boost
    "boost_mean_velocity",     # average velocity during boost
    "boost_max_acc",           # peak acceleration proxy
    "boost_mean_acc",          # mean acceleration proxy
    "boost_duration",          # number of samples in boost phase
    "boost_alt_at_burnout",    # altitude at end of boost
    "boost_vel_rate",          # (max_vel - first_vel) / duration
    "boost_energy_at_burnout", # kinetic + potential energy at burnout
]

# ─────────────────────────────────────────────
# REAL-TIME PREDICTOR (Step 07)
# ─────────────────────────────────────────────
REALTIME_BUFFER_LEN  = LSTM_SEQUENCE_LEN  # samples kept in FIFO buffer
REALTIME_DEFAULT_MODEL = "xgboost"
