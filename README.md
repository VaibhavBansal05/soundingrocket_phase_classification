# Sounding Rocket Flight Phase Classification & Kinematic Modeling

**A comprehensive, physics-informed Machine Learning pipeline for the automated classification of sounding rocket flight phases from telemetry data.**

This project uses flight computer sensory data across **7 sounding rocket flights** with rigorous **Leave-One-Rocket-Out Cross Validation (LORO-CV)**. It incorporates standard baselines as well as **state-of-the-art custom architectures** like **PhaseNet** (Physics-Informed Multi-Scale Temporal Fusion Network) and **PI-XGB** (Physics-Informed Custom Objective XGBoost). Additionally, it features real-time streaming inference, early apogee altitude prediction, and deep model explainability (XAI) using SHAP and LIME.

---

## 🎯 Target Flight Phases

The model classifies real-time telemetry into one of 5 critical flight phases:

| Phase | Description |
|-------|-------------|
| 🚀 `Boost` | Motor burning — rapid altitude and velocity gain |
| 💨 `Coast` | Motor burnout — ascending under inertia |
| ⛰️ `Apogee` | Peak altitude — near-zero vertical velocity |
| 🪂 `Descent` | Falling safely under parachute |
| 🌍 `Landed` | On ground — near-zero velocity and altitude |

---

## 🧠 SOTA Novel Models & Contributions

This repository contains custom, highly specialized architectures designed to inject physical domain knowledge directly into learning objectives:

### 1. PhaseNet (`09_phasenet_model.py`)
A custom end-to-end deep learning framework featuring:
- **Multi-Scale TCN Encoder:** Parallel dilated Temporal Convolutional Networks capture patterns at fine (k=3), medium (k=7), and coarse (k=15) temporal scales.
- **Cross-Scale Attention Fusion:** A multi-head cross-attention mechanism dynamically learns which temporal scale matters per flight instant (interpretable attention).
- **Physics-Informed Constraint Loss:** A custom Keras loss function `L_total = L_CE + λ₁·L_energy + λ₂·L_transition + λ₃·L_kinematics` penalizes impossible predictions (e.g., predicting `Descent` when energy is increasing).

### 2. PI-XGB (Physics-Informed XGBoost) (`10_pi_xgb.py`)
An advanced gradient boosting strategy utilizing a **custom objective function**. Instead of relying purely on categorical cross-entropy, PI-XGB leverages mathematical derivations of expected penalty, integrating an algorithmic penalty matrix $C$ into exact gradients and approximate Hessians. High-speed, natively compatible with tree boosters, and prevents kinematically impossible state predictions.

### 3. Explainability Suite (XAI) (`11_explainability.py`)
A transparency package enabling deep interpretability of tree models globally and locally.
- **SHAP Summary Plots:** Determines global feature impacts across all telemetry.
- **SHAP Waterfall Plots:** Visualizes exact token-level feature push/pull locally.
- **LIME Explainability:** Decodes the decision boundary around difficult misclassifications and transition boundaries natively in HTML.

### 4. Baseline Models
The project thoroughly evaluates baselines:
- **XGBoost** — Gradient boosted trees
- **Random Forest** — Ensemble mapping 
- **SVM** — Support Vector Machine (RBF Kernel)
- **Bi-LSTM** — Standard sequential sequence-aware deep neural network

---

## 🧮 Engineered Features

13 robust kinetic and statistical features extracted purely from velocity and altitude strings.

| Feature | Physical Interpretation |
|---------|-------------|
| `alt_diff` / `vel_diff` | $\Delta$altitude and $\Delta$velocity per timestep (proxy for acceleration) |
| `acc_proxy` / `jerk_proxy` | $2^{nd}$ and $3^{rd}$ derivatives of altitude |
| `alt_rolling_mean/std` | Moving statistics for smoothing hardware sensor noise |
| `vel_rolling_mean/std` | Moving statistical variance over velocity |
| `speed_abs` | Absolute speed $\mid V \mid$ |
| `is_ascending` | Binary ascending boolean |
| `vel_sign_change` | Triggered precisely at apogee velocity inversion |
| `energy_proxy` | $\frac{1}{2}v^2 + g \Delta h$ (Kinetic + Potential Specific Energy) |
| `altitude_from_ground` | Bias-corrected relative altitude normalized by launch site elevation |

---

## 📂 Project Structure Directory

```text
soundingrocket_phase_classification/
├── data/                        ← Raw CSV flight telemetry data logs
├── outputs/
│   ├── labeled/                 ← Automatically physics-labeled ground-truth CSVs
│   ├── models/                  ← Persisted pipelines (*.pkl, *.keras, *_meta.pkl)
│   └── plots/                   ← Visualizations, heatmaps, CMs, and SHAP trees
├── config.py                    ← Master hyperparameter and mapping configurations
├── utils.py                     ← Shared generic IO, parsing, and pipeline macros
├── 01_label_data.py             ← Deterministic state-code to physics map labeling
├── 02_feature_engineering.py    ← Feature space augmentation / dataset fusion
├── 03_train_evaluate.py         ← Trains standard SVM, default XGBoost, and RF 
├── 04_lstm_model.py             ← Generates the Bi-LSTM baseline baseline network
├── 05_compare_models.py         ← Generates bar charts and publication analytics
├── 06_predict.py                ← Batch off-line pipeline inference testing
├── 07_realtime_predict.py       ← Socket/FIFO based Live Streaming Inference simulator
├── 08_apogee_predictor.py       ← Early Flight ML Regressor for maximum altitude predictor
├── 09_phasenet_model.py         ← 🚀 PhaseNet Core training (Keras + Multiscale + Attention)
├── phasenet_layers.py           ← 🧩 Custom PhaseNet layer definitions
├── 10_pi_xgb.py                 ← 💡 PI-XGB (XGBoost with Custom Gradients)
└── 11_explainability.py         ← 🧠 SHAP/LIME Explainability Report Generator
```

---

## 🛠️ Installation & Quick Start

1. Install minimum dependencies:
```bash
pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn joblib imbalanced-learn scipy shap lime
```

2. Add your flight CSVs to `data/` and execute the pipeline sequentially:

```bash
# Data Preprocessing
python 01_label_data.py          
python 02_feature_engineering.py 

# Baseline ML Models
python 03_train_evaluate.py      
python 04_lstm_model.py          

# 🚀 Novel Models 
python 09_phasenet_model.py      # Trains PhaseNet + generates attention mapping
python 10_pi_xgb.py              # Trains Custom Objective PI-XGB
python 11_explainability.py      # Output XAI analytics for XGBoost/PI-XGB

# Analytics & Ancillary Functions
python 05_compare_models.py      # Compares all built algorithms 
python 08_apogee_predictor.py    # Generates apogee extrapolations
```

---

## 📡 Live Stream Emulation (`07_realtime_predict.py`)

Emulates real-world UDP/Socket downlinks using a rolling FIFO buffer system. Includes color-coated temporal transition logs indicating exact timeline states, buffer window loading, and sub-10ms batch metrics. 

```bash
# Simulate live execution at 100ms clock ticks natively
python 07_realtime_predict.py --csv flight_data.csv --model xgboost --delay 100
```

---

## 🔬 Evaluation & Cross Verification

Implemented specifically via **Leave-One-Rocket-Out Cross-Validation (LORO-CV)**. By explicitly dropping an entire vehicle's telemetry from a fold, the platform evaluates absolute domain adaptation robustness spanning hardware variations and separate flight kinetic footprints. Outlier evaluation metrics strictly use **Macro F1 Scores** to circumvent major data class imbalances spanning extremely short transitions (Boost, Apogee).
