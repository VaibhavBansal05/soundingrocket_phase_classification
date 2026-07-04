# Sounding Rocket Flight Phase Classification & Kinematic Modeling

**A physics-informed machine learning pipeline for classifying sounding rocket flight phases from GPS/telemetry data — without a barometric pressure sensor.**

Onboard flight computers typically detect apogee from barometric pressure. This project targets a vehicle class that has no such sensor: it classifies five flight phases (Boost, Coast, Apogee, Descent, Landed) from GPS-derived altitude and velocity alone, trained across **7 sounding rocket flights** and evaluated under rigorous **Leave-One-Rocket-Out Cross-Validation (LOROCV)** — every fold holds out one entire rocket, so reported metrics measure generalisation to a vehicle the model has never seen, not just held-out samples from familiar ones.

The pipeline covers five baseline models (XGBoost, Random Forest, SVM, Bi-LSTM) alongside two novel architectures — **PhaseNet** (a physics-informed multi-scale temporal fusion network) and **PI-XGB** (XGBoost with a physics-penalised custom objective) — plus real-time streaming inference, early apogee prediction, and post-hoc explainability (SHAP/LIME).

## Table of Contents
- [Target Flight Phases](#-target-flight-phases)
- [Models & Contributions](#-sota-novel-models--contributions)
- [Results](#-results-lorocv-mean-macro-f1)
- [Engineered Features](#-engineered-features)
- [Project Structure](#-project-structure-directory)
- [Installation & Quick Start](#-installation--quick-start)
- [Live Stream Emulation](#-live-stream-emulation-07_realtime_predictpy)
- [Evaluation Methodology](#-evaluation--cross-verification)

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
- **Random Forest** — Bagged decision tree ensemble
- **SVM** — Support Vector Machine (RBF Kernel)
- **Bi-LSTM** — Sequence-aware recurrent baseline

---

## 📊 Results (LOROCV Mean Macro F1)

| Model | Mean Macro F1 | Std Dev |
|-------|--------------:|--------:|
| **PI-XGB** | **0.838** | ±0.112 |
| XGBoost | 0.830 | ±0.116 |
| Random Forest | 0.823 | ±0.130 |
| Bi-LSTM | 0.729 | ±0.265 |
| SVM (RBF) | 0.713 | ±0.227 |
| PhaseNet | 0.711 | ±0.284 |

PI-XGB's kinematic penalty matrices give it the most consistent
cross-rocket performance (lowest std dev among the top three). Fold
variance is high for every model on a couple of outlier rockets at the
extremes of the apogee-altitude range — see `outputs/plots/per_fold_f1_lines.png`
after running `05_compare_models.py` for the per-fold breakdown.

> **Note:** `09_phasenet_model.py`'s custom training loop previously had a
> bug where the physics-informed transition loss (`L_transition`) was
> compared against temporally misaligned predictions (batches were
> shuffled, breaking the "previous timestep" assumption the loss depends
> on). This has been fixed — the PhaseNet number above predates the fix
> and should be treated as a lower bound; re-run `09_phasenet_model.py` to
> get updated figures.

---

## 🧮 Engineered Features

13 robust kinetic and statistical features extracted purely from velocity and altitude streams.

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
├── 04_lstm_model.py             ← Generates the Bi-LSTM baseline network
├── 05_compare_models.py         ← Generates bar charts and publication analytics
├── 06_predict.py                ← Batch off-line pipeline inference testing
├── 07_realtime_predict.py       ← Socket/FIFO based Live Streaming Inference simulator
├── 08_apogee_predictor.py       ← Early Flight ML Regressor for maximum altitude predictor
├── 09_phasenet_model.py         ← 🚀 PhaseNet Core training (Keras + Multiscale + Attention)
├── phasenet_layers.py           ← 🧩 Custom PhaseNet layer definitions
├── 10_pi_xgb.py                 ← 💡 PI-XGB (XGBoost with Custom Gradients)
├── 11_explainability.py         ← 🧠 SHAP/LIME Explainability Report Generator
├── verify.py                    ← Dev sanity-check script for config/pipeline consistency
└── requirements.txt             ← Pinned minimum dependency versions
```

---

## 🛠️ Installation & Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Add your flight CSVs to `data/` and execute the pipeline **in this order**
   (`03_train_evaluate.py` rebuilds `outputs/model_summary.json` from
   scratch each run, while `04`/`09`/`10` merge their results into it —
   running `03` after the others will silently wipe their entries, so it
   must go first):

```bash
# Data Preprocessing
python 01_label_data.py          
python 02_feature_engineering.py 

# Baseline ML Models (03 first — see note above)
python 03_train_evaluate.py      
python 04_lstm_model.py          

# 🚀 Novel Models 
python 09_phasenet_model.py      # Trains PhaseNet + generates attention mapping
python 09_phasenet_model.py --ablation  # Optional: PhaseNet component ablation study
python 10_pi_xgb.py              # Trains Custom Objective PI-XGB
python 11_explainability.py      # Output XAI analytics for XGBoost/PI-XGB

# Analytics & Ancillary Functions
python 05_compare_models.py      # Compares all models: bar chart + per-fold F1 line plot
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

Implemented specifically via **Leave-One-Rocket-Out Cross-Validation (LOROCV)**: in each of the 7 folds, one rocket's telemetry is held out entirely as the test set while the remaining six train the model. This evaluates true cross-rocket generalisation rather than just held-out samples from already-seen flights. All evaluation metrics use **macro F1** rather than accuracy, since accuracy would be dominated by the majority Descent class and would mask poor performance on rare-but-safety-critical phases like Boost and Apogee.
