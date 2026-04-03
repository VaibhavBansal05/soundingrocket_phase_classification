# Sounding Rocket Flight Phase Classification

Supervised ML pipeline for classifying sounding rocket flight phases using
flight computer sensor data from **7 flights** (Leave-One-Rocket-Out CV).
Also includes real-time streaming prediction and apogee altitude regression.

## Target Classes
| Phase | Description |
|-------|-------------|
| `Boost` | Motor burning — rapid altitude and velocity gain |
| `Coast` | Motor burnout — still ascending under inertia |
| `Apogee` | Peak altitude — near-zero vertical velocity |
| `Descent` | Falling under parachute |
| `Landed` | On ground — near-zero velocity and altitude |

## Models (Classification)
- **XGBoost** — Gradient boosted trees  (`mean macro F1 ≈ 0.840`)
- **Random Forest** — Ensemble tree model (`mean macro F1 ≈ 0.837`)
- **SVM** — Support Vector Machine (RBF kernel) (`mean macro F1 ≈ 0.715`)
- **LSTM** — Bidirectional LSTM — sequence-aware (`mean macro F1 ≈ 0.731`)

## Input CSV Format (exact columns required)
```
link, ts[deciseconds], state, errors, lat[deg/10000], lon[deg/10000],
altitude[m], velocity[m/s], battery[decivolts], pyro1, pyro2
```
> **No pressure column** — pipeline is fully configured without it.

The `state` column contains raw flight computer codes:
- `3` = Boost, `4` = Coast/Apogee region, `5` = Descent, `6` = Landed

## Engineered Features
| Feature | Description |
|---------|-------------|
| `alt_diff` | Δaltitude per timestep |
| `vel_diff` | Δvelocity per timestep ≈ acceleration |
| `acc_proxy` | 2nd Δ of altitude |
| `jerk_proxy` | 3rd Δ of altitude (rate of acceleration change) |
| `alt_rolling_mean/std` | Rolling statistics of altitude |
| `vel_rolling_mean/std` | Rolling statistics of velocity |
| `speed_abs` | |velocity| |
| `is_ascending` | 1 if velocity > 0 |
| `vel_sign_change` | 1 when velocity crosses zero (apogee signal) |
| `energy_proxy` | ½v² + g·(alt − ground_alt) [J/kg] |
| `altitude_from_ground` | alt − estimated launch-site elevation |

## Project Structure
```
soundingrocket_phase_classification/
├── data/                        ← Place your CSVs here
├── outputs/
│   ├── labeled/                 ← Auto-labeled CSVs
│   ├── models/                  ← .pkl / .keras / _meta.pkl
│   └── plots/                   ← All evaluation and prediction plots
├── config.py                    ← All parameters (columns, thresholds, hyperparams)
├── utils.py                     ← Shared helpers + engineer_features (single source)
├── 01_label_data.py             ← Physics-based auto-labeling
├── 02_feature_engineering.py    ← Feature extraction & dataset merge
├── 03_train_evaluate.py         ← XGBoost / RF / SVM training
├── 04_lstm_model.py             ← LSTM training (saves _meta.pkl with scaler)
├── 05_compare_models.py         ← Publication comparison chart
├── 06_predict.py                ← Batch inference on new CSV + smoothing
├── 07_realtime_predict.py       ← [NEW] Real-time streaming classifier
└── 08_apogee_predictor.py       ← [NEW] Apogee altitude regression
```

## Quick Start
```bash
pip install -r requirements.txt

# 1. Copy your CSVs into data/
# 2. Run pipeline in order:
python 01_label_data.py          # Auto-label + verify timeline plots
python 02_feature_engineering.py # Feature engineering + merge
python 03_train_evaluate.py      # Train XGBoost, RF, SVM
python 04_lstm_model.py          # Train LSTM (saves .keras + _meta.pkl)
python 05_compare_models.py      # Paper-ready comparison chart
python 08_apogee_predictor.py    # Train apogee altitude regressor
```

## Batch Prediction (06_predict.py)
```bash
python 06_predict.py --csv your_flight.csv --model xgboost
python 06_predict.py --csv your_flight.csv --model lstm --no-smooth
```
Outputs: `outputs/predicted_<name>_<model>.csv` with `predicted_phase` and
`confidence` columns.  Also saves a two-panel plot (altitude + confidence).

## Real-Time Streaming Prediction (07_realtime_predict.py)
```bash
# Simulate live ground station — fastest:
python 07_realtime_predict.py --csv your_flight.csv

# With 100 ms tick delay (simulate real data rate):
python 07_realtime_predict.py --csv your_flight.csv --model xgboost --delay 100

# Use LSTM (needs 04 to have been run):
python 07_realtime_predict.py --csv your_flight.csv --model lstm

# Suppress live output (batch mode):
python 07_realtime_predict.py --csv your_flight.csv --no-live
```
Features colour-coded live terminal output, FIFO buffer, per-tick confidence,
phase transition event log, and a summary plot.

## Apogee Altitude Prediction (08_apogee_predictor.py)
```bash
python 08_apogee_predictor.py                         # train + evaluate
python 08_apogee_predictor.py --predict --csv f.csv   # predict for new flight
```
Given only the **Boost phase** data of a flight, predicts peak apogee altitude.
Useful for range safety pre-flight estimation and parachute decision support.

## Evaluation Protocol
**Leave-One-Rocket-Out Cross-Validation** — trains on N-1 rockets, tests on 1,
rotates through all flights. Tests genuine cross-vehicle generalisation with no
data leakage. Best fold is selected by macro F1 automatically.

Per model outputs:
- Confusion matrix
- Per-class precision / recall / F1
- ROC/AUC curves (One-vs-Rest)
- Feature importance (XGBoost + RF)
- Learning curves (LSTM)
- Per-fold JSON (`model_summary_perFold.json`)

## Requirements
```
pip install pandas numpy scikit-learn xgboost tensorflow matplotlib seaborn joblib imbalanced-learn scipy
```
