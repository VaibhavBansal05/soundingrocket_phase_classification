"""
02_feature_engineering.py
──────────────────────────
Reads labeled CSVs → engineers features → produces one merged dataset.

All feature engineering lives in utils.engineer_features().  Any changes
to the feature set should be made there — this script just orchestrates.

Engineered features:
  alt_diff            Δaltitude per timestep
  vel_diff            Δvelocity per timestep  ≈ acceleration
  acc_proxy           second Δ of altitude
  jerk_proxy          third Δ of altitude (rate of acceleration change)
  alt_rolling_mean    rolling mean of altitude
  vel_rolling_mean    rolling mean of velocity
  alt_rolling_std     rolling std  of altitude
  vel_rolling_std     rolling std  of velocity
  speed_abs           |velocity|  (symmetric, useful near apogee)
  is_ascending        1 if velocity > 0 else 0
  vel_sign_change     1 at samples where velocity crosses zero
  energy_proxy        0.5·v² + g·(alt - ground_alt)  [J/kg]
  altitude_from_ground altitude above estimated launch site

flight_id is added for Leave-One-Rocket-Out CV.
"""

import _compat  # UTF-8 console fix for Windows
import os
import numpy as np
import pandas as pd
import config
import utils


def main():
    utils.ensure_dirs()
    print("\n" + "="*55)
    print("  STEP 2 — Feature Engineering")
    print("="*55)

    labeled_files = sorted([
        f for f in os.listdir(config.LABELED_DIR)
        if f.endswith("_labeled.csv")
    ])
    if not labeled_files:
        raise FileNotFoundError(
            "No labeled CSVs found. Run 01_label_data.py first."
        )

    all_dfs = []
    for fid, fname in enumerate(labeled_files):
        path = os.path.join(config.LABELED_DIR, fname)
        df   = pd.read_csv(path)
        print(f"\n  [{fid}] {fname}  ({len(df)} rows)")

        # Compute per-flight ground altitude so altitude_from_ground is correct
        alt = df[config.COL_ALTITUDE].astype(float)
        ground_alt = float(np.percentile(alt.values, 2))

        df_feat = utils.engineer_features(df, flight_id=fid,
                                          ground_alt=ground_alt)
        all_dfs.append(df_feat)

    merged = pd.concat(all_dfs, ignore_index=True)

    feature_cols_present = [c for c in config.FEATURE_COLS if c in merged.columns]
    before = len(merged)
    merged.dropna(subset=feature_cols_present, inplace=True)
    print(f"\n  Dropped {before - len(merged)} NaN rows → {len(merged)} total samples")

    # Class distribution
    print("\n  Class distribution:")
    counts = merged[config.LABEL_COL].value_counts()
    max_n  = counts.max()
    for phase in config.PHASE_LABELS:
        n   = counts.get(phase, 0)
        bar = "█" * max(1, int(n / max(max_n // 30, 1)))
        print(f"    {phase:10s}: {n:6d}  {bar}")

    merged.to_csv(config.FEATURES_PATH, index=False)
    print(f"\n✓ Saved → {config.FEATURES_PATH}  shape={merged.shape}")
    print(f"  Features: {feature_cols_present}")


if __name__ == "__main__":
    main()
