"""
01_label_data.py
────────────────
Physics-based automatic labeling of flight phases.

Strategy (hybrid — uses flight computer state + physics refinement):
  1. Map numeric state codes → coarse phase labels via RAW_STATE_MAP
  2. Refine state=4 (Coast) region to split into Coast / Apogee:
       - Find apogee index (max altitude)
       - Window around apogee where |velocity| < APOGEE_VEL_THRESHOLD → Apogee
       - Rest of state=4 region → Coast
  3. Apply velocity/altitude sanity checks for Boost boundaries
  4. Confirm Landed using low altitude + low velocity

Outputs: labeled CSVs → outputs/labeled/  +  timeline plots
"""

import _compat  # UTF-8 console fix for Windows
import os
import numpy as np
import pandas as pd
import config
import utils


def label_flight(df: pd.DataFrame, flight_name: str) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()   # preserve original case/brackets

    alt  = df[config.COL_ALTITUDE].values.astype(float)
    vel  = df[config.COL_VELOCITY].values.astype(float)
    raw  = df[config.COL_RAW_STATE].values
    n    = len(df)

    labels = np.empty(n, dtype=object)

    # ── Step 1: coarse map from flight computer state ────────────────
    for i in range(n):
        labels[i] = config.RAW_STATE_MAP.get(int(raw[i]), "Coast")

    # ── Step 2: split Coast region into Coast / Apogee ───────────────
    coast_mask = labels == "Coast"
    if coast_mask.sum() > 0:
        coast_indices = np.where(coast_mask)[0]

        # Apogee = sample with max altitude within the coast region
        apogee_idx = coast_indices[np.argmax(alt[coast_indices])]

        # Apogee window: half-window or velocity threshold, whichever is wider
        half_win = max(2, int(n * config.APOGEE_WINDOW_FRAC))

        for i in coast_indices:
            near_peak = abs(i - apogee_idx) <= half_win
            slow       = abs(vel[i]) < config.APOGEE_VEL_THRESHOLD
            if near_peak or slow:
                labels[i] = "Apogee"
            else:
                labels[i] = "Coast"

    # ── Step 3: Boost sanity check ────────────────────────────────────
    # If anything labeled Boost has suspiciously low velocity, reclassify
    for i in range(n):
        if labels[i] == "Boost" and vel[i] < 5.0:
            labels[i] = "Coast"

    # ── Step 4: Landed confirmation ───────────────────────────────────
    ground_alt = np.percentile(alt, 2)
    for i in range(n):
        if labels[i] in ("Descent", "Landed"):
            low_alt = (alt[i] - ground_alt) < config.LANDED_ALT_THRESHOLD
            low_vel = abs(vel[i]) < config.LANDED_VEL_THRESHOLD
            if low_alt and low_vel:
                labels[i] = "Landed"

    df[config.LABEL_COL] = labels

    # ── Summary ─────────────────────────────────────────────────────
    counts = pd.Series(labels).value_counts()
    print(f"  {flight_name}:")
    for phase in config.PHASE_LABELS:
        print(f"    {phase:10s}: {counts.get(phase, 0):5d} samples")

    return df


def main():
    utils.ensure_dirs()
    print("\n" + "="*55)
    print("  STEP 1 — Physics-Based Phase Labeling")
    print("="*55)

    flights = utils.load_flight_csvs()

    for name, df in flights.items():
        print(f"\nLabeling: {name}")
        labeled = label_flight(df, name)
        out_path = os.path.join(config.LABELED_DIR, f"{name}_labeled.csv")
        labeled.to_csv(out_path, index=False)
        print(f"  → Saved: {out_path}")
        utils.plot_phase_timeline(labeled, name)

    print("\n✓ Labeling complete. Check outputs/labeled/ and outputs/plots/")


if __name__ == "__main__":
    main()
