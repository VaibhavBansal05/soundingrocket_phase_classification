"""
11_explainability.py
────────────────────
SHAP & LIME Global and Local Explainability for the Phase Classification model.

Produces publication-ready interpretability artifacts:
  - SHAP Summary Plot (Global Feature Importance)
  - SHAP Waterfall Plot (Local single-prediction explanation)
  - LIME text/tabular explanation (Local interpretability)

Defaults to interpreting the PI-XGB model if available, otherwise standard XGBoost.
"""

import _compat
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

import shap
import lime
import lime.lime_tabular

import config
import utils

import xgboost as xgb

class PIXGBWrapper:
    """Wraps the XGBoost Booster to provide sklearn-like predict() and predict_proba()."""
    def __init__(self, booster, num_classes):
        self.booster = booster
        self.num_classes = num_classes
        self.classes_ = np.arange(num_classes)
        
    def predict_proba(self, X):
        dmat = xgb.DMatrix(X)
        return self.booster.predict(dmat)
        
    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

def load_data_and_model():
    """Loads the model, scales the data, and returns everything needed for XAI."""
    print("  Loading data and best model for explanations...")
    
    # Try PI-XGB first, fallback to standard XGBoost
    try:
        saved, fname = utils.load_best_model("PI-XGB")
        model_name = "PI-XGB"
    except FileNotFoundError:
        print("  PI-XGB not found, falling back to standard XGBoost...")
        saved, fname = utils.load_best_model("XGBoost")
        model_name = "XGBoost"

    model = saved["model"]
    scaler = saved["scaler"]
    le = saved["le"]
    feature_cols = saved["feature_cols"]
    
    print(f"  Using model: {model_name} from {fname}")

    df = pd.read_csv(config.FEATURES_PATH)
    df.columns = df.columns.str.strip()
    X = df[feature_cols].fillna(0).values
    y_raw = df[config.LABEL_COL].values
    y_enc = le.transform(y_raw)
    
    # Scale ALL data for explainability (we just want to explain the predictions)
    X_scaled = scaler.transform(X)
    
    return model, X_scaled, y_enc, feature_cols, X, le, model_name


def generate_shap_plots(model, X_scaled, feature_cols, out_dir, model_name):
    """
    Generates Global (Summary) and Local (Waterfall/Force) SHAP plots.
    """
    print("  Generating SHAP plots...")
    
    # SHAP is computationally heavy, use a background subset of 500 random elements
    np.random.seed(42)
    idx_bg = np.random.choice(X_scaled.shape[0], min(500, X_scaled.shape[0]), replace=False)
    X_bg = X_scaled[idx_bg]
    
    # Use TreeExplainer specifically for XGBoost / PI-XGB booster
    booster = getattr(model, "_Booster", getattr(model, "booster", model.get_booster() if hasattr(model, "get_booster") else model))
    explainer = shap.TreeExplainer(booster)
    
    # Calculate SHAP values for the subset
    shap_values = explainer.shap_values(X_bg)
    
    # Depending on xgboost version, shap_values might be a list (one per class)
    # or a 3D array (samples, features, classes). Let's extract safely.
    if isinstance(shap_values, list):
        shap_vals_matrix = np.array(shap_values) # (classes, samples, features)
        shap_vals_matrix = np.transpose(shap_vals_matrix, (1, 2, 0)) # -> (samples, features, classes)
    else:
        shap_vals_matrix = shap_values
        
    class_names = config.PHASE_LABELS

    # 1. Global Summary Bar Plot
    # Explains average impact of features on all classes
    plt.figure(figsize=(10, 6))
    plt.title(f"SHAP Global Feature Importance ({model_name})", fontsize=14, pad=15)
    
    # Create the plot, don't show it, save it
    # SHAP expects a list of (samples, features) arrays for multi-class
    shap_list = [shap_vals_matrix[:, :, i] for i in range(len(class_names))]
    
    shap.summary_plot(
        shap_list, 
        features=X_bg, 
        feature_names=feature_cols, 
        class_names=class_names, 
        show=False,
        plot_type="bar"
    )
    plt.tight_layout()
    summary_path = os.path.join(out_dir, f"shap_summary_{model_name.lower()}.png")
    plt.savefig(summary_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    -> Saved SHAP global summary plot: {summary_path}")

    # 2. Local Explanation Plot using KernelExplainer (safer for multi-class waterfall)
    # Pick a specific interesting sample: row #100
    sample_idx = 100
    sample_feat = X_bg[sample_idx]
    
    # Predict the phase for this specific sample
    pred_prob = model.predict_proba(sample_feat.reshape(1, -1))[0]
    pred_class_idx = np.argmax(pred_prob)
    pred_class_name = class_names[pred_class_idx]
    
    # Because waterfall requires an Explanation object, we use KernelExplainer for this single sample
    prob_explainer = shap.KernelExplainer(model.predict_proba, X_bg[:50]) # very small bg for kernel speed
    shap_val_single = prob_explainer.shap_values(sample_feat, nsamples=500)
    
    # shap_val_single is a list of arrays (one per class). Get the array for the predicted class
    shap_for_pred = shap_val_single[pred_class_idx]
    expected_value = prob_explainer.expected_value[pred_class_idx]
    
    plt.figure(figsize=(8, 5))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_for_pred, 
            base_values=expected_value, 
            data=sample_feat, 
            feature_names=feature_cols
        ),
        max_display=8,
        show=False
    )
    plt.title(f"SHAP Waterfall (Sample #{sample_idx}) → Predicted: {pred_class_name}", pad=20)
    plt.tight_layout()
    waterfall_path = os.path.join(out_dir, f"shap_waterfall_{model_name.lower()}_sample{sample_idx}.png")
    plt.savefig(waterfall_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    -> Saved SHAP local waterfall plot: {waterfall_path}")


def generate_lime_explanation(model, X_scaled, feature_cols, out_dir, model_name):
    """
    Generates a LIME explanation for an interesting misclassified edge case or transition point.
    """
    print("  Generating LIME explanations...")
    
    # Initialize the explainer with a background dataset
    np.random.seed(42)
    idx_bg = np.random.choice(X_scaled.shape[0], min(500, X_scaled.shape[0]), replace=False)
    X_bg = X_scaled[idx_bg]
    
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_bg,
        feature_names=feature_cols,
        class_names=config.PHASE_LABELS,
        mode="classification",
        random_state=42
    )

    # Pick a random sample and explain it
    sample_idx = 150
    sample_feat = X_scaled[sample_idx]
    
    pred_prob = model.predict_proba(sample_feat.reshape(1, -1))[0]
    pred_class_idx = np.argmax(pred_prob)
    pred_class_name = config.PHASE_LABELS[pred_class_idx]
    
    # We want to explain why it predicted pred_class_idx
    exp = explainer.explain_instance(
        data_row=sample_feat, 
        predict_fn=model.predict_proba, 
        num_features=6,
        top_labels=1
    )
    
    # Save the LIME explanation to HTML
    html_path = os.path.join(out_dir, f"lime_explanation_{model_name.lower()}_sample{sample_idx}.html")
    exp.save_to_file(html_path)
    
    # Save a static plot of the LIME weights for the top predicted class
    fig = exp.as_pyplot_figure(label=pred_class_idx)
    plt.title(f"LIME Feature Weights → Predicted: {pred_class_name}", pad=15)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, f"lime_weights_{model_name.lower()}_sample{sample_idx}.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"    -> Saved LIME HTML report: {html_path}")
    print(f"    -> Saved LIME static plot: {plot_path}")


def main():
    utils.ensure_dirs()
    print("\n" + "="*55)
    print("  STEP 11 — Model Explainability (SHAP & LIME)")
    print("="*55)

    out_dir = os.path.join(config.PLOTS_DIR, "explainability")
    os.makedirs(out_dir, exist_ok=True)

    try:
        model, X_scaled, y_enc, feature_cols, X_raw, le, model_name = load_data_and_model()
    except FileNotFoundError as e:
        print(f"\n  [Error] {e}")
        return

    generate_shap_plots(model, X_scaled, feature_cols, out_dir, model_name)
    generate_lime_explanation(model, X_scaled, feature_cols, out_dir, model_name)
    
    print("\n✓ Explainability artifacts successfully generated.")

if __name__ == "__main__":
    main()
