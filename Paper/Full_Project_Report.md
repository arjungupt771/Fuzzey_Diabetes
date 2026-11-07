<!--- Auto-generated comprehensive project report for PredDiabetes_Project -->
# PredDiabetes_Project — Full Project Report

Date: 2025-11-09

This document collects all relevant project information useful both for turning this work into a short conference paper and for presenting the repository as a project demonstration. It synthesizes the design, methods, implementation, artifacts, reproduction steps, results summary, limitations, and next steps.

## Abstract

PredDiabetes_Project is a prototype system for predicting three clinical states from tabular clinical features: `normal`, `prediabetic`, and `diabetic`. The approach fuses a RandomForest classifier with a small fuzzy risk score and wraps predictions with inductive conformal calibration to provide interpretable, calibrated, and uncertainty-aware outputs. SHAP is used to provide feature-level explanations. The repository contains training scripts, evaluation utilities, a Streamlit demo, and artifacts (models, calibration scores, and figures) intended for quick reproduction or as a draft for conference submission.

Keywords: prediabetes, RandomForest, fuzzy inference, conformal prediction, SHAP, explainability, calibration

## Project summary and novelty

- Three-class labeling from fasting Glucose using ADA-like thresholds: normal (<100 mg/dL), prediabetic (100–125 mg/dL), diabetic (>=126 mg/dL).
- Fusion: probabilistic outputs from a calibrated RandomForest classifier are fused with a small, transparent fuzzy risk score (glucose-centric) to produce a single fused risk in [0,1].
- Inductive conformal prediction provides p-values and prediction sets, enabling well-calibrated uncertainty quantification.
- SHAP explanations allow feature-level interpretation, and a Streamlit demo provides an interactive front-end for inspection.

The current project layout and quick-start guide are summarized in `README.md`. Reusable code lives in `diabetes_risk/`; command-line workflows live in `scripts/`.

- `scripts/train_model.py` — quick training of an RF model (3-class labels).
- `scripts/train_calibrated.py` — trains calibrated models, computes conformal calibration scores, and writes artifacts to `model/`.
- `diabetes_risk/predict.py` — loads saved models and runs a single-row prediction.
- `diabetes_risk/conformal_predict.py` — computes p-values and prediction sets from saved calibration scores.
- `scripts/evaluate_cv.py` / `scripts/generate_results.py` — evaluation and figure generation workflows.
- `diabetes_risk/utils.py` — fuzzy membership functions and risk scoring implementation.
- `app/streamlit_app.py` — demo UI.

## Data and labels

- Expected input: a tabular CSV dataset containing a `Glucose` column and other numeric features common to Pima-style diabetes datasets: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`.
- Label creation: `make_labels_from_glucose` in `diabetes_risk/ml_pipeline.py` maps glucose to {0=normal, 1=prediabetic, 2=diabetic} using thresholds as described above. The training workflow selects numeric features and excludes the target-defining column.

## Methods (concise)

1. Preprocessing and splits
   - Numeric-only columns are used; missing values are filled with column medians.
   - Data splits for the calibrated training script are: 60% training, 20% calibration, 20% test (achieved via two `train_test_split` calls).

2. Base classifier
   - RandomForestClassifier (n_estimators=200, random_state=42) as the baseline model.

3. Probability calibration
   - `CalibratedClassifierCV` with method='sigmoid' and cv=5 is used to produce calibrated probabilities.

4. Inductive conformal prediction
   - Nonconformity score: 1 - p_true where p_true is calibrated probability assigned to the true class on the calibration set.
   - Calibration nonconformity scores are saved to `model/conformal_calib.npy` for later p-value computation.

5. Fuzzy risk score
   - Implemented in `diabetes_risk/utils.py` using triangular and trapezoid membership functions. Glucose is omitted from the fuzzy branch when it defines the target.
   - The `fuzzy_risk_score` returns a normalized value in [0,1] with glucose weighted most heavily.

6. Fusion
   - Fused risk = weighted combination of model probability-derived risk and the fuzzy score (the fusion logic is in `diabetes_risk/predict.py`).

7. Explainability
   - SHAP (TreeExplainer) is used to compute feature importances and per-example explanations. The Streamlit demo attempts to render a SHAP bar plot when running locally.

## Implementation details and artifacts

- Requirements: see `requirements.txt` (pandas, numpy, scikit-learn, joblib, streamlit, shap, matplotlib, seaborn, pytest).
- Main artifacts produced by `scripts/train_calibrated.py` and saved under `model/`:
  - `model_rf_raw.pkl` — raw RandomForest (pickled with joblib)
  - `model_calibrated.pkl` — CalibratedClassifierCV object
  - `conformal_calib.npy` — array of nonconformity scores for conformal calibration
  - `feature_names.txt` and `model_metadata.pkl`

Paths in this repository (important files):

- `README.md` — quick start and project highlights.
- `scripts/train_calibrated.py` — training, calibration, and conformal score generation.
- `diabetes_risk/utils.py` — fuzzy helpers: `triangular`, `trapezoid`, and fuzzy risk scoring.
- `app/streamlit_app.py` — interactive demo (uses package prediction helpers).
- `Paper/results_summary.md` — existing result tables and calibration figures references.

## Results (repo-provided summary)

The repository includes `Paper/results_summary.md` with an example comparison table between raw and calibrated models. At the time of the summary, metrics shown were (example values taken from the file):

| model | accuracy | precision_macro | recall_macro | f1_macro | brier |
| - | -: | -: | -: | -: | -: |
| raw | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.022355 |
| calibrated | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.003215 |

Notes:
- The results in `Paper/results_summary.md` appear to show perfect classification (accuracy=1.0) — this strongly suggests either a toy dataset, data leakage, or a very small/test dataset; exercise caution before claiming clinical performance. Conduct external validation and audit data splits to rule out leakage.
- Calibration plots for each class are referenced in `Paper/figures` (e.g., `figures/calibration_compare_class_0.png`). Include these in a paper by exporting the actual figure files.

## How to reproduce (short)

The repository includes PowerShell-friendly quick steps in `README.md`. A minimal reproduction sequence (PowerShell) is:

```powershell
cd C:\Users\shreyash\PredDiabet\PredDiabetes_Project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Train calibrated model and produce conformal calibration
python -m scripts.train_calibrated --data "..\Fuzzy_Diabetes\Model\diabetes.csv" --out model

# Evaluate or generate results & figures
python -m scripts.evaluate_cv --data "..\Fuzzy_Diabetes\Model\diabetes.csv"
python -m scripts.generate_results --data "..\Fuzzy_Diabetes\Model\diabetes.csv"

# Run demo
streamlit run app/streamlit_app.py
```

Important reproducibility notes:
- Random seed(s): training uses `random_state=42` for reproducibility in `scripts/train_calibrated.py`.
- Splitting strategy: two-layer `train_test_split` is used to allocate train/calibration/test sets (60/20/20).
- Make sure the dataset used for training has the `Glucose` column; label creation depends on this.

## Testing and quality gates

- Unit tests: the `tests/` directory contains tests for fuzzy utilities (`tests/test_utils.py`). Run tests with `pytest -q`.
- Suggested checks before paper submission:
  - Run cross-site external validation on an independent cohort.
  - Recompute calibration/conformal statistics on external data.
  - Verify no data leakage (feature engineering or label leakage).

## Demo

Use `app/streamlit_app.py` for an interactive demo. The Streamlit app expects a model at `model/model_rf.pkl` by default; update the path or run the training script to create the expected file.

## Limitations and important caveats

- The current prototype uses glucose thresholds for labeling; consider multi-modal labeling (A1C, OGTT) when available.
- Results in `Paper/results_summary.md` suggest perfect classification on the current dataset; this is often a sign of dataset issues or leakage — do not claim clinical readiness.
- Conformal prediction implementation is inductive and depends on a representative calibration set. If calibration set is not representative of deployment distribution, p-values/prediction sets will be miscalibrated.

## Next steps (for conference-ready paper)

1. External / hold-out validation on a separate cohort to estimate generalization and avoid optimistic bias.
2. Subgroup and fairness analysis (age, sex, BMI) to ensure equitable performance.
3. Detailed calibration / reliability diagrams and decision-curve analysis to examine clinical utility.
4. Expand the evaluation section with per-class ROC/PR curves, class-wise calibration tables, and confidence intervals (bootstrap or cross-site).
5. Create a short Jupyter notebook (`Paper/results_notebook.ipynb`) that embeds figures, summarizes methods, and produces a reproducible narrative for the paper.

## Contact and authorship

Creator: Shreyash Anand <anandshreyash747@gmail.com>

## Appendix: code pointers and artifact locations

- Training and calibration: `scripts/train_calibrated.py` (saves to `model/`)
- Fuzzy functions: `diabetes_risk/utils.py` — `triangular`, `trapezoid`, and fuzzy risk scoring
- Demo: `app/streamlit_app.py` (uses `diabetes_risk.predict` helpers)
- Results summary: `Paper/results_summary.md` and `Paper/figures`
- Requirements: `requirements.txt`

If you want, I can:
- (A) Add a short paper-style `paper.md` that condenses Abstract/Intro/Methods/Results/Discussion into a conference-ready ~2-page manuscript draft.
- (B) Insert the actual figures from `Paper/figures` into this report and produce a single PDF draft (requires matplotlib/LaTeX-free export).
- (C) Commit this new report to the local git repository and optionally push to a remote you provide.

---

Generated from repository materials including `README.md`, `Paper/Draft_Methods.md`, `Paper/results_summary.md`, `scripts/train_calibrated.py`, `diabetes_risk/utils.py`, and `app/streamlit_app.py`.
