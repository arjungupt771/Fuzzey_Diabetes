# Methods (Draft)

This document briefly describes the methods used in the PredDiabetes_Project prototype.

1. Dataset and labels
- We used the Pima-style dataset (`diabetes.csv`) with numeric features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.
- Labels: we create a 3-class label using fasting glucose thresholds: normal (<100 mg/dL), prediabetic (100--125 mg/dL), diabetic (>=126 mg/dL).

2. Models
- A RandomForest classifier (200 trees) is used as the base model.
- We train a calibrated classifier using `CalibratedClassifierCV` with sigmoid calibration to improve probability estimates.

3. Fuzzy risk score
- A lightweight fuzzy inference score is computed using triangular and trapezoidal membership functions over Glucose, BMI, and Age. Outputs are normalized to [0,1].

4. Fusion
- The final fused risk is a weighted combination of model probabilities and the fuzzy score. This provides both probabilistic predictions and human-readable fuzzy semantics.

5. Uncertainty (conformal prediction)
- We implement an inductive conformal predictor: after training, a calibration set is used to compute nonconformity scores (1 - probability assigned to true class). For a new sample, p-values for each candidate label are computed, and prediction sets are returned for a chosen significance level alpha.

6. Explainability
- SHAP (TreeExplainer) is used to produce feature attributions; the Streamlit demo displays SHAP summaries when available.

7. Evaluation
- We provide cross-validated ROC and PR curves per class, calibration plots, and confusion matrices. Figures are saved in `Paper/figures` for inclusion in papers.

Notes: This is a reproducible prototype intended for conference submissions. Next steps include external validation, more robust calibration/conformal baselines, and clinical utility analysis.
