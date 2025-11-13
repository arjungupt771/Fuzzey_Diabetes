# PredDiabetes_Project — Copilot context

Purpose
- Prototype to predict three clinical states from tabular clinical features: `normal`, `prediabetic`, and `diabetic`.
- Combines a machine-learning classifier (RandomForest) with a small fuzzy risk score and an inductive conformal wrapper to provide interpretable, calibrated, and uncertainty-aware outputs.

Key design / novelty points
- 3-class labeling from fasting `Glucose` (normal <100, prediabetic 100–125, diabetic >=126).
- Fusion: model probabilities + fuzzy risk score -> fused risk in [0,1].
- Calibrated classifier + inductive conformal prediction for prediction sets and p-values.
- SHAP explainability and Streamlit demo for interactive inspection.

Top-level files and purpose
- `train_model.py` — quick training of RandomForest and save `model/model_rf.pkl`.
- `train_calibrated.py` — trains raw RF and calibrated classifier; saves `model/model_calibrated.pkl`, `model/conformal_calib.npy`, `model/model_metadata.pkl`.
- `predict.py` — load model, compute probabilities, fuzzy score and fused score for a single input dict/row.
- `conformal_predict.py` — helper to compute p-values and prediction sets using saved calibration scores.
- `evaluate_cv.py` — 5-fold CV evaluation; saves ROC/PR/calibration and SHAP summary to `Paper/figures`.
- `generate_results.py` — compares raw vs calibrated classifiers, computes Brier scores and metrics, writes `Paper/results_summary.md` and calibration comparison plots.
- `utils.py` — triangular / trapezoid membership and `fuzzy_risk_score`.
- `app/streamlit_app.py` — interactive demo UI.
- `Paper/` — methods draft, figures and results summary for conference inclusion.
- `tests/` — unit tests for fuzzy functions.

Quick environment / run commands (PowerShell)
```
cd C:\Users\shreyash\fuzzy_diabetes_v2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Train baseline model
python train_model.py --data "..\Fuzzy_Diabetes\Model\diabetes.csv" --out model/model_rf.pkl

# Train calibrated model + conformal calibration
python train_calibrated.py --data "..\Fuzzy_Diabetes\Model\diabetes.csv" --out model

# Evaluate cross-validated and generate figures
python evaluate_cv.py

# Generate comparison results and markdown summary
python generate_results.py

# Run unit tests
pytest -q

# Run the interactive demo
streamlit run app/streamlit_app.py
```

Important variables and artifacts
- Dataset expected: CSV with `Glucose` column. Example in parent workspace: `..\Fuzzy_Diabetes\Model\diabetes.csv`.
- Models and artifacts live under `model/`.
- Figures and paper artifacts under `Paper/` and `Paper/figures`.

Where to start for common tasks
- Add a new feature or change fuzzy rules: edit `utils.py` and add unit tests in `tests/`.
- Improve calibration/conformal: update `train_calibrated.py` and re-run `generate_results.py`.
- Add external validation: write a small script that loads another CSV, transforms it like `train_calibrated.py`, and runs `predict.py` / `conformal_predict.py`.

Suggested next items for paper (prioritized)
1. External/hold-out validation from another cohort to avoid optimistic bias.
2. Subgroup/performance fairness analysis (age/BMI/sex).
3. A short Jupyter notebook (`Paper/results_notebook.ipynb`) that embeds the saved figures and short captions for the paper.

Notes for Copilot or contributors
- Keep behavior stable: when adding features, preserve feature ordering or update `model/model_metadata.pkl` to include new `feature_names`.
- When changing labeling thresholds, update `make_labels_from_glucose` consistently in the scripts.
- Aim for small, well-tested changes: add unit tests for fuzzy rules and smoke tests for prediction.

Contact
- This file is a project-level context summary to help code completion and reviewers quickly understand the repo. Update it as the project evolves.
