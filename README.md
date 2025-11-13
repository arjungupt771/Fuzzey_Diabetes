# PredDiabetes_Project

This is a prototype inspired by the original `Fuzzy_Diabetes` repository. It extends the idea to predict both prediabetes and diabetes (three-class prediction) and adds an explainable fusion novelty: combining a classic ML classifier (RandomForest) with a lightweight fuzzy inference risk score and SHAP-based explanations. This produces both probabilistic predictions and human-readable fuzzy risk outputs — useful for conference submissions.

Project highlights / novelty
- 3-class label creation from a standard diabetes dataset using ADA glucose thresholds: normal / prediabetic / diabetic.
- Ensemble fusion: RandomForest probabilities fused with a fuzzy risk score to provide a single, interpretable risk measure.
- SHAP explanations integrated into the demo to help explain which features drive predictions.
- Streamlit-based interactive demo for quick visualization, plus unit tests.

Quick start
1. Install dependencies (use a virtualenv):

```powershell
cd C:\Users\shreyash\fuzzy_diabetes_v2
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Train a model (point to your dataset, e.g., the `diabetes.csv` in the original repo):

```powershell
python train_model.py --data "..\Fuzzy_Diabetes\Model\diabetes.csv" --out model/model_rf.pkl
```

3. Start the Streamlit demo:

```powershell
streamlit run app/streamlit_app.py
```

Files
- `train_model.py`: script to build 3-class labels and train a RandomForest model.
- `predict.py`: load model, compute class probabilities, fuzzy risk score, and fused risk.
- `app/streamlit_app.py`: interactive demo.
- `utils.py`: fuzzy membership helpers.
- `requirements.txt`: required packages.
- `tests/test_utils.py`: small unit tests for the fuzzy utilities.

Notes
- The train script forms prediabetes labels from `Glucose` using thresholds 100-125 mg/dL (prediabetic) and >=126 mg/dL (diabetic). If your dataset has A1C, you can adapt thresholds accordingly.
- This is a prototype: for conference submission, consider adding cross-site validation, conformal prediction for calibrated uncertainties, and an expanded explanation section.

Contact
- Created as a prototype. I can add a paper-ready README, figure generation, and an extended evaluation pipeline on request.
