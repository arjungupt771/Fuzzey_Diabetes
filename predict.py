"""
Prediction utilities: load a saved model (joblib) and provide predict_input function
that returns class probabilities, fuzzy_score, fused_score and optional SHAP explanations.
"""
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any
from utils import fuzzy_risk_score

LABELS = {0: 'normal', 1: 'prediabetic', 2: 'diabetic'}


def load_model(path: str):
    data = joblib.load(path)
    model = data.get('model') if isinstance(data, dict) else data
    feature_names = data.get('feature_names', None) if isinstance(data, dict) else None
    return model, feature_names


def predict_input(model, feature_names, X: Dict[str, Any]):
    """X: dict of feature_name -> value
    Returns: dict with probabilities, fuzzy_score, fused_score, predicted_label
    """
    # Build DataFrame row ensuring feature order
    if feature_names is None:
        # infer from dict
        df = pd.DataFrame([X])
    else:
        # ensure columns exist
        row = {f: float(X.get(f, 0.0)) for f in feature_names}
        df = pd.DataFrame([row], columns=feature_names)

    # fill nan
    df = df.fillna(df.median())

    proba = None
    try:
        proba = model.predict_proba(df)[0]
    except Exception:
        # some sklearn pipelines may not implement predict_proba
        preds = model.predict(df)
        proba = np.zeros((len(preds), 3))[0]
        proba[int(preds[0])] = 1.0

    # map probabilities to names (ensure length 3)
    proba = np.asarray(proba)
    if proba.shape[0] < 3:
        # pad
        p = np.zeros(3)
        p[:proba.shape[0]] = proba
        proba = p

    proba_dict = {LABELS[i]: float(proba[i]) for i in range(3)}

    # compute fuzzy score using glucose, bmi, age if present
    glucose = float(X.get('Glucose', df['Glucose'].iloc[0]) if 'Glucose' in df.columns else 0.0)
    bmi = float(X.get('BMI', df['BMI'].iloc[0]) if 'BMI' in df.columns else None)
    age = float(X.get('Age', df['Age'].iloc[0]) if 'Age' in df.columns else None)

    fuzzy = fuzzy_risk_score(glucose, bmi=bmi, age=age)

    # fuse: weighted combination of model probabilities and fuzzy score
    fused_score = 0.6 * proba_dict['diabetic'] + 0.3 * proba_dict['prediabetic'] + 0.1 * fuzzy

    # normalize fused score to [0,1] (should be already)
    fused_score = max(0.0, min(1.0, fused_score))

    pred_label = LABELS[int(np.argmax(proba))]

    return {
        'probabilities': proba_dict,
        'fuzzy_score': float(fuzzy),
        'fused_score': float(fused_score),
        'predicted_label': pred_label
    }


# Optional: SHAP explanation helper (returns numpy array of shap values if available)
def explain_shap(model, X_df):
    try:
        import shap
    except Exception:
        return None
    try:
        explainer = shap.Explainer(model)
        vals = explainer(X_df)
        return vals
    except Exception:
        return None


if __name__ == '__main__':
    # quick smoke demo - expects model at model/model_rf.pkl
    try:
        m, features = load_model('model/model_rf.pkl')
        sample = {f: 0 for f in (features or ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])}
        sample['Glucose'] = 120
        sample['BMI'] = 30
        print(predict_input(m, features, sample))
    except Exception as e:
        print('Model load/predict failed:', e)
