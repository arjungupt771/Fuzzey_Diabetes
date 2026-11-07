"""
Prediction utilities: load a saved model (joblib) and provide predict_input function
that returns class probabilities, fuzzy_score, fused_score and optional SHAP explanations.
"""
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any
from .utils import fuzzy_risk_assessment
from .ml_pipeline import LABELS
from .reliability import assess_ood



def load_model(path: str):
    data = joblib.load(path)
    model = data.get('model') if isinstance(data, dict) else data
    feature_names = data.get('feature_names', None) if isinstance(data, dict) else None
    return model, feature_names


def predict_input(model, feature_names, X: Dict[str, Any], label_names=None,
                  use_glucose_fuzzy=False, ood_detector=None):
    """X: dict of feature_name -> value
    Returns: dict with probabilities, fuzzy_score, fused_score, predicted_label
    """
    if feature_names is None:
        feature_names = list(X)
    missing = [name for name in feature_names if name not in X or X[name] is None]
    if missing:
        raise ValueError(f"Missing required model features: {', '.join(missing)}")
    try:
        row = {name: float(X[name]) for name in feature_names}
    except (TypeError, ValueError) as exc:
        raise ValueError("Model features must be numeric values") from exc
    if not np.isfinite(list(row.values())).all():
        raise ValueError("Model features must be finite; missing values are not accepted at prediction time")
    df = pd.DataFrame([row], columns=feature_names)
    ood_assessment = assess_ood(ood_detector, df)

    proba = None
    try:
        proba = model.predict_proba(df)[0]
    except Exception:
        # some sklearn pipelines may not implement predict_proba
        preds = model.predict(df)
        classes = np.asarray(getattr(model, 'classes_', [0, 1, 2]))
        proba = np.zeros(len(classes))
        proba[np.where(classes == preds[0])[0][0]] = 1.0

    classes = list(getattr(model, 'classes_', range(len(proba))))
    proba = np.asarray(proba)
    label_names = LABELS if label_names is None else label_names
    proba_dict = {label_names.get(int(label), str(label)): float(value)
                  for label, value in zip(classes, proba)}

    # The fuzzy branch excludes Glucose when it defines the target.
    glucose = float(X['Glucose']) if use_glucose_fuzzy and X.get('Glucose') is not None else None
    bmi = float(X['BMI']) if X.get('BMI') is not None else None
    age = float(X['Age']) if X.get('Age') is not None else None

    fuzzy_assessment = fuzzy_risk_assessment(
        glucose=glucose, bmi=bmi, age=age, family_history=X.get('FamilyHistory')
    )
    fuzzy = fuzzy_assessment['score']

    # Fuse only when target names have the expected ordinal clinical meaning.
    normalized_names = {name.lower().replace(' ', '-') for name in proba_dict}
    if {'normal', 'diabetic'}.issubset(normalized_names) and normalized_names.intersection({'prediabetic', 'pre-diabetic'}):
        diabetic_probability = next(value for name, value in proba_dict.items() if name.lower().replace(' ', '-') == 'diabetic')
        prediabetic_probability = next(value for name, value in proba_dict.items()
                                      if name.lower().replace(' ', '-') in {'prediabetic', 'pre-diabetic'})
        fused_score = max(0.0, min(1.0, 0.6 * diabetic_probability + 0.3 * prediabetic_probability + 0.1 * fuzzy))
    else:
        fused_score = None

    predicted_class = int(classes[int(np.argmax(proba))])
    pred_label = label_names.get(predicted_class, str(predicted_class))
    sensitivity = {}
    if hasattr(model, 'named_steps') and 'imputer' in model.named_steps:
        baseline_values = model.named_steps['imputer'].statistics_
        target_index = int(np.argmax(proba))
        for index, name in enumerate(feature_names):
            perturbed = df.copy()
            perturbed.loc[0, name] = baseline_values[index]
            changed = np.asarray(model.predict_proba(perturbed)[0])
            sensitivity[name] = float(proba[target_index] - changed[target_index])

    return {
        'probabilities': proba_dict,
        'fuzzy_score': float(fuzzy),
        'fuzzy_assessment': fuzzy_assessment,
        'fused_score': float(fused_score),
        'predicted_label': pred_label,
        'feature_sensitivity': sensitivity,
        'ood_assessment': ood_assessment,
    }


# Optional: SHAP explanation helper (returns numpy array of shap values if available)
def explain_shap(model, X_df):
    try:
        import shap
    except Exception:
        return None
    try:
        if hasattr(model, 'named_steps') and 'model' in model.named_steps:
            estimator = model.named_steps['model']
            X_model = model.named_steps['imputer'].transform(X_df)
            if 'scaler' in model.named_steps:
                X_model = model.named_steps['scaler'].transform(X_model)
            explainer = shap.Explainer(estimator)
            vals = explainer(X_model)
        else:
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
