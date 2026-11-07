"""Stateless FastAPI inference service for the research prototype."""

import logging
import math
import os
import secrets
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import joblib
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from .conformal_predict import load_conformal, predict_set
from .predict import load_model, predict_input
from .ml_pipeline import LABELS


logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO').upper())
logger = logging.getLogger('diabetes_risk_api')
MODEL_DIR = Path(os.getenv('MODEL_DIR', 'model'))
MODEL_PATH = Path(os.getenv('MODEL_PATH', str(MODEL_DIR / 'model_best_cv.pkl')))
if not MODEL_PATH.exists():
    MODEL_PATH = MODEL_DIR / 'model_calibrated.pkl'
if not MODEL_PATH.exists():
    MODEL_PATH = MODEL_DIR / 'model_rf.pkl'

app = FastAPI(
    title='Research Diabetes Risk API',
    description='Prototype inference API. Outputs are not diagnoses or medical advice.',
    version='0.1.0',
)
API_KEY = os.getenv('API_KEY')
model = None
feature_names = None
conformal = None
label_names = LABELS
fuzzy_uses_glucose = False
ood_detector = None
try:
    if MODEL_PATH.exists():
        model, feature_names = load_model(str(MODEL_PATH))
        artifact = joblib.load(MODEL_PATH)
        if isinstance(artifact, dict):
            label_names = artifact.get('label_names', label_names)
            fuzzy_uses_glucose = artifact.get('fuzzy_uses_glucose', False)
            ood_detector = artifact.get('ood_detector')
        if 'Glucose' in (feature_names or []) and not fuzzy_uses_glucose:
            raise ValueError('Saved model contains the target-defining Glucose feature; retrain before serving.')
        if MODEL_PATH.name == 'model_calibrated.pkl':
            metadata_path = MODEL_DIR / 'model_metadata.pkl'
            if metadata_path.exists():
                conformal = load_conformal(str(metadata_path))
        logger.info('Loaded model artifact %s', MODEL_PATH)
    else:
        logger.warning('No model artifact found under %s', MODEL_DIR)
except Exception:
    logger.exception('Model artifact failed to load')


class PredictionRequest(BaseModel):
    features: Dict[str, float]
    alpha: float = Field(0.1, gt=0.0, lt=1.0)


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    """Enforce a configured shared API key; local development can omit API_KEY."""
    if API_KEY and (x_api_key is None or not secrets.compare_digest(x_api_key, API_KEY)):
        raise HTTPException(status_code=401, detail='Valid API key required.')


@app.get('/health')
def health():
    return {
        'status': 'ready' if model is not None else 'model_unavailable',
        'model_artifact': str(MODEL_PATH),
        'conformal_available': conformal is not None,
    }


@app.post('/predict', dependencies=[Depends(require_api_key)])
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail='A leakage-safe model artifact is not available.')
    if not request.features:
        raise HTTPException(status_code=422, detail='At least one feature value is required.')
    if feature_names:
        missing = [name for name in feature_names if name not in request.features]
        if missing:
            raise HTTPException(status_code=422, detail=f'Missing required model features: {", ".join(missing)}')
    if any(not math.isfinite(value) for value in request.features.values()):
        raise HTTPException(status_code=422, detail='Feature values must be finite numbers.')
    try:
        result = predict_input(model, feature_names, request.features, label_names,
                               fuzzy_uses_glucose, ood_detector)
        if conformal is not None:
            row = pd.DataFrame([{name: request.features[name] for name in feature_names}])
            interval = predict_set(conformal, row, alpha=request.alpha)
            result['conformal'] = {
                'alpha': request.alpha,
                'classes': [label_names.get(int(label), str(label)) for label in conformal['classes']],
                'pvalues': interval['pvalues'][0].tolist(),
                'prediction_set': [label_names.get(int(label), str(label))
                                   for label in interval['prediction_sets'][0]],
            }
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception('Prediction failed')
        raise HTTPException(status_code=500, detail='Prediction could not be computed.') from exc
