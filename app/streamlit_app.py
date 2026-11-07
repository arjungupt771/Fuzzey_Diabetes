import streamlit as st
import pandas as pd
import os
import joblib
from pathlib import Path
from datetime import datetime
from diabetes_risk.predict import load_model, predict_input, explain_shap
from diabetes_risk.conformal_predict import load_conformal, predict_set

st.set_page_config(page_title='PredDiabetes Demo', layout='centered')
st.title('Research diabetes risk assessment prototype')

st.markdown('This is a research and risk-assessment prototype, not a diagnostic tool or substitute for professional medical evaluation. Glucose is excluded from both prediction branches when it defines the training target; models trained on independent labels may use it.')

MODEL_DIR = Path(__file__).resolve().parents[1] / 'model'
available_models = [path for path in [MODEL_DIR / 'model_best_cv.pkl', MODEL_DIR / 'model_calibrated.pkl', MODEL_DIR / 'model_rf.pkl'] if path.exists()]
if len(available_models) > 1:
    selected_path = st.selectbox('Classifier artifact', available_models,
                                 format_func=lambda path: {
                                     'model_best_cv.pkl': 'Cross-validation selected',
                                     'model_calibrated.pkl': 'Calibrated + conformal',
                                     'model_rf.pkl': 'Random Forest baseline',
                                 }.get(path.name, path.name))
else:
    selected_path = available_models[0] if available_models else MODEL_DIR / 'model_rf.pkl'
MODEL_PATH = selected_path
if not os.path.exists(MODEL_PATH):
    st.warning(f'Model not found at {MODEL_PATH}. Train it with `python -m scripts.train_model --data <path_to_csv> --out model/model_rf.pkl`.')

model = None
feature_names = None
feature_defaults = {}
model_artifact = {}
label_names = {0: 'normal', 1: 'prediabetic', 2: 'diabetic'}
fuzzy_uses_glucose = False
ood_detector = None
try:
    model, feature_names = load_model(MODEL_PATH)
    artifact = joblib.load(MODEL_PATH)
    if isinstance(artifact, dict):
        model_artifact = artifact
        feature_defaults = artifact.get('feature_defaults', {})
        label_names = artifact.get('label_names', label_names)
        fuzzy_uses_glucose = artifact.get('fuzzy_uses_glucose', False)
        ood_detector = artifact.get('ood_detector')
except Exception:
    model = None
if feature_names and 'Glucose' in feature_names and not fuzzy_uses_glucose:
    st.error('This saved model uses Glucose as a predictor. Retrain it with the current leakage-safe pipeline before using this app.')
    model = None
if model is not None and feature_names and any(name not in feature_defaults for name in feature_names):
    st.error('This model artifact lacks training-based input defaults. Retrain it with the current pipeline before using this app.')
    model = None
if MODEL_PATH.name == 'model_best_cv.pkl':
    st.caption('Using the model selected by stratified cross-validation.')
elif MODEL_PATH.name == 'model_calibrated.pkl':
    st.caption('Using the calibrated classifier matched to the conformal prediction artifacts.')

conformal = None
try:
    if MODEL_PATH.name == 'model_calibrated.pkl' and (MODEL_DIR / 'model_metadata.pkl').exists():
        conformal = load_conformal(str(MODEL_DIR / 'model_metadata.pkl'))
except Exception as exc:
    st.warning(f'Conformal prediction artifacts could not be loaded: {exc}')

st.subheader('Assessment inputs')
input_data = {}
input_features = feature_names or ['Pregnancies', 'BloodPressure', 'SkinThickness', 'Insulin',
                                  'BMI', 'DiabetesPedigreeFunction', 'Age']
if 'Glucose' not in input_features:
    st.number_input('Glucose (mg/dL, reference only)', min_value=0, max_value=400, value=120)
cols = st.columns(2)
for index, name in enumerate(input_features):
    default = float(feature_defaults.get(name, {'Pregnancies': 0, 'BloodPressure': 70,
                                                'SkinThickness': 20, 'Insulin': 85,
                                                'BMI': 28, 'DiabetesPedigreeFunction': 0.5,
                                                'Age': 45}.get(name, 0)))
    with cols[index % 2]:
        input_data[name] = st.number_input(name, value=default, key=f'feature_{name}')

for name, default in [('BMI', 28.0), ('Age', 45.0)]:
    if name not in input_data:
        input_data[name] = st.number_input(f'{name} (for fuzzy assessment)', value=default, key=f'fuzzy_{name}')
if 'FamilyHistory' not in input_data:
    history_choice = st.selectbox('Family history of diabetes', ['No', 'Yes'])
    input_data['FamilyHistory'] = {'No': 0, 'Yes': 1}[history_choice]

if st.button('Predict'):
    if model is None:
        st.error('Model not loaded. Train the model first.')
    else:
        out = predict_input(model, feature_names, input_data, label_names, fuzzy_uses_glucose, ood_detector)
        st.subheader('Predicted probabilities')
        probs = out['probabilities']
        dfp = pd.DataFrame({'class': list(probs.keys()), 'probability': list(probs.values())})
        st.bar_chart(dfp.set_index('class'))

        st.write('Predicted label:', out['predicted_label'])
        if out['ood_assessment'] is not None:
            if out['ood_assessment']['is_unusual']:
                st.warning('This input is unusual compared with the training data. Treat the output as unreliable.')
            else:
                st.caption('Input-distribution check did not flag an unusual case; this is not a guarantee of reliability.')
        if out['feature_sensitivity']:
            st.subheader('Local model sensitivity')
            sensitivity_table = pd.DataFrame([
                {'feature': name, 'change in predicted-class probability': value}
                for name, value in out['feature_sensitivity'].items()
            ]).sort_values('change in predicted-class probability', key=lambda values: values.abs(), ascending=False)
            st.bar_chart(sensitivity_table.set_index('feature'))
            st.caption('Each value compares this input with that feature set to its training median, one feature at a time. This is a model sensitivity view, not a causal explanation.')
        st.write('Fuzzy risk score:', f"{out['fuzzy_score']:.3f}")
        if out['fused_score'] is not None:
            st.write('Fused risk score (model+fuzzy):', f"{out['fused_score']:.3f}")
        else:
            st.caption('A combined risk score is not defined for this target label set; review the class probabilities and fuzzy score separately.')
        fuzzy_details = out['fuzzy_assessment']
        st.caption(f"Fuzzy input ambiguity: {fuzzy_details['uncertainty']:.2f} (membership overlap; prototype measure)")
        st.write('Fuzzy rule strengths:', {
            f"{name.title()} risk": round(value, 3)
            for name, value in fuzzy_details['rule_strengths'].items()
        })
        if 'prediction_history' not in st.session_state:
            st.session_state['prediction_history'] = []
        st.session_state['prediction_history'].append({
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'most_likely_class': out['predicted_label'],
            'fused_risk': round(out['fused_score'], 3) if out['fused_score'] is not None else 'not defined',
        })

        prediction_set = None
        if conformal is not None:
            model_row = pd.DataFrame([{name: input_data[name] for name in feature_names}])
            conformal_result = predict_set(conformal, model_row, alpha=0.1)
            prediction_set = conformal_result['prediction_sets'][0]
            readable_set = [label_names.get(label, str(label)) for label in prediction_set]
            st.subheader('Conformal uncertainty (90% target coverage)')
            if len(readable_set) == 1:
                st.success(f"Prediction set: {readable_set[0]}")
            elif readable_set:
                st.warning(f"Abstain: ambiguous prediction set ({', '.join(readable_set)}). Seek professional evaluation.")
            else:
                st.warning('The conformal prediction set is empty; the model abstains. Seek professional evaluation.')
            st.caption('Coverage is a statistical target under exchangeability, not a clinical confidence guarantee.')

        try:
            from diabetes_risk.assessment_report import build_assessment_pdf
            pdf_bytes = build_assessment_pdf(out, prediction_set,
                                             model_artifact.get('target_definition'))
            st.download_button('Download assessment PDF', pdf_bytes,
                               file_name='research_risk_assessment.pdf',
                               mime='application/pdf')
        except ImportError:
            st.info('PDF reports are optional. Install reportlab from requirements-optional.txt to enable downloads.')

        if st.session_state['prediction_history']:
            st.subheader('This-session assessment history')
            st.dataframe(pd.DataFrame(st.session_state['prediction_history']), use_container_width=True)

        # SHAP explanation if available
        try:
            model_input = pd.DataFrame([{name: input_data[name] for name in feature_names}])
            exp = explain_shap(model, model_input)
            if exp is None:
                st.info('SHAP not available or failed to compute (optional).')
            else:
                st.subheader('SHAP explanation (if available)')
                # shap returns object with values and feature_names in newer versions
                try:
                    import shap
                    st.pyplot(shap.plots.bar(exp, show=False))
                except Exception:
                    st.write('Could not render SHAP plot in this environment.')
        except Exception:
            st.info('SHAP explanation skipped.')

st.markdown('---')
st.markdown(f"Target definition: {model_artifact.get('target_definition', 'glucose-threshold-derived research categories')}. Independent labels and external validation are required before interpreting results as screening performance.")
