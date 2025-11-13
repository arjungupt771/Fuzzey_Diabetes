import streamlit as st
import pandas as pd
import numpy as np
import os
from predict import load_model, predict_input, explain_shap

st.set_page_config(page_title='PredDiabetes Demo', layout='centered')
st.title('PredDiabetes: Predict Normal / Prediabetic / Diabetic')

st.markdown('This demo fuses a RandomForest classifier with a fuzzy risk score and shows probabilities and an interpretable fused risk.')

MODEL_PATH = os.path.join('..', 'model', 'model_rf.pkl')
if not os.path.exists(MODEL_PATH):
    st.warning('Model not found at model/model_rf.pkl. Please run `python train_model.py --data <path_to_csv>` first.')

model = None
feature_names = None
try:
    model, feature_names = load_model(MODEL_PATH)
except Exception:
    model = None

# Input widgets - common features in Pima-like datasets
cols = st.columns(2)
with cols[0]:
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
    glucose = st.number_input('Glucose (mg/dL)', min_value=0, max_value=400, value=120)
    blood_pressure = st.number_input('BloodPressure', min_value=0, max_value=150, value=70)
    skin = st.number_input('SkinThickness', min_value=0, max_value=100, value=20)
with cols[1]:
    insulin = st.number_input('Insulin', min_value=0, max_value=900, value=85)
    bmi = st.number_input('BMI', min_value=0.0, max_value=80.0, value=28.0)
    dpf = st.number_input('DiabetesPedigreeFunction', min_value=0.0, max_value=5.0, value=0.5)
    age = st.number_input('Age', min_value=0, max_value=120, value=45)

input_data = {
    'Pregnancies': pregnancies,
    'Glucose': glucose,
    'BloodPressure': blood_pressure,
    'SkinThickness': skin,
    'Insulin': insulin,
    'BMI': bmi,
    'DiabetesPedigreeFunction': dpf,
    'Age': age,
}

if st.button('Predict'):
    if model is None:
        st.error('Model not loaded. Train the model first.')
    else:
        out = predict_input(model, feature_names, input_data)
        st.subheader('Predicted probabilities')
        probs = out['probabilities']
        dfp = pd.DataFrame({'class': list(probs.keys()), 'probability': list(probs.values())})
        st.bar_chart(dfp.set_index('class'))

        st.write('Predicted label:', out['predicted_label'])
        st.write('Fuzzy risk score:', f"{out['fuzzy_score']:.3f}")
        st.write('Fused risk score (model+fuzzy):', f"{out['fused_score']:.3f}")

        # SHAP explanation if available
        try:
            exp = explain_shap(model, pd.DataFrame([input_data]))
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
st.markdown('Notes: This is a prototype. For a conference submission consider adding cross-validation, more datasets, and calibrated uncertainty (conformal prediction).')
