import streamlit as st
import joblib
import pandas as pd
import numpy as np
from src.data_prep import preprocess_pipeline

st.title("🫀 Heart Disease Prediction System")
st.markdown("Production-ready model using best Random Forest (F1: XX%)")

# Load best model
@st.cache_resource
def load_model():
    preprocessor, _, _ = preprocess_pipeline()
    model = joblib.load('../models/randomforest_best.pkl')
    return preprocessor, model

preprocessor, model = load_model()

# Input widgets (adapt to your features)
with st.sidebar:
    st.header("Patient Data")
    age = st.slider("Age", 20, 80, 50)
    sex = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x==0 else "Male")
    cp = st.selectbox("Chest Pain Type", [1,2,3,4])
    trestbps = st.slider("Resting BP", 90, 200, 120)
    chol = st.slider("Cholesterol", 100, 600, 250)
    fbs = st.selectbox("Fasting Blood Sugar >120", [0,1])
    restecg = st.selectbox("Resting ECG", [0,1,2])
    thalach = st.slider("Max Heart Rate", 70, 220, 150)
    exang = st.selectbox("Exercise Angina", [0,1])
    oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = st.selectbox("ST Slope", [1,2,3])
    ca = st.selectbox("Major Vessels", [0,1,2,3])
    thal = st.selectbox("Thalassemia", [3,6,7])

if st.button("🔮 Predict Risk"):
    input_data = pd.DataFrame({
        'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps],
        'chol': [chol], 'fbs': [fbs], 'restecg': [restecg],
        'thalach': [thalach], 'exang': [exang], 'oldpeak': [oldpeak],
        'slope': [slope], 'ca': [ca], 'thal': [thal]
    })
    
    pred = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0, 1]
    
    st.success("✅ Prediction Complete!")
    st.metric("Risk Level", "HIGH RISK" if pred==1 else "LOW RISK", delta=f"{prob:.1%}")
    st.progress(prob)
