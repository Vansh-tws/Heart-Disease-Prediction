import streamlit as st
import pandas as pd
import pickle
import numpy as np


try:
    from pyngrok import ngrok
    public_url = ngrok.connect(port='8501')
    st.success(f"Public URL for Streamlit App: {public_url}")
except ImportError:
    st.warning("pyngrok not installed. Streamlit app will not be publicly accessible via ngrok.")
except Exception as e:
    st.error(f"Error starting ngrok tunnel: {e}. Please ensure pyngrok is installed and ngrok is properly configured.")


with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("columns.pkl", "rb") as f:
    original_columns = pickle.load(f)
print(f"Loaded model type: {type(model)}")
print(f"Model has predict_proba attribute: {hasattr(model, 'predict_proba')}")
print(f"Model probability attribute: {getattr(model, 'probability', 'N/A')}")


st.title("❤️ Heart Disease Prediction App")

st.markdown("""
This app predicts the **likelihood of heart disease** based on a few important health parameters.
""")


st.header("Patient Information")

age = st.slider("Age", 1, 100, 50)
sex_choice = st.selectbox("Sex", ["Male", "Female"])
chest_pain_type = st.selectbox("Chest Pain Type", ["ASY (Asymptomatic)", "ATA (Typical Angina)", "NAP (Non-Anginal Pain)", "TA (Typical Angina)"])
resting_bp = st.slider("Resting Blood Pressure (mm/Hg)", 80, 200, 120)
cholesterol = st.slider("Cholesterol (mg/dl)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1]) # 0=False, 1=True
max_hr = st.slider("Maximum Heart Rate Achieved", 60, 220, 150)
exercise_angina_choice = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
oldpeak = st.slider("Oldpeak (ST depression induced by exercise relative to rest)", 0.0, 6.2, 1.0)

input_data = {}
for col in original_columns:
    input_data[col] = 0

# Age
input_data['Age'] = age
# Sex
input_data['Sex_M'] = 1 if sex_choice == "Male" else 0
# Chest Pain Type
if "ATA" in chest_pain_type:
    input_data['ChestPainType_ATA'] = 1
elif "NAP" in chest_pain_type:
    input_data['ChestPainType_NAP'] = 1
elif "TA" in chest_pain_type:
    input_data['ChestPainType_TA'] = 1
# RestingBP
input_data['RestingBP'] = resting_bp
# Cholesterol
input_data['Cholesterol'] = cholesterol
# FastingBS
input_data['FastingBS'] = fasting_bs
# MaxHR
input_data['MaxHR'] = max_hr
# Exercise Angina
input_data['ExerciseAngina_Y'] = 1 if exercise_angina_choice == "Yes" else 0
# Oldpeak
input_data['Oldpeak'] = oldpeak


input_df = pd.DataFrame([input_data], columns=original_columns)


input_scaled = scaler.transform(input_df)

# Make prediction
prediction = model.predict(input_scaled)

# Get prediction probabilities
if hasattr(model, 'predict_proba'):
    prediction_proba = model.predict_proba(input_scaled)[:,1]
    prob_available = True
else:
    prediction_proba = np.array([0.0])
    prob_available = False
    st.warning("Probability estimation not available for the selected model. Displaying raw prediction.")

st.subheader("Prediction Result:")
if prediction[0] == 1:
    if prob_available:
        st.error(f"The model predicts **Heart Disease** with probability {prediction_proba[0]:.2f}")
    else:
        st.error(f"The model predicts **Heart Disease**.")
else:
    if prob_available:
        st.success(f"The model predicts **No Heart Disease** with probability {1-prediction_proba[0]:.2f}")
    else:
        st.success(f" The model predicts **No Heart Disease**.")
