import streamlit as st
import pickle
import numpy as np
import pandas as pd
import base64
import streamlit as st
import plotly.express as px


# Load the model
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)
    

def predict_default(age, smoking, yellow_fingers,anxiety, peer_pressure, chronic_disease, fatigue,
                    allergy, wheezing,alcohol_consuming, coughing, shortness_of_breath, swallowing_difficulty, chest_pain):
    
    # Making predictions
    prediction = model.predict([[age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue,
                                 allergy, wheezing,alcohol_consuming,  coughing, shortness_of_breath, swallowing_difficulty, chest_pain]])

    # Convert the prediction to a more understandable form
    if prediction == 1:  # Assuming 1 indicates presence of cancer
        pred = 'presence of lung cancer'
    else:
        pred = 'absence of lung cancer'
    
    return pred

st.title("Lung Cancer Prediction system")

with st.form("lung_cancer_prediction_form"):
    age = st.number_input("Age:", min_value=0)
    smoking = st.number_input("Smoking :", min_value=0, max_value=2, step=1)
    yellow_fingers = st.number_input("Yellow Fingers :", min_value=0, max_value=2, step=1)
    anxiety = st.number_input("anxiety :", min_value=0, max_value=2, step=1)
    peer_pressure = st.number_input("peer_pressure :", min_value=0, max_value=2, step=1)
    chronic_disease = st.number_input("Chronic Disease :", min_value=0, max_value=2, step=1)
    fatigue = st.number_input("Fatigue :", min_value=0, max_value=2, step=1)
    allergy = st.number_input("Allergy :", min_value=0, max_value=2, step=1)
    wheezing = st.number_input("Wheezing :", min_value=0, max_value=2, step=1)
    alcohol_consuming = st.number_input("alcohol consuming :", min_value=0, max_value=2, step=1)
    coughing = st.number_input("Coughing :", min_value=0, max_value=2, step=1)
    shortness_of_breath = st.number_input("Shortness of Breath :", min_value=0, max_value=2, step=1)
    swallowing_difficulty = st.number_input("swallowing_difficult :", min_value=0, max_value=2, step=1)
    chest_pain = st.number_input("Chest Pain :", min_value=0, max_value=2, step=1)

    submitted = st.form_submit_button("Predict")
    if submitted:
        prediction = predict_default(age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, allergy, wheezing, alcohol_consuming, coughing, shortness_of_breath, swallowing_difficulty, chest_pain)
        st.success('The result is: {}'.format(prediction))
