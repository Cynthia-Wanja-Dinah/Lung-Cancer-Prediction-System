from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

def predict_default(age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue,
                    allergy, wheezing, alcohol_consuming, coughing, shortness_of_breath, swallowing_difficulty, chest_pain):
    
    # Making predictions
    prediction = model.predict([[age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue,
                                 allergy, wheezing, alcohol_consuming, coughing, shortness_of_breath, swallowing_difficulty, chest_pain]])

    # Convert the prediction to a more understandable form
    if prediction == 1:  # Assuming 1 indicates presence of cancer
        pred = 'Presence of lung cancer : with an accuracy of 90%'
    else:
        pred = 'Absence of lung cancer with an accuracy of 87%'
    
    return pred

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        smoking = int(request.form['smoking'])
        yellow_fingers = int(request.form['yellow_fingers'])
        anxiety = int(request.form['anxiety'])
        peer_pressure = int(request.form['peer_pressure'])
        chronic_disease = int(request.form['chronic_disease'])
        fatigue = int(request.form['fatigue'])
        allergy = int(request.form['allergy'])
        wheezing = int(request.form['wheezing'])
        alcohol_consuming = int(request.form['alcohol_consuming'])
        coughing = int(request.form['coughing'])
        shortness_of_breath = int(request.form['shortness_of_breath'])
        swallowing_difficulty = int(request.form['swallowing_difficulty'])
        chest_pain = int(request.form['chest_pain'])

        # Get the prediction
        prediction = predict_default(age, smoking, yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, allergy, wheezing, alcohol_consuming, coughing, shortness_of_breath, swallowing_difficulty, chest_pain)

        return render_template('result.html', prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
