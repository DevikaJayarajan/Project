from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load stress prediction model (scaler + model tuple)
with open('final.pkl', 'rb') as f:
    stress_scaler, stress_model = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')  # Landing page

@app.route('/predict-stress', methods=['GET', 'POST'])
def predict_stress():
    prediction = None
    if request.method == 'POST':
        inputs = [
            float(request.form['journal_sentiment']),
            float(request.form['body_survey_avg_severity']),
            float(request.form['energy_level']),
            float(request.form['sleep_hours']),
            float(request.form['hrv_ms']),
            float(request.form['steps']),
            float(request.form['nutrition_item_count'])
        ]
        input_array = np.array(inputs).reshape(1, -1)
        input_scaled = stress_scaler.transform(input_array)
        prediction = round(stress_model.predict(input_scaled)[0], 2)

    return render_template('stress_predict.html', prediction=prediction)

@app.route('/predict-hrv', methods=['GET', 'POST'])
def predict_hrv():
    prediction = None
    if request.method == 'POST':
        inputs = [
            float(request.form['journal_sentiment']),
            float(request.form['body_survey_avg_severity']),
            float(request.form['energy_level']),
            float(request.form['sleep_hours']),
            float(request.form['stress_level']),
            float(request.form['steps']),
            float(request.form['nutrition_item_count'])
        ]
        input_array = np.array(inputs).reshape(1, -1)

        # Load HRV model (replace with your actual pickle file)
        with open('linreg_HRVmodel.pkl', 'rb') as f:
            hrv_model = pickle.load(f)

        prediction = round(hrv_model.predict(input_array)[0], 2)

    return render_template('hrv_predict.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
