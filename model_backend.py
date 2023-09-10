from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('mvp_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/predict", methods = ["POST"])
def predict():
    
   
    user_input = {}

    user_input['PYARDS'] = float(request.form['PYARDS'])
    user_input['TD'] = float(request.form['TD'])
    user_input['INT'] = float(request.form['INT'])
    user_input['CMP%'] = float(request.form['CMP%'])
    user_input['RYARD'] = float(request.form['RYARD'])
    user_input['RYA'] = float(request.form['RYA'])
    user_input['RTD'] = float(request.form['RTD'])
    user_input['REC'] = float(request.form['REC'])
    user_input['Y/A'] = float(request.form['Y/A'])
    user_input['YEAR'] = float(request.form['YEAR'])

    input_df = pd.DataFrame([user_input])

    prediction = model.predict(input_df)

    return render_template('index.html', prediction_text = "The Predicted MVP {}".format(prediction))


if __name__ == "__main__":
    app.run(debug = True, port = 8080)

    
