from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open("Students_mark_predictor_model.pkl", "rb"))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    
    input_features = [float(x) for x in request.form.values()]
    value = np.array(input_features)
    
    if input_features[0] <0 or input_features[0] >24:
        return render_template('index.html', prediction_text='Please enter valid hours between 1 to 24 if you live on the Earth')
    
    output = model.predict([value])[0][0].round(2)
    return render_template('index.html', Prediction_text = f"you will get {output}% marks, when you do study {input_features} hours per day")

if __name__ == '__main__':
    app.debug = True
    app.run()
