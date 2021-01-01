from flask import Flask, render_template, request
import jsonify
import requests
import pickle
import numpy as np
import sklearn
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open("student_mark_predictor.pkl", "rb"))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    global df
    
    if request.method == 'POST':

    input_features = [int(x) for x in request.form.values()]
    features_value = np.array(input_features)
    
    #validate input hours
    if input_features[0] <0 or input_features[0] >24:
        return render_template('index.html', prediction_text='Please enter valid hours between 1 to 24 if you live on the Earth')
        

    output = model.predict([features_value])[0][0].round(2)

    return render_template('index.html', prediction_text='You will get [{}%] marks, when you do study [{}] hours per day '.format(output, int(features_value[0])))

if __name__=="__main__":
    app.run(debug=True)
