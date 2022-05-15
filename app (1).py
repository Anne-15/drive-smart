from flask import Flask, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from functools import wraps
import numpy as np
import datetime
import pickle
import uuid
import os
load_dotenv()

app = Flask(__name__)


@app.route("/")
def hello_world():
    return "Ina work"


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(data)

    distance = data['distance']
    # time = data['time']

    print(distance)
    # print(time)

    model_path = 'model_saved'
    model = pickle.load(open(model_path, 'rb'))

    data = np.array([[distance]])

    prediction_array = model.predict(data)
    print(prediction_array)

    prediction = prediction_array[0]

    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)
