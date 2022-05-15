from flask import Flask, request, jsonify, make_response
from flask_sqlalchemy import SQLAlchemy
from dotenv import load_dotenv
from functools import wraps
import numpy as np
import datetime
import pickle
import uuid
import os

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Ina work"

# @app.route('/predic', methods=['POST'])
# def predict():
#     data = request.get_json()
#     print(data)

if __name__ == '__main__':
    app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)
