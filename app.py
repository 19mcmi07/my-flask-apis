from flask import Flask, jsonify, request
from flask_cors import CORS

import joblib
import traceback
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/')
def home_route():
    return jsonify({"txt": "This is the home page of my flask ML apis."})

@app.route('/predict', methods=['POST'])
def predict():
    if lr:
        try:
            json_ = request.json
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(columns=model_columns, fill_value=0)

            prediction = list(lr.predict(query))

            return jsonify({'prediction': str(prediction)})

        except:
            return jsonify({'trace': traceback.format_exc()})
    else:
        return ('No model here to use')

if __name__ == '__main__':

    lr = joblib.load("titanic_model.pkl")
    model_columns = joblib.load("titanic_model_columns.pkl")

    app.run()
