from flask import Flask, jsonify, request
from flask_cors import CORS

import joblib
import traceback
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)
CORS(app)
# salary_model = pickle.load("salary_model.pkl", 'rb')
lr = joblib.load("titanic_model.pkl")
model_columns = joblib.load("titanic_model_columns.pkl")

@app.route('/')
def home_route():
    return jsonify({"txt": "This is the home page of my flask ML apis."})

@app.route('/titanic_predict', methods=['POST'])
def titanic_predict():
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
        return jsonify({"text": "No model here to use"})

# @app.route('/salary_predict',methods=['POST'])
# def salary_predict():
#     '''
#     For rendering results on HTML GUI
#     '''
#     int_features = [int(x) for x in request.form.values()]
#     final_features = [np.array(int_features)]
#     prediction = salary_model.predict(final_features)

#     output = round(prediction[0], 2)

#     return jsonify(prediction_text='Employee Salary should be $ {}'.format(output))

if __name__ == '__main__':

    app.run()
