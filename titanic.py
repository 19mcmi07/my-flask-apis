import pandas as pd
import numpy as np

url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
df = pd.read_csv(url)
include = ['Age', 'Sex', 'Embarked', 'Survived']
df_ = df[include]

categoricals = []
for col, col_type in df_.dtypes.iteritems():
    if col_type == 'O':
        categoricals.append(col)
    else:
        df_[col].fillna(0, inplace=True)


df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

from sklearn.linear_model import LogisticRegression
dependent_variable = 'Survived'
x = df_ohe[df_ohe.columns.difference([dependent_variable])]
y = df_ohe[dependent_variable]
lr = LogisticRegression()
lr.fit(x,y)

import joblib
joblib.dump(lr, 'titanic_model.pkl')
print("Model Dumped!")

lr = joblib.load('titanic_model.pkl')

model_columns = list(x.columns)
joblib.dump(model_columns, 'titanic_model_columns.pkl')
print("Model columns dumped!")