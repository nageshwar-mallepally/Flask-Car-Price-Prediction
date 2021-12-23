# Simple Linear Regression

'''
This model predicts the salary of the employ based on experience using simple linear regression model.
'''

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import requests
import json

def trim(x):
    if x.dtype == object:
        x = x.str.split(' ').str[0]
    return(x)

# Importing the dataset
df = pd.read_csv('car_details.csv');
dfn = df.drop(['name'], axis = 1)
dfn['current_year'] = 2021
dfn['no_year'] = dfn['current_year']-dfn['year']
dfny = dfn.drop(['year'], axis=1)
df1 = dfny.drop(['current_year', 'torque'], axis=1)
df1['engine'] = df['engine'].str.replace(' CC', '')
df1['mileage'] = df['mileage'].str.replace(' kmpl', '')
df1['max_power'] = df['max_power'].str.replace(' bhp', '')
df1 = df1.apply(trim)
df1['max_power'] = pd.to_numeric(df1['max_power'],errors = 'coerce')
df1['engine'] = pd.to_numeric(df1['engine'],errors = 'coerce')
df1['mileage'] = pd.to_numeric(df1['mileage'],errors = 'coerce')
df1['seats'] = pd.to_numeric(df1['seats'],errors = 'coerce')
df1['no_year'] = pd.to_numeric(df1['no_year'],errors = 'coerce')
df2 = pd.get_dummies(df1,drop_first=True)
df2['mileage'].fillna(value=df2['mileage'].mean(), inplace=True)
df2['engine'].fillna(value=df2['engine'].mean(), inplace=True)
df2['max_power'].fillna(value=df2['max_power'].mean(), inplace=True)
df2['seats'].fillna(value=df2['seats'].mean(), inplace=True)

x = df2.iloc[:,1:]
y = df2.iloc[:,0]
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(x,y)

preds = model.predict(x)
pd1 = pd.DataFrame(preds)
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
print('MAE: ', MAE(y, preds))

# Saving model using pickle
file = open("model.pkl","wb+")
pickle.dump(model, file)
file.close()

model = pickle.load(open('model.pkl','rb+'))
print(model.predict([[4500, 20, 1200, 120, 5, 5, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0]]))
