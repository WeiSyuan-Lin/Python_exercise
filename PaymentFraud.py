# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 16:37:21 2022

@author: nightrain
"""

import pandas as pd
import numpy as np
#%%
path="D:\Python\datasets\PaymentFraud\credit card.csv"
#%%
data = pd.read_csv(path)
print(data.head())
#%%
print(data.isnull().sum())
#%%
"""
look at the type of transaction mentioned in the dataset:
"""
# Exploring transaction type
print(data.type.value_counts())
#%%
Type = data["type"].value_counts()
transactions = Type.index
quantity = Type.values

import plotly.express as px
figure = px.pie(data, 
             values=quantity, 
             names=transactions,hole = 0.5, 
             title="Distribution of Transaction Type")
figure.show()
#%%
"""
look at the correlation between 
the features of the data with the isFraud column:
"""
# Checking correlation
correlation = data.corr()
print(correlation["isFraud"].sort_values(ascending=False))
#%%
"""
let’s transform the categorical features into numerical.
Here I will also transform 
the values of the isFraud column into No Fraud and Fraud labels 
to have a better understanding of the output:
"""
data["type"] = data["type"].map({"CASH_OUT": 1, "PAYMENT": 2, 
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
print(data.head())
#%%
"""
Online Payments Fraud Detection Model
"""
# splitting the data
from sklearn.model_selection import train_test_split
x = np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y = np.array(data[["isFraud"]])
#%%
# training a machine learning model
from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))
#%%
# prediction
#features = [type, amount, oldbalanceOrg, newbalanceOrig]
features = np.array([[4, 9000.60, 0.0, 9000.60]])
print(model.predict(features))






