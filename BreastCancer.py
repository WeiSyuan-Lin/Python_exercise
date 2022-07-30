# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 19:20:28 2022

@author: nightrain
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
pio.renderers.default='jpg'

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
#%%
path="D:\Python\datasets\BreastCancer\\BRCA.csv"
#%%
data = pd.read_csv(path)
print(data.head())
#%%
print(data.isnull().sum())
#%%
"""
So this dataset has some null values in each column, 
drop these null values:
"""
data = data.dropna()
#%%
"""
look at the Gender column to see how many females and males are there:
"""
print(data.Gender.value_counts())
#%%
"""
look at the stage of tumour of the patients:
"""
# Tumour Stage
stage = data["Tumour_Stage"].value_counts()
transactions = stage.index
quantity = stage.values
#%%
figure = px.pie(data, 
             values=quantity, 
             names=transactions,hole = 0.5, 
             title="Tumour Stages of Patients")
figure.show()
#%%
# Histology
histology = data["Histology"].value_counts()
transactions = histology.index
quantity = histology.values
figure = px.pie(data, 
             values=quantity, 
             names=transactions,hole = 0.5, 
             title="Histology of Patients")
figure.show()
#%%
# ER status
print(data["ER status"].value_counts())
# PR status
print(data["PR status"].value_counts())
# HER2 status
print(data["HER2 status"].value_counts())
#%%
# Surgery_type
surgery = data["Surgery_type"].value_counts()
transactions = surgery.index
quantity = surgery.values
figure = px.pie(data, 
             values=quantity, 
             names=transactions,hole = 0.5, 
             title="Type of Surgery of Patients")
figure.show()
#%%
"""
To use this data to train a machine learning model, 
we need to transform the values of all the categorical columns. 
Here is how we can transform values of the categorical features
"""
data["Tumour_Stage"] = data["Tumour_Stage"].map({"I": 1, "II": 2, "III": 3})
data["Histology"] = data["Histology"].map({"Infiltrating Ductal Carcinoma": 1, 
                                           "Infiltrating Lobular Carcinoma": 2, "Mucinous Carcinoma": 3})
data["ER status"] = data["ER status"].map({"Positive": 1})
data["PR status"] = data["PR status"].map({"Positive": 1})
data["HER2 status"] = data["HER2 status"].map({"Positive": 1, "Negative": 2})
data["Gender"] = data["Gender"].map({"MALE": 0, "FEMALE": 1})
data["Surgery_type"] = data["Surgery_type"].map({"Other": 1, "Modified Radical Mastectomy": 2, 
                                                 "Lumpectomy": 3, "Simple Mastectomy": 4})
print(data.head())
#%%
"""
Breast Cancer Survival Prediction Model
"""
# # Splitting data
x = np.array(data[['Age', 'Gender', 'Protein1', 'Protein2', 'Protein3','Protein4', 
                    'Tumour_Stage', 'Histology', 'ER status', 'PR status', 
                    'HER2 status', 'Surgery_type']])
y = np.array(data[['Patient_Status']])
# xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.01, random_state=42)
# ytrain=ytrain.flatten()
# model = SVC()
# model.fit(xtrain, ytrain)
#%%
# training a machine learning model
from sklearn.tree import DecisionTreeClassifier
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.10, random_state=42)
model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))
#%%
# Prediction
# features = [['Age', 'Gender', 'Protein1', 'Protein2', 'Protein3',
#'Protein4', 'Tumour_Stage', 'Histology', 'ER status', 'PR status', 
#'HER2 status', 'Surgery_type']]
features = np.array([[7.3000e+01, 1.0000e+00, 4.4857e-01, 2.3013e+00, 1.1659e-02,
       6.0686e-01, 2.0000e+00, 2.0000e+00, 1.0000e+00, 1.0000e+00,
       2.0000e+00, 1.0000e+00]])
print(model.predict(features))
#%%










