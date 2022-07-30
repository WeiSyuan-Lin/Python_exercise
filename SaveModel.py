# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 21:27:45 2022

@author: nightrain
"""

import matplotlib.pylab as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn import datasets
#%%
diabetes = datasets.load_diabetes()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=0)
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, test_size=0.2, random_state=0)
# There are three steps to model something with sklearn
# 1. Set up the model
model = LinearRegression()
# 2. Use fit
model.fit(X_train, y_train)
#%%
# save model : saved as a byte stream
import pickle
with open("pickle_model", "wb") as file:
    pickle.dump(model, file)
#%%
model2=LinearRegression()
with open("pickle_model", "rb") as file:
    model2 = pickle.load(file)

predictions = model2.predict(X_test)

#%%
print(predictions)    