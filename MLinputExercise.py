# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 12:01:30 2022

@author: nightrain
"""

import pandas as pd
import numpy as np
#%%
path="D:\Python\datasets\MLinputExercise\\IRIS.csv"
#%%
iris = pd.read_csv(path)
# splitting the dataset
x = iris.drop("species", axis=1)
y = iris["species"]
#%%
print(iris.head())
#%%
print(iris.describe())
#%%
print("Target Labels", iris["species"].unique())
#%%
import plotly.express as px
import plotly.io as pio
pio.renderers.default='jpg'
fig = px.scatter(iris, x="sepal_width", y="sepal_length", color="species")
fig.show()
#%%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2, 
                                                    random_state=0)
# training the model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
# giving inputs to the machine learning model
#%%
# features = [[sepal_length, sepal_width, petal_length, petal_width]]
features = np.array([[4.9, 2.5, 4.5, 1.7]])
# using inputs to predict the output
prediction = knn.predict(features)
print("Prediction: {}".format(prediction))