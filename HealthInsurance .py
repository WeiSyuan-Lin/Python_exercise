# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 12:47:46 2022

@author: nightrain
"""

import numpy as np
import pandas as pd
#%%
path="D:\Python\datasets\Health\\Health_insurance.csv"
#%%
data = pd.read_csv(path)
print(data.head())
#%%
data.isnull().sum()
#%%
import plotly.express as px
import plotly.io as pio
pio.renderers.default='jpg'
figure = px.histogram(data, x = "sex", color = "smoker", title= "Number of Smokers")
figure.show()
#%%
data["sex"] = data["sex"].map({"female": 0, "male": 1})
data["smoker"] = data["smoker"].map({"no": 0, "yes": 1})
print(data.head())
#%%
import plotly.express as px
pie = data["region"].value_counts()
regions = pie.index
population = pie.values
fig = px.pie(data, values=population, names=regions)
fig.show()
#%%
print(data.corr())
#%%
"""
Health Insurance Premium Prediction Model
"""
x = np.array(data[["age", "sex", "bmi", "smoker"]])
y = np.array(data["charges"])

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor()
forest.fit(xtrain, ytrain)
#%%
ypred = forest.predict(xtest)
data = pd.DataFrame(data={"Predicted Premium Amount": ypred})
print(data.head())
#%%






