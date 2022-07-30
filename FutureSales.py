# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 23:33:45 2022

@author: nightrain
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#%%
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/advertising.csv")
print(data.head())
#%%
print(data.isnull().sum())
#%%
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='jpg'
figure = px.scatter(data_frame = data, x="Sales",
                    y="TV", size="TV", trendline="ols")
figure.show()
#%%
figure = px.scatter(data_frame = data, x="Sales",
                    y="Newspaper", size="Newspaper", trendline="ols")
figure.show()
#%%
figure = px.scatter(data_frame = data, x="Sales",
                    y="Radio", size="Radio", trendline="ols")
figure.show()
#%%
"""
look at the correlation of all the columns with the sales column:
"""
correlation = data.corr()
print(correlation["Sales"].sort_values(ascending=False))
#%%
"""
Future Sales Prediction Model
"""
x = np.array(data.drop(["Sales"], 1))
y = np.array(data["Sales"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y,test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))
#%%
#features = [[TV, Radio, Newspaper]]
features = np.array([[30.1, 330.8, 0.2]])
print(model.predict(features))











