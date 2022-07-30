# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 12:17:51 2022

@author: nightrain
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
pio.renderers.default='jpg'
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
#%%
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/demand.csv")
print(data.head())
#%%
data.isnull().sum()
data = data.dropna()
#%%
fig = px.scatter(data, x="Units Sold", y="Total Price",
                 size='Units Sold')
fig.show()
#%%
print(data.corr())
#%%
correlations = data.corr() #method='pearson'
plt.figure(figsize=(15, 12))
sns.heatmap(correlations, cmap="coolwarm", annot=True)
plt.show()
#%%
"""
Product Demand Prediction Model
"""
x = data[["Total Price", "Base Price"]]
y = data["Units Sold"]
xtrain, xtest, ytrain, ytest = train_test_split(x, y,test_size=0.2,random_state=42)
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
#%%
#features = [["Total Price", "Base Price"]]
features = np.array([[13.00, 140.00]])
model.predict(features)













