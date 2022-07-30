# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 21:53:18 2022

@author: nightrain
"""

import pandas as pd
import numpy as np
#%%
path="D:\Python\datasets\COVID19\\COVID19.csv"
#%%
data = pd.read_csv(path)
print(data.head())
#%%
data.isnull().sum()
#%%
data = data.drop("Date", axis=1)
#%%
import plotly.express as px
import plotly.io as pio
pio.renderers.default='jpg'

fig = px.bar(data, x='Date_YMD', y='Daily Confirmed')
fig.show()
#%%
"""
Covid-19 Death Rate Analysis
"""
cases = data["Daily Confirmed"].sum()
deceased = data["Daily Deceased"].sum()

labels = ["Confirmed", "Deceased"]
values = [cases, deceased]

fig = px.pie(data, values=values, 
             names=labels, 
             title='Daily Confirmed Cases vs Daily Deaths', hole=0.5)
fig.show()
#%%
death_rate = (data["Daily Deceased"].sum() / data["Daily Confirmed"].sum()) * 100
print(death_rate)
#%%
fig = px.bar(data, x='Date_YMD', y='Daily Deceased')
fig.show()
#%%
"""
Covid-19 Deaths Prediction Model
"""
from autots import AutoTS
#  AutoTS library, which is one of the best Automatic Machine Learning libraries for Time Series Analysis
model = AutoTS(forecast_length=30, frequency='infer', ensemble='simple')
model = model.fit(data, date_col="Date_YMD", value_col='Daily Deceased', id_col=None)
prediction = model.predict()
forecast = prediction.forecast
print(forecast)
#%%



