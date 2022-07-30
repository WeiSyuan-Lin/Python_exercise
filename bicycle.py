# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 20:44:24 2022

@author: nightrain
"""
import pandas as pd
#%%
path="D:\Python\datasets\Bicycle\\fremont-bridge.csv"
#%%
data = pd.read_csv(path, index_col= 'Date',parse_dates=True)
print(data.head())
#%%
data.columns = ["West", "East"]
data["Total"] = data["West"] + data["East"] 
print(data.head())
#%%
data.dropna().describe()
#%%
#Visualizing the data
import matplotlib.pyplot as plt
import seaborn
seaborn.set()
data.plot()
plt.ylabel("Hourly Bicycle count")
plt.show()
#%%
weekly = data.resample("W").sum()
#%%
weekly.plot(style=[':', '--', '-'])
plt.ylabel('Weekly bicycle count')
plt.show()
#%%
daily = data.resample('D').sum()
daily.rolling(30, center=True).sum().plot(style=[':', '--', '-'])
plt.ylabel('mean hourly count')
plt.show()
#%%
daily.rolling(50, center=True,
              win_type='gaussian').sum(std=10).plot(style=[':','--', '-'])
plt.show()
#%%
import numpy as np
by_time = data.groupby(data.index.time).mean()
hourly_ticks = 4 * 60 * 60 * np.arange(6)
by_time.plot(xticks= hourly_ticks, style=[':', '--', '-'])
plt.ylabel("Traffic according to time")
plt.show()





