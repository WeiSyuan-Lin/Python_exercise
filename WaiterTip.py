# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 16:39:53 2022

@author: nightrain
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default='jpg'
#%%
path='D:\Python\datasets\WaiterTip\\tips.csv'
#%%
data = pd.read_csv(path)
print(data.head())
#%%
"""
Waiter Tips Analysis :
    
look at the tips given to the waiters according to:

the total bill paid

number of people at a table

the day of the week:
    
"""
#%%
figure = px.scatter(data_frame = data, x="total_bill",
                    y="tip", size="size", color= "day")
figure.show()
#%%
figure = px.scatter(data_frame = data, x="total_bill",
                    y="tip", size="size", color= "sex")
figure.show()
#%%
figure = px.scatter(data_frame = data, x="total_bill",
                    y="tip", size="size", color= "time")
figure.show()
#%%
"""
tips given to the waiters according to 
the days to find out which day the most tips are given to the waiters
"""
figure = px.pie(data, 
             values='tip', 
             names='day',hole = 0.5)
figure.show()
#%%
"""
he number of tips given to waiters 
by gender of the person paying the bill to see who tips waiters the most
"""
figure = px.pie(data, 
             values='tip', 
             names='sex',hole = 0.5)
figure.show()
#%%
"""
Now let’s see if a smoker tips more or a non-smoker:
"""
figure = px.pie(data, 
             values='tip', 
             names='smoker',hole = 0.5)
figure.show()
#%%
figure = px.pie(data, 
             values='tip', 
             names='time',hole = 0.5)
figure.show()
#%%
"""
Waiter Tips Prediction Model
"""
#%%
"""
do some data transformation 
by transforming the categorical values into numerical values:
"""
data["sex"] = data["sex"].map({"Female": 0, "Male": 1})
data["smoker"] = data["smoker"].map({"No": 0, "Yes": 1})
data["day"] = data["day"].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})
data["time"] = data["time"].map({"Lunch": 0, "Dinner": 1})
data.head()
#%%
x = np.array(data[["total_bill", "sex", "smoker", "day", 
                   "time", "size"]])
y = np.array(data["tip"])

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)

#%%
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(xtrain, ytrain)
#%%
# features = [[total_bill, "sex", "smoker", "day", "time", "size"]]
features = np.array([[100.50, 1, 1, 0, 1, 4]])
model.predict(features)

