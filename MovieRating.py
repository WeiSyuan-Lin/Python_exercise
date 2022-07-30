# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 12:56:31 2022

@author: nightrain
"""
import numpy as np
import pandas as pd
#%%
path="D:\Python\datasets\MovieRating\\movies.dat"
#%%
movies = pd.read_csv(path, delimiter='::')
print(movies.head())
#%%
movies.columns = ["ID", "Title", "Genre"]
print(movies.head())
#%%
path2="D:\Python\datasets\MovieRating\\ratings.dat"
ratings = pd.read_csv(path2, delimiter='::')
print(ratings.head())
#%%
ratings.columns = ["User", "ID", "Ratings", "Timestamp"]
print(ratings.head())
#%%
"""
merge these two datasets into one
"""
data = pd.merge(movies, ratings, on=["ID", "ID"])
print(data.head())
#%%
ratings = data["Ratings"].value_counts()
numbers = ratings.index
quantity = ratings.values
import plotly.express as px
fig = px.pie(data, values=quantity, names=numbers)
fig.show()
#%%
data2 = data.query("Ratings == 10")
print(data2["Title"].value_counts().head(10))
#%%




