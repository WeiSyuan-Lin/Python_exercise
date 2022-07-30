# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 13:15:11 2022

@author: nightrain
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
#%%
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/bbc-news-data.csv", sep='\t')
print(data.head())
#%%
print(data.isnull().sum())
#%%
data["category"].value_counts()
#%%
"""
News Classification Model
"""
data = data[["title", "category"]]

x = np.array(data["title"])
y = np.array(data["category"])

cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
model = MultinomialNB()
model.fit(X_train,y_train)
#%%
user = input("Enter a Text: ")
data = cv.transform([user]).toarray()
output = model.predict(data)
print(output)
#%%



