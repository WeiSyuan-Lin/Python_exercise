# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 15:13:00 2022

@author: nightrain
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#Seaborn is a library for making attractive and informative statistical graphics in Python.
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
#Word Cloud is a data visualization technique used for representing text data in which the size of each word indicates its frequency or importance.
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor
#%%
path='D:\Python\datasets\InstagramReach\Instagram.csv'
#%%
data = pd.read_csv(path, encoding = 'latin1')
print(data.head())
#%%
"""
Before starting everything, 
let’s have a look at whether this dataset contains any null values or not:
"""
data.isnull().sum()
#%%
"""
So it has a null value in every column. 
Let’s drop all these null values and move further:
"""
data = data.dropna()
#%%
data.info()
#%%
"""
Analyzing Instagram Reach
"""
#%%
plt.figure(figsize=(10, 8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impressions From Home")
sns.distplot(data['From Home'])
plt.show()
#%%
plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Hashtags")
sns.distplot(data['From Hashtags'])
plt.show()
#%%
plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Explore")
sns.distplot(data['From Explore'])
plt.show()
#%%
import plotly.io as pio
pio.renderers.default='jpg'

home = data["From Home"].sum()
hashtags = data["From Hashtags"].sum()
explore = data["From Explore"].sum()
other = data["From Other"].sum()

labels = ['From Home','From Hashtags','From Explore','Other']
values = [home, hashtags, explore, other]

fig = px.pie(data, values=values, names=labels, 
             title='Impressions on Instagram Posts From Various Sources', hole=0.5)
fig.show()
#%%
"""
Analyzing Content

understand the kind of content  posted on Instagram.
"""
text = " ".join(i for i in data.Caption)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.style.use('classic')
plt.figure( figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
#%%
text = " ".join(i for i in data.Hashtags)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure( figsize=(12,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
#%%
"""
Analyzing Relationships
"""
#relationship between the number of likes and the number of impressions
figure = px.scatter(data_frame = data, x="Impressions",y="Likes", 
                    title = "Relationship Between Likes and Impressions")
figure.show()
#%%
#elationship between the number of comments and the number of impressions 
figure = px.scatter(data_frame = data, x="Impressions",
                    y="Comments", size="Comments",  
                    title = "Relationship Between Comments and Total Impressions")
figure.show()
#%%
#the relationship between the number of shares and the number of impressions
figure = px.scatter(data_frame = data, x="Impressions",
                    y="Shares", size="Shares",
                    title = "Relationship Between Shares and Total Impressions")
figure.show()
#%%
# relationship between the number of saves and the number of impressions
figure = px.scatter(data_frame = data, x="Impressions",
                    y="Saves", size="Saves",
                    title = "Relationship Between Post Saves and Total Impressions")
figure.show()
#%%
# look at the correlation of all the columns with the Impressions column
correlation = data.corr()
print(correlation["Impressions"].sort_values(ascending=False))
#%%
#Analyzing Conversion Rate
conversion_rate = (data["Follows"].sum() / data["Profile Visits"].sum()) * 100
print(conversion_rate)
#%%
figure = px.scatter(data_frame = data, x="Profile Visits",
                    y="Follows", size="Follows",  
                    title = "Relationship Between Profile Visits and Followers Gained")
figure.show()
#%%
"""
Instagram Reach Prediction Model
"""
x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 
                   'Profile Visits', 'Follows']])
y = np.array(data["Impressions"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)
#%%
model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)
#%%
# Features = [['Likes','Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']]
features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
model.predict(features)

#%%









