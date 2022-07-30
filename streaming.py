# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 16:36:40 2022

@author: nightrain
"""

import numpy as np # linear algebra
import pandas as pd # data processing

import plotly
import plotly.express as px
import plotly.io as pio
pio.renderers.default='jpg'
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
#%%
path='D:\Python\datasets\streaming\\tv_shows.csv'
#%%
tv_shows = pd.read_csv(path)
tv_shows.head()
#%%
"""
Most of the work can be done 
by visualizing and analyzing the ratings of shows on the streaming platforms.
"""
#preparing the data by dropping the duplicate values based on the title of the shows:
tv_shows.drop_duplicates(subset='Title',keep='first',inplace=True)
#%%
"""
fill the null values in the data with zeroes and 
then convert them into integer data types:
"""
tv_shows['Rotten Tomatoes'] = tv_shows['Rotten Tomatoes'].fillna('0%')
tv_shows['Rotten Tomatoes'] = tv_shows['Rotten Tomatoes'].apply(lambda x : x.rstrip('/100%'))
#%%
tv_shows['Rotten Tomatoes'] = pd.to_numeric(tv_shows['Rotten Tomatoes'])
#%%
tv_shows['IMDb'] = tv_shows['IMDb'].fillna('0.0/10')
#%%
tv_shows['IMDb'] = tv_shows['IMDb'].apply(lambda x : x.rstrip('/10'))
#%%
tv_shows['IMDb'] = tv_shows['IMDb'].astype('float')
#%%
tv_shows_long=pd.melt(tv_shows[['Title','Netflix','Hulu','Disney+',
                                'Prime Video']],id_vars=['Title'],
                      var_name='StreamingOn', value_name='Present')
tv_shows_long = tv_shows_long[tv_shows_long['Present'] == 1]
tv_shows_long.drop(columns=['Present'],inplace=True)
#%%
tv_shows_combined = tv_shows_long.merge(tv_shows, on='Title', how='inner')
tv_shows_combined.drop(columns = ['Unnamed: 0','Netflix',
                                  'Hulu', 'Prime Video', 'Disney+', 'type'], inplace=True)

#%%
tv_shows_both_ratings = tv_shows_combined[(tv_shows_combined.IMDb > 0) & tv_shows_combined['Rotten Tomatoes'] > 0]
tv_shows_combined.groupby('StreamingOn').Title.count().plot(kind='bar')
#%%
figure = []
figure.append(px.violin(tv_shows_both_ratings, x = 'StreamingOn', y = 'IMDb', color='StreamingOn'))
figure.append(px.violin(tv_shows_both_ratings, x = 'StreamingOn', y = 'Rotten Tomatoes', color='StreamingOn'))
fig = make_subplots(rows=2, cols=4, shared_yaxes=True)

for i in range(2):
    for j in range(4):
        fig.add_trace(figure[i]['data'][j], row=i+1, col=j+1)

fig.update_layout(autosize=False, width=800, height=800)        
fig.show()
#%%
"""
use a scatter plot to compare the ratings 
between IMBD and Rotten Tomatoes to compare 
which streaming platform has the best ratings in both the user rating platforms:
"""
px.scatter(tv_shows_both_ratings, x='IMDb',
           y='Rotten Tomatoes',color='StreamingOn')
#%%







