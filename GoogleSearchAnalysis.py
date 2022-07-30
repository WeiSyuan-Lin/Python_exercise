# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 15:11:39 2022

@author: nightrain
"""
import pandas as pd
from pytrends.request import TrendReq
"""
Google Trends provides an API that can be used to 
analyze the daily searches on Google.
"""
import matplotlib.pyplot as plt
#%%
trends = TrendReq()
#%%
"""
let’s create a DataFrame of the top 10 countries which search 
for “Machine Learning” on Google
"""
trends.build_payload(kw_list=["Machine Learning"])
data = trends.interest_by_region()
data = data.sort_values(by="Machine Learning", ascending=False)
data = data.head(10)
print(data)
#%%
data.reset_index().plot(x="geoName", y="Machine Learning", 
                        figsize=(20,15), kind="bar")
plt.style.use('fivethirtyeight')
plt.show()
#%%
"""
look at the trend of searches to see how 
the total search queries based on “Machine Learning” increased or decreased on Google:
"""
data = TrendReq(hl='en-US', tz=360)
data.build_payload(kw_list=['deep Learning'])
data = data.interest_over_time()
fig, ax = plt.subplots(figsize=(20, 15))
data['deep Learning'].plot()
plt.style.use('fivethirtyeight')
plt.title('Total Google Searches for deep Learning', fontweight='bold')
plt.xlabel('Year')
plt.ylabel('Total Count')
plt.show()
#%%











