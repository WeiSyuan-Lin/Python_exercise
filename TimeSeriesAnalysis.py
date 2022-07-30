# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:20:04 2022

@author: nightrain
"""

import pandas as pd
import yfinance as yf
import datetime
from datetime import date, timedelta
today = date.today()
#%%
d1 = today.strftime("%Y-%m-%d")
end_date = d1
d2 = date.today() - timedelta(days=720)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2

data = yf.download('AAPL', 
                      start=start_date, 
                      end=end_date, 
                      progress=False)
#%%
print(data.head())
#%%
import plotly.express as px
import plotly.io as pio
pio.renderers.default='jpg'

"""
A line plot is one of the best visualization tools 
while working on Time series analysis.
"""
figure = px.line(data, x = data.index, 
                 y = "Close", 
                 title = "Time Series Analysis (Line Plot)")
figure.show()
#%%
"""
A candlestick chart is always helpful in 
the time series analysis of a financial instrument
"""
import plotly.graph_objects as go
figure = go.Figure(data=[go.Candlestick(x = data.index,
                                        open = data["Open"], 
                                        high = data["High"],
                                        low = data["Low"], 
                                        close = data["Close"])])
figure.update_layout(title = "Time Series Analysis (Candlestick Chart)", 
                     xaxis_rangeslider_visible = False)
figure.show()
#%%
"""
The line chart and candlestick chart 
show you increase and decrease of the price, 
but if you want to see the price increase and decrease in the long term, 
you should always prefer a bar chart.
"""
figure = px.bar(data, x = data.index, 
                y = "Close", 
                title = "Time Series Analysis (Bar Plot)" )
figure.show()
#%%
figure = px.line(data, x = data.index, 
                 y = 'Close', 
                 range_x = ['2022-05-01','2022-6-30'], 
                 title = "Time Series Analysis (Custom Date Range)")
figure.show()
#%%
"""
you can manually select the time interval in 
the output visualization itself.
"""
figure = go.Figure(data = [go.Candlestick(x = data.index,
                                        open = data["Open"], 
                                        high = data["High"],
                                        low = data["Low"], 
                                        close = data["Close"])])
figure.update_layout(title = "Time Series Analysis (Candlestick Chart with Buttons and Slider)")

figure.update_xaxes(
    rangeslider_visible = True,
    rangeselector = dict(
        buttons = list([
            dict(count = 1, label = "1m", step = "month", stepmode = "backward"),
            dict(count = 6, label = "6m", step = "month", stepmode = "backward"),
            dict(count = 1, label = "YTD", step = "year", stepmode = "todate"),
            dict(count = 1, label = "1y", step = "year", stepmode = "backward"),
            dict(step = "all")
        ])
    )
)
figure.show()
#%%



