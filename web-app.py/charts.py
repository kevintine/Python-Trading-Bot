import plotly as py
from plotly.subplots import make_subplots
import pandas as pd
import datetime as dt
import yfinance as yf

# Date Calculations
current_date = dt.datetime.now()
one_year_ago = current_date - dt.timedelta(days=365)
date_one_year_ago = one_year_ago.strftime("%Y-%m-%d")
current_date = dt.datetime.now().strftime("%Y-%m-%d")

# Function to create a candlestick chart with volume subchart
def get_chart(symbol):
    # Get Data
    stockData = yf.Ticker(symbol).history(period='1d', start=date_one_year_ago, end=current_date)

    # Create a subplot figure
    fig = make_subplots(
        rows=2, cols=1,  
        shared_xaxes=True,  
        vertical_spacing=0.1,  
        row_heights=[0.8, 0.2]  
    )

    # Plot Data
    fig.add_trace(
        py.graph_objs.Candlestick(
            x=stockData.index,
            open=stockData['Open'],
            high=stockData['High'],
            low=stockData['Low'],
            close=stockData['Close'],
            name='Candlestick',
        ),
        row=1, col=1  
    )

    # Add a volume bar subchart
    fig.add_trace(
        py.graph_objs.Bar(
            x=stockData.index,
            y=stockData['Volume'],
            name='Volume',
            marker_color='blue',
        ),
        row=2, col=1  
    )
    
    fig.write_html('candlestick_chart.html')
