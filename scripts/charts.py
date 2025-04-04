import os
import sys
sys.path.append(os.path.abspath('../scripts'))
sys.path.append(os.path.abspath('../tradingbot/models'))
import plotly as py
from plotly.subplots import make_subplots
import pandas as pd
import datetime as dt
import yfinance as yf
from pattern_detect import get_stock_with_symbol, get_candlestick_pattern_from_stock, get_current_date, get_past_date_by_day
from static.patterns import pattern_descriptions
from strategies import cdl_hammer_web, cdl_hammer_bot, get_engulfing, get_hammer, get_volume_indicator, get_green_candle, get_sma_indicator, get_volume_indicator

# Keep dates as datetime objects
current_date = dt.datetime.now()
one_year_ago = current_date - dt.timedelta(days=365)
three_years_ago = current_date - dt.timedelta(days=1095)
five_years_ago = current_date - dt.timedelta(days=1825)

# Only convert to string if necessary for APIs that require it
date_one_year_ago_str = one_year_ago.strftime("%Y-%m-%d")
current_date_str = current_date.strftime("%Y-%m-%d")  # If needed

# FUNCTION: Create a candlestick chart with volume subchart
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

    file_name = f"{symbol}_candlestick_chart.html"
    file_path = os.path.join('static', file_name)
    fig.write_html(file_path)

    return file_path

# FUNCTION: Create a candlestick chart with volume subchart and pattern detection
def get_chart_with_pattern(symbol, pattern):
    data = get_candlestick_pattern_from_stock(symbol, pattern)
    
    # Convert NumPy array to Pandas DataFrame
    data = data.to_frame(name='Pattern')

    # Get stock data
    stockDataOriginal = yf.Ticker(symbol).history(period='1d', start=one_year_ago, end=current_date)
    
    # Calculate 50-day SMA and 50-day average volume
    stockDataOriginal['50_day_SMA'] = stockDataOriginal['Close'].rolling(window=50).mean()
    stockDataOriginal['50_day_avg_volume'] = stockDataOriginal['Volume'].rolling(window=50).mean()

    # Calculate support and resistance levels
    supports = stockDataOriginal[stockDataOriginal['Low'] == stockDataOriginal['Low'].rolling(5, center=True).min()]['Low']
    resistances = stockDataOriginal[stockDataOriginal['High'] == stockDataOriginal['High'].rolling(5, center=True).max()]['High']
    levels = pd.concat([supports, resistances])
    levels = levels[abs(levels.diff()) > 1]
    
    # Localize the index
    stockDataOriginal.index = stockDataOriginal.index.tz_localize(None)

    # Concatenate the two DataFrames
    stockData = pd.concat([stockDataOriginal, data], axis=1)

    # Remove rows where the pattern is 0
    stockData = stockData[stockData['Pattern'] != 0]

    # Identify significant volume spikes (1.3× 50-day avg volume)
    stockData['Volume_Spike'] = stockData['Volume'] > (1.3 * stockData['50_day_avg_volume'])
    
    # Get buy positions
    # use the functions 
    

    # Plot the data
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.8, 0.2]
    )

    # Candlestick chart
    fig.add_trace(
        py.graph_objs.Candlestick(
            x=stockDataOriginal.index,
            open=stockDataOriginal['Open'],
            high=stockDataOriginal['High'],
            low=stockDataOriginal['Low'],
            close=stockDataOriginal['Close'],
            name='Candlestick',
        ),
        row=1, col=1
    )

    # Highlight detected patterns with orange/purple candlesticks
    fig.add_trace(
        py.graph_objs.Candlestick(
            x=stockData.index,
            open=stockData['Open'],
            high=stockData['High'],
            low=stockData['Low'],
            close=stockData['Close'],
            name='Pattern',
            increasing=dict(line=dict(color='orange'), fillcolor='orange'),
            decreasing=dict(line=dict(color='purple'), fillcolor='purple'),
        ),
        row=1, col=1
    )

    # Volume bar subchart
    fig.add_trace(
        py.graph_objs.Bar(
            x=stockDataOriginal.index,
            y=stockDataOriginal['Volume'],
            name='Volume',
            marker_color='blue',
        ),
        row=2, col=1
    )

    # 50-day SMA line
    fig.add_trace(
        py.graph_objs.Scatter(
            x=stockDataOriginal.index,
            y=stockDataOriginal['50_day_SMA'],
            name='50-day SMA',
            line=dict(color='green', width=2),
        ),
        row=1, col=1
    )

    # Add support and resistance lines
    # for index, row in levels.items():
    #     fig.add_shape(
    #         type="line",
    #         x0=index, x1=current_date,  # Extend line to the end
    #         y0=row, y1=row,
    #         line=dict(color="blue", width=1)
    # )

    # Update layout
    fig.update_layout(
        title=f"{pattern} for {symbol}",
        title_font=dict(size=20, family="Arial", color="black"),
        title_x=0.5
    )

    # Save the file
    file_name = f"{symbol}_candlestick_chart.html"
    file_path = os.path.join('static', file_name)
    fig.write_html(file_path)

    return file_path


def get_pattern_descriptions(pattern):
    return pattern_descriptions.get(pattern)


# Testing to get the supports and resistances


# #  Testing for get_chart_with_pattern()
# import yfinance as yf
# import pandas as pd
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# # Testing for get_chart_with_pattern()
# symbol_tester = "INTC"
# pattern_tester = "CDLENGULFING"

# # Get stock data
# current_date = pd.Timestamp.today().date()
# date_one_year_ago = current_date - pd.DateOffset(years=1)
# stockDataOriginal = yf.Ticker(symbol_tester).history(period='1d', start=date_one_year_ago, end=current_date)

# # Calculate 50-day SMA
# stockDataOriginal['50_day_SMA'] = stockDataOriginal['Close'].rolling(window=50).mean()

# # Testing to get the supports and resistances
# supports = stockDataOriginal[stockDataOriginal['Low'] == stockDataOriginal['Low'].rolling(5, center=True).min()]['Low']
# resistances = stockDataOriginal[stockDataOriginal['High'] == stockDataOriginal['High'].rolling(5, center=True).max()]['High']
# levels = pd.concat([supports, resistances])
# levels = levels[abs(levels.diff()) > 2]

# # Localize the index
# stockDataOriginal.index = stockDataOriginal.index.tz_localize(None)

# # Plot the data
# fig = make_subplots(
#     rows=2, cols=1,  
#     shared_xaxes=True,  
#     vertical_spacing=0.1,  
#     row_heights=[0.8, 0.2]  
# )

# # Add candlestick chart
# fig.add_trace(
#     go.Candlestick(
#         x=stockDataOriginal.index,
#         open=stockDataOriginal['Open'],
#         high=stockDataOriginal['High'],
#         low=stockDataOriginal['Low'],
#         close=stockDataOriginal['Close'],
#         name='Candlestick',
#     ),
#     row=1, col=1    
# )

# # Add 50-day SMA line
# fig.add_trace(
#     go.Scatter(
#         x=stockDataOriginal.index,
#         y=stockDataOriginal['50_day_SMA'],
#         name='50-day SMA',
#         line=dict(color='green', width=2),
#     ),
#     row=1, col=1  # Specify row and col here
# )

# # Add a volume bar subchart
# fig.add_trace(
#     go.Bar(
#         x=stockDataOriginal.index,
#         y=stockDataOriginal['Volume'],
#         name='Volume',
#         marker_color='blue',
#     ),
#     row=2, col=1  
# )

# # # Add support and resistance lines
# # for index, row in levels.items():
# #     fig.add_shape(
# #         type="line",
# #         x0=index, x1=current_date,  # Extend line to the end
# #         y0=row, y1=row,
# #         line=dict(color="blue", width=1)
# # )

# # Update layout
# fig.update_layout(
#     title=f"{symbol_tester} Candlestick Chart with 50-day SMA, Support/Resistance, and Volume",
#     xaxis_title="Date",
#     yaxis_title="Price",
#     xaxis_rangeslider_visible=False,
#     showlegend=True
# )

# # Show the chart
# fig.show()

# # Test for get_pattern_descriptions()
# pattern_tester = "CDLENGULFING"
# print(get_pattern_descriptions(pattern_tester))