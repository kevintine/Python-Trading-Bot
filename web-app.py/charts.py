import os
import sys
sys.path.append(os.path.abspath('../scripts'))
from pattern_detect import get_stock_with_symbol, get_candlestick_pattern_from_stock, get_current_date, get_past_date_by_day
from static.patterns import pattern_descriptions
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
    # Convert the NumPy array to a Pandas DataFrame
    data = data.to_frame(name='Pattern')
    # Get stcok data
    stockDataOriginal = yf.Ticker(symbol).history(period='1d', start=date_one_year_ago, end=current_date)
    # Localize the index
    stockDataOriginal.index = stockDataOriginal.index.tz_localize(None)
    # Concatenate the two DataFrames
    stockData = pd.concat([stockDataOriginal, data], axis=1)
    # Remove rows where the pattern is 0
    stockData = stockData[stockData['Pattern'] != 0]

    # plot the data
    fig = make_subplots(
        rows=2, cols=1,  
        shared_xaxes=True,  
        vertical_spacing=0.1,  
        row_heights=[0.8, 0.2] 
    )
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

    # Add a volume bar subchart
    fig.add_trace(
        py.graph_objs.Bar(
            x=stockDataOriginal.index,
            y=stockDataOriginal['Volume'],
            name='Volume',
            marker_color='blue',
        ),
        row=2, col=1  
    )

    # Update the layout to include a title
    fig.update_layout(
        title=f"{pattern} for {symbol}",
        title_font=dict(size=20, family="Arial", color="black"),
        title_x=0.5  # Center the title
    )
    file_name = f"{symbol}_candlestick_chart.html"
    file_path = os.path.join('static', file_name)
    fig.write_html(file_path)
    return file_path    

def get_pattern_descriptions(pattern):
    return pattern_descriptions.get(pattern)


# Testing to get the supports and resistances


#  Testing for get_chart_with_pattern()
symbol_tester = "SPY"
pattern_tester = "CDLENGULFING"

data = get_candlestick_pattern_from_stock(symbol_tester, pattern_tester)
# Convert the NumPy array to a Pandas DataFrame
data = data.to_frame(name='Pattern')
# Get stcok data
stockDataOriginal = yf.Ticker(symbol_tester).history(period='1d', start=date_one_year_ago, end=current_date)
# # Testing to get the supports and resistances
supports = stockDataOriginal[stockDataOriginal['Low'] == stockDataOriginal['Low'].rolling(5, center=True).min()]['Low']
resistances = stockDataOriginal[stockDataOriginal['High'] == stockDataOriginal['High'].rolling(5, center=True).max()]['High']
levels = pd.concat([supports, resistances])
levels = levels[abs(levels.diff()) > 2]
print(levels)
# resistances = stockDataOriginal[stockDataOriginal['High']== stockDataOriginal['High'].rolling(5, center=True).max()]['High']
# level = pd.concat([supports, resistances])
# print(level)
# Localize the index
stockDataOriginal.index = stockDataOriginal.index.tz_localize(None)
# Concatenate the two DataFrames
stockData = pd.concat([stockDataOriginal, data], axis=1)
# Remove rows where the pattern is 0
stockData = stockData[stockData['Pattern'] != 0]

# plot the data
fig = make_subplots(
    rows=2, cols=1,  
    shared_xaxes=True,  
    vertical_spacing=0.1,  
    row_heights=[0.8, 0.2]  
)
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

# Add a volume bar subchart
fig.add_trace(
    py.graph_objs.Bar(
        x=stockDataOriginal.index,
        y=stockDataOriginal['Volume'],
        name='Volume',
        marker_color='blue',
    ),
    row=2, col=1  
)

for index, row in levels.items():
    fig.add_shape(
        type="line",
        x0=index, x1=current_date,  # Extend line to the end
        y0=row, y1=row,
        line=dict(color="blue", width=1)
    )
fig.show()

# # Test for get_pattern_descriptions()
# pattern_tester = "CDLENGULFING"
# print(get_pattern_descriptions(pattern_tester))