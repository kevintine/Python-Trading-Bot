import os
import sys
sys.path.append(os.path.abspath('../scripts'))
from pattern_detect import get_stock_with_symbol, get_candlestick_pattern_from_stock, get_current_date, get_past_date_by_day
import charts

from flask import Flask, render_template, request
from flask_scss import Scss
from flask_sqlalchemy import SQLAlchemy
from static.patterns import patterns, stocks
from flask import jsonify
import mplfinance as mpf


app = Flask(__name__)

# create a candlestick chart for a stock
def plot_candlestick_chart(data, filename):
    mpf.plot(data, type='candle', style='charles', volume=True, savefig=filename, figscale=2, figratio=(5,4))
# create a chart for a stock and a pattern
def plot_candlestick_chart_with_pattern(data, filename, ymax, ymin):
    mpf.plot(data, type='candle', style='charles', volume=True, savefig=filename, figscale=2, figratio=(5,4), ylim=(ymin, ymax))


@app.route("/")
def index():
    return render_template("index.html", patterns=patterns, stocks=stocks)

@app.route("/handle_form", methods=["GET", "POST"])
def handle_form():
    # pull options from url parameters
    pattern = request.args.get("pattern")
    stock = request.args.get("stock")

    # get the stock data
    data = get_stock_with_symbol(stock)

    # get the pattern data
    pattern_recognition = get_candlestick_pattern_from_stock(stock, pattern)

    # Define the file path to save the chart
    chart_path = os.path.join('static', 'charts', f'{stock}.png')
    candlestick_chart_path = os.path.join('static', 'charts', f'{stock}_{pattern}.png')

    # NEEDS WORK: numbers to set the y axis limits. Needs to be updated automatically based off the data
    ymax = 130
    ymin = 80

    # Plot the candlestick chart in the main thread
    plot_candlestick_chart(data, chart_path)

    # if there is a pattern, create a chart with that data and display it on the html page
    data['Pattern'] = pattern_recognition
    columns_to_zero = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    data.loc[data['Pattern'] != 100, columns_to_zero] = 0
    data_with_pattern = data.drop(columns=['Pattern'])

    # Check for uptrend or downtrend in charts, in cycles of 1 week, 2 weeks, 1 month, 1 year
    data = get_stock_with_symbol(stock)
    print(len(data))

    # Get the trend images path
    uptrend = 'static/images/uptrend_arrow.jpg'
    downtrend = 'static/images/downtrend_arrow.jpg'

    sevenDays = ''
    fourteenDays = ''
    oneMonth = ''
    oneYear = ''

    if data.iloc[0]['Close'] < data.iloc[5]['Close']:
        sevenDays = downtrend
    else:
        sevenDays = uptrend

    if data.iloc[0]['Close'] < data.iloc[10]['Close']:
        fourteenDays = downtrend
    else:
        fourteenDays = uptrend

    if data.iloc[0]['Close'] < data.iloc[22]['Close']:
        oneMonth = downtrend
    else:
        oneMonth = uptrend
    
    if data.iloc[0]['Close'] < data.iloc[249]['Close']:
        oneYear = downtrend
    else:
        oneYear = uptrend
    

    plot_candlestick_chart_with_pattern(data_with_pattern, candlestick_chart_path, ymax, ymin)

    return render_template("index.html", 
                           patterns=patterns, 
                           stocks=stocks, 
                           chart_path=chart_path, 
                           candlestick_chart_path=candlestick_chart_path, 
                           sevenDays = sevenDays,
                           fourteenDays = fourteenDays,
                           oneMonth = oneMonth,
                           oneYear = oneYear
    )


# 2025 Changes


if __name__ in "__main__":
    app.run(debug=True, port=5001)