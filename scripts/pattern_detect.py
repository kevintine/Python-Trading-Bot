import talib
import yfinance as yf
import mplfinance as mpf
from datetime import datetime, timedelta

# Current Date
current_date = datetime.now()

# Subtract 365 days from the current date to get the date one year ago
# Format the date as a string in the desired format
one_year_ago = current_date - timedelta(days=365)
date_one_year_ago = one_year_ago.strftime("%Y-%m-%d")
current_date = datetime.now().strftime("%Y-%m-%d")

# FUNCTION: Returns the current date
def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")

# FUNCTION: Returns the past date specifed by the number of days provided
def get_past_date_by_day(numOfDays):
    current_date = datetime.now()
    one_year_ago = current_date - timedelta(days=numOfDays)
    date_one_year_ago = one_year_ago.strftime("%Y-%m-%d")
    return date_one_year_ago


# Sample code to how to use talib candlestick patterns
# data = yf.download("SPY", start="2020-01-01", end="2020-08-01")
# engulfing = talib.CDLENGULFING(data['Open'], data['High'], data['Low'], data['Close'])
# morning_star = talib.CDLMORNINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])
# data['Morning Star'] = morning_star
# data['Englufing'] = engulfing

# This takes a string of a stock symbol and returns the data for that stock from 2022-01-01 till today's date
def get_stock_with_symbol(symbol):
    data = yf.download(symbol, start=date_one_year_ago, end=current_date)
    return data
# This takes a string of a stock symbol and a pattern of a candlestick pattern and returns the data which contains that pattern int the stock data
def get_candlestick_pattern_from_stock(symbol, pattern):
    data = yf.download(symbol, start=date_one_year_ago, end=current_date)
    pattern_function = getattr(talib, pattern)
    result = pattern_function(data['Open'], data['High'], data['Low'], data['Close'])
    return result




