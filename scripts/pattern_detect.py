import os
import sys
sys.path.append(os.path.abspath('../tradingbot/models'))
import talib
import pandas as pd
import yfinance as yf
import mplfinance as mpf
from datetime import datetime, timedelta
from strategies import cdl_hammer_web
# Current Date
current_date = datetime.now()

# Subtract 365 days from the current date to get the date one year ago
# Format the date as a string in the desired format
one_year_ago = current_date - timedelta(days=365)
date_one_year_ago = one_year_ago.strftime("%Y-%m-%d")
date_three_years_ago = one_year_ago - timedelta(days=1095)
date_five_years_ago = one_year_ago - timedelta(days=1825)
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

# This takes a string of a stock symbol and returns the data for that stock from 2022-01-01 till today's date
def get_stock_with_symbol(symbol):
    data = yf.download(symbol, start=date_five_years_ago, end=current_date)
    return data
# This takes a string of a stock symbol and a pattern of a candlestick pattern and returns the data which contains that pattern int the stock data
def get_candlestick_pattern_from_stock(symbol, pattern):
    data = yf.download(symbol, start=date_one_year_ago, end=current_date)
    df = data[['Open', 'High', 'Low', 'Close']].copy()
    if pattern == "HAMMER":
        result = cdl_hammer_web(data['Open'].squeeze(),
        data['High'].squeeze(),
        data['Low'].squeeze(),
        data['Close'].squeeze(), 1, 1)
        return result
    
    pattern_function = getattr(talib, pattern)
    result = pattern_function(
        data['Open'].squeeze(),
        data['High'].squeeze(),
        data['Low'].squeeze(),
        data['Close'].squeeze()
    )
    return result

get_candlestick_pattern_from_stock("AC.TO", "CDLHAMMER")
# Sample code to how to use talib candlestick patterns
# data = yf.download("SPY", start="2020-01-01", end="2020-08-01")
# data = data.to_numpy()
# print(data)
# engulfing = talib.CDLENGULFING(data['Open'], data['High'], data['Low'], data['Close'])
# morning_star = talib.CDLMORNINGSTAR(data['Open'], data['High'], data['Low'], data['Close'])
# data['Morning Star'] = morning_star
# data['Engulfing'] = engulfing
# pd.set_option('display.max_rows', None)
# print(engulfing)




