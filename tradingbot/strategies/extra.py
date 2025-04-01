import yfinance as yf
import talib as ta
import numpy as np
import pandas as pd
from collections import deque

stocks = ["AC.TO", "SU.TO", "RY.TO", "TD.TO", "CM.TO"]  
patterns = ["CDLENGULFING", "CDLHAMMER", "CDLSHOOTINGSTAR"] 
start="2023-01-01"
end="2025-01-01"

# FUNCTION: to get ta-lib pattern
def get_talib_pattern(stock, pattern):
    # get ta-lib pattern
    # takes a stock string and pattern string
    # some issues:
    # maybe two years ago, yfinance used to return data a certain way and now its changed where ta-lib wasnt able to take in its values 
    # i had to change the code for ta-lib to continue to accept a pandas series where yfinance was give it a pandas dataframe. 
    # yfinance issue
    data = yf.download(stock, start=start,end=end)
    pattern_function = getattr(ta, pattern)
    integer = pattern_function(data['Open'].squeeze(), data['High'].squeeze(), data['Low'].squeeze(), data['Close'].squeeze())
    return integer

# FUNCTION: get instances of above average volume
def get_above_average_volume(stock):
    # calculate the average volume and pick out days where the volume is 1.4 times higher than that
    # return something accurate like the ta-lib pattern with 0, 100 and -100
    data = yf.download(stock, start=start,end=end)
    volume = data['Volume'].squeeze().mean()
    threshold = volume * 1.7
    volume_indicator = data['Volume'].squeeze().apply(lambda x: 100 if x > threshold else 0)
    volume_df = pd.DataFrame({
        'Date': data.index,
        'Volume Indicator': volume_indicator
    })
    volume_df = volume_df["Volume Indicator"]
    return volume_df

# FUNCTION: get both the ta-lib and volume df and check when both are not 0 then initiate a buy
def buy(talib, volume):
    # check when talib is 100 and volume is 100
    for index, (day_value, volume_value) in enumerate(zip(talib, volume)):
        if day_value == 100 and volume_value != 0:
            print(f"Buy! Talib Index: {talib.index[index]}")  # Print the actual index
            return 0
    print("No buys!")
    return 0

def restore_yfinance_structure(df):
    """
    Takes a simplified or nested Series-based rolling window DataFrame and restores it to the 
    yfinance-style structure with flat float values, expected columns, and a datetime index.

    Parameters:
        df (pd.DataFrame): DataFrame with nested Series or flat OHLC columns

    Returns:
        pd.DataFrame: DataFrame with yfinance-style columns (Open, High, Low, Close, Adj Close, Volume)
    """
    df_restored = df.copy()

    # Flatten nested Series inside each cell (replacing deprecated applymap)
    for col in df_restored.columns:
        df_restored[col] = df_restored[col].apply(lambda x: x.iloc[0] if isinstance(x, pd.Series) else x)

    # Add missing yfinance-style columns
    if 'Adj Close' not in df_restored.columns:
        df_restored['Adj Close'] = df_restored['Close']
    
    if 'Volume' not in df_restored.columns:
        df_restored['Volume'] = 0  # Placeholder value

    # Enforce correct data types
    for col in df_restored.columns:
        if col == "Volume":
            df_restored[col] = df_restored[col].astype(int)
        else:
            df_restored[col] = pd.to_numeric(df_restored[col], errors='coerce')

    # Reorder columns to match yfinance format
    columns_order = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    df_restored = df_restored[columns_order]

    # Optional: add datetime index if missing
    if not isinstance(df_restored.index, pd.DatetimeIndex):
        df_restored.index = pd.date_range(end=pd.Timestamp.today(), periods=len(df_restored))

    return df_restored





