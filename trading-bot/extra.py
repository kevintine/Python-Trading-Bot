import yfinance as yf
import talib as ta
import numpy as np
import pandas as pd

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


a = 2
b = 100
print(a)
print(b)

a = a + b
b = a - b
a = a - b

print(a)
print(b)