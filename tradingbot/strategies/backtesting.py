import yfinance as yf
import talib as ta
import numpy as np
import pandas as pd
from collections import deque
from alpaca.trading.client import TradingClient
import os
import sys
sys.path.append(os.path.abspath('..'))
from models.classes import Signal, Trade, Account
from models.strategies import cdl_hammer_bot

trading_client = TradingClient("PKGJHW34RDC8CG26S1OB", "NDu8J0yVPfo7enmZaTFBPSUpbCVdw1ccWYWExNxI")
stocks = ["AC.TO", "SU.TO", "RY.TO", "TD.TO", "CM.TO"]  
patterns = ["CDLENGULFING", "CDLHAMMER", "CDLSHOOTINGSTAR"] 
start="2023-01-01"
end="2025-01-01"

import pandas as pd
import yfinance as yf
import talib as ta
from collections import deque

# ["AC.TO", "SU.TO", "RY.TO", "TD.TO", "CM.TO", "AW.TO", "ENB.TO", "CNQ.TO"]
# ["INTC", "AMD", "MSFT", "AAPL"]
# 10 stocks in the nyse worth between 10-60 dollars
# ["BAC", "T", "PFE", "VZ", "CSCO", "F", "GE", "FCX", "X", "MTCH", "TOST", "TEVA", "AMCR", "AMC", "AMN", "AMD"]



def main():
    # Parameters
    symbols = ["BAC", "T", "PFE", "VZ", "CSCO", "F", "GE", "FCX", "X", "MTCH", "TOST", "TEVA", "AMCR", "AMC", "AMN", "AMD"]
  # List of stocks
    lookback_period = 252  # 1-year historical data (trading days)
    account = Account(2000)  # Create a single account

    for symbol in symbols:
        print(f"\nFetching data for {symbol}...")

        # Fetch last 2 years of data
        end_date = pd.Timestamp.today().date()
        start_date = end_date - pd.DateOffset(years=2)
        data = yf.download(symbol, start=start_date, end=end_date)

        if data.empty:
            print(f"No data retrieved for {symbol}. Skipping...")
            continue

        # Calculate 50-day SMA for the entire dataset
        data['50_day_SMA'] = data['Close'].rolling(window=50).mean()

        # Split into historical & simulated data
        split_index = len(data) // 2
        historical_data = data.iloc[:split_index].copy()
        simulation_data = data.iloc[split_index:].copy()

        # Initialize rolling window with historical OHLC data
        rolling_window = deque(maxlen=lookback_period)
        for _, row in historical_data.iterrows():
            rolling_window.append((row["Open"], row["High"], row["Low"], row["Close"]))

        average_volume = historical_data["Volume"].mean()

        # Calculate average volume for historical data
        average_volume = historical_data["Volume"].mean().item()

        print(f"Starting simulation for {symbol} with {len(historical_data)} historical days and {len(simulation_data)} simulated days.")

        # Simulate each day in the second year
        for date, row in simulation_data.iterrows():
            # Add new day's OHLC data
            rolling_window.append((row["Open"], row["High"], row["Low"], row["Close"]))
            # Convert to Pandas DataFrame for TA-Lib
            df = pd.DataFrame(rolling_window, columns=["Open", "High", "Low", "Close"])
            # Extract row values as scalars
            open_price = row["Open"].item()
            high_price = row["High"].item()
            low_price = row["Low"].item()
            close_price = row["Close"].item()
            volume = row["Volume"].item()
            sma_50 = row["50_day_SMA"].item() 
            # Calculate engulfing pattern
            engulfing = float(ta.CDLENGULFING(df["Open"], df["High"], df["Low"], df["Close"]).iloc[-1])
            # Calculater hammer pattern
            hammer = float(ta.CDLHAMMER(df["Open"], df["High"], df["Low"], df["Close"]).iloc[-1])
            # Calculate volume indicator
            volume_indicator = 100 if volume > (1.2 * average_volume) else 0
            # Calcualte the sma indicator
            sma_indicator = 100 if close_price > sma_50 else 0
            # Calculate hammer pattern using in house function
            # Use .values to pass only the numerical data (prices) to the cdl_hammer function
            new_hammer = float(cdl_hammer_bot(df["Open"], df["High"], df["Low"], df["Close"]).iloc[-1])




            # # Print daily simulation results
            # print(
            #     f"Date: {date.date()}, Symbol: {symbol}, "
            #     f"Open: {open_price:.2f}, High: {high_price:.2f}, Low: {low_price:.2f}, Close: {close_price:.2f}, "
            #     f"Indicator: {engulfing}, Volume Indicator: {volume_indicator}, SMA Indicator: {sma_indicator}"
            # )

            # Run account check trades
            account.check_trades(close_price, symbol)

            # Buy condition: Engulfing + High Volume
            # if engulfing == 100 and volume_indicator == 100:
            #     num_stocks = 500 // open_price
            #     account.add_trade(open_price, num_stocks, symbol, date)
            # Buy condition: Hammer + High SMA
            # if engulfing != 0 and sma_indicator == 100:
            #     num_stocks = 500 // open_price
            #     account.add_trade(open_price, num_stocks, symbol, date)
            # Buy condition: High Volume + High SMA
            # if volume_indicator == 100 and sma_indicator == 100:
            #     num_stocks = 500 // open_price
            #     account.add_trade(open_price, num_stocks, symbol, date)
            # Buy condition: High Volume + High SMA + Engulfing
            # if hammer != 0 and volume_indicator == 100 and sma_indicator == 100:
            #     num_stocks = 500 // open_price
            #     account.add_trade(open_price, num_stocks, symbol, date)
            # Buy Condition: New Hammer + High SMA
            if new_hammer != 0 and sma_indicator == 100:
                num_stocks = 300 // open_price
                account.add_trade(open_price, num_stocks, symbol, date)
            
            

        print(f"Simulation completed for {symbol}.")


    # PRINT TRADES 
    account.print_sold_trades()
    # Print final account balance   
    account.print_balance()
    # Print existing trades
    account.print_existing_trades()

    return 0



if __name__ == "__main__":
    main()






        