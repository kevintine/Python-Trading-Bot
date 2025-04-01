import yfinance as yf
import talib as ta
import numpy as np
import pandas as pd
from collections import deque
from alpaca.trading.client import TradingClient
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import sys
import plotly.graph_objects as go
sys.path.append(os.path.abspath('..'))
from models.classes import Signal, Trade, Account
from models.strategies import cdl_hammer_bot, get_engulfing, get_hammer, get_sma_indicator, get_volume_indicator, get_green_candle, detect_dip_signal, detect_supports, check_for_sma_dip, smart_money_trend_strategy, volume_sma_buy_signal
from models.charts import plot_trades

trading_client = TradingClient("PKGJHW34RDC8CG26S1OB", "NDu8J0yVPfo7enmZaTFBPSUpbCVdw1ccWYWExNxI")
stocks = ["AC.TO", "SU.TO", "RY.TO", "TD.TO", "CM.TO"]  
patterns = ["CDLENGULFING", "CDLHAMMER", "CDLSHOOTINGSTAR"] 
start="2023-01-01"
end="2025-01-01"

# ["AC.TO", "SU.TO", "RY.TO", "TD.TO", "CM.TO", "AW.TO", "ENB.TO", "CNQ.TO"]
# ["INTC", "AMD", "MSFT", "AAPL"]
# 10 stocks in the nyse worth between 10-60 dollars
# ["BAC", "T", "PFE", "VZ", "CSCO", "F", "GE", "FCX", "X", "MTCH", "TOST", "TEVA", "AMCR", "AMC", "AMN", "AMD"]
# "BAC", "T", "PFE", "VZ", "CSCO", "F", "GE", 

def fetch_data(symbols):
    full_data = {}
    end_date = pd.Timestamp.today().date()
    start_date = end_date - pd.DateOffset(years=2)

    for symbol in symbols:
        print(f"\nFetching data for {symbol}...")
        data = yf.download(symbol, start=start_date, end=end_date)

        if data.empty:
            print(f"No data retrieved for {symbol}. Skipping...")
        else:
            data['50_day_SMA'] = data['Close'].rolling(window=50).mean()
            full_data[symbol] = data.copy()

    return full_data

def run_simulation(full_data, account):
    lookback_period = 252  # 1-year historical data

    for symbol, data in full_data.items():
        print(f"\nStarting simulation for {symbol}...")

        split_index = len(data) // 2
        historical_data = data.iloc[:split_index].copy()
        simulation_data = data.iloc[split_index:].copy()

        rolling_window = deque(maxlen=lookback_period)
        for _, row in historical_data.iterrows():
            rolling_window.append((row["Open"], row["High"], row["Low"], row["Close"]))

        average_volume = historical_data["Volume"].mean().item()

        for date, row in simulation_data.iterrows():
            rolling_window.append((row["Open"], row["High"], row["Low"], row["Close"]))
            df = pd.DataFrame(rolling_window, columns=["Open", "High", "Low", "Close"])

            open_price = row["Open"].item()
            close_price = row["Close"].item()
            volume = row["Volume"].item()
            sma_50 = row["50_day_SMA"].item()

            ############### Calculate indicators ##################
            engulfing = get_engulfing(df)
            hammer = get_hammer(df)
            sma_indicator = get_sma_indicator(close_price, sma_50)
            volume_indicator = get_volume_indicator(volume, average_volume)
            new_hammer = float(cdl_hammer_bot(df["Open"], df["High"], df["Low"], df["Close"]).iloc[-1])
            green_candle = get_green_candle(close_price, open_price)
            rsi_bollinger_dip = detect_dip_signal(df)
            support = detect_supports(df)
            sma_dip = check_for_sma_dip(df, short_window=25, long_window=50)
            smart_money = smart_money_trend_strategy(df, use_volume=True, use_green_prev=True, use_sma_alignment=True)
            volume_surge = volume_sma_buy_signal(df, volume_multiplier=1.1)
            
            # Run account check trades
            account.check_trades(close_price, symbol, date)

            # if engulfing == 100 and volume_indicator == 100:
            #     num_stocks = 2500 // open_price
            #     account.add_trade(open_price, num_stocks, symbol, date)

            # if hammer != 0 and sma_indicator == 100:
            #     num_stocks = 2500 // open_price
            #     account.add_trade(open_price, num_stocks, symbol, date)

            # if volume_indicator == 100 and sma_indicator == 100:
            #     num_stocks = 2500 // open_price
            #     account.add_trade(open_price, num_stocks, symbol, date)

            # if new_hammer == 100 and volume_indicator == 100:
            #     num_stocks = 2500 // open_price
            #     account.add_trade(open_price, num_stocks, symbol, date)

            # if new_hammer == 100 and sma_indicator == 100:
            #     num_stocks = 2500 // open_price
            #     account.add_trade(open_price, num_stocks, symbol, date)

            # if green_candle != 0 and new_hammer == 100:
            #     num_stocks = 2500 // open_price
            #     account.add_trade(open_price, num_stocks, symbol, date)

            # if rsi_bollinger_dip == 100:
            #     num_stocks = 2500 // open_price
            #     account.add_trade(open_price, num_stocks, symbol, date)

            # if support == 100:
            #     num_stocks = 2500 // open_price
            #     account.add_trade(open_price, num_stocks, symbol, date)
            
            # if sma_dip == 100:
            #     num_stocks = 2500 // open_price
            #     account.add_trade(open_price, num_stocks, symbol, date)
            if volume_surge == 100:
                num_stocks = 2500 // open_price
                account.add_trade(open_price, num_stocks, symbol, date)

        print(f"Simulation completed for {symbol}.")

def main():
    symbols = ["BAC", "T", "PFE", "VZ", "CSCO", "F", "GE", "FCX", "X", "MTCH", "TOST", "TEVA", "AMCR", "AMC", "AMN", "AMD","AC.TO", "SU.TO", "RY.TO", "TD.TO", "CM.TO", "AW.TO", "ENB.TO", "CNQ.TO"]
    account = Account(100000)

    full_data = fetch_data(symbols)  # Step 1: Fetch all data first
    run_simulation(full_data, account)  # Step 2: Run the simulation

    account.print_sold_trades()
    account.print_balance()
    account.print_existing_trades()

if __name__ == "__main__":
    main()


        