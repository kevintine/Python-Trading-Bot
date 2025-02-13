# This is where the bot will run
from alpaca.trading.client import TradingClient
import yfinance as yf
import talib as ta
import numpy as np
import pandas as pd

from models.trade import Trade, Signal, Account

trading_client = TradingClient("PKGJHW34RDC8CG26S1OB", "NDu8J0yVPfo7enmZaTFBPSUpbCVdw1ccWYWExNxI")
stocks = ["AC.TO", "SU.TO", "RY.TO", "TD.TO", "CM.TO"]  
patterns = ["CDLENGULFING", "CDLHAMMER", "CDLSHOOTINGSTAR"] 
start="2023-01-01"
end="2025-01-01"

def main():
    # Define multiple stocks and patterns to trade
    stocks = ["AC.TO", "SU.TO", "RY.TO", "TD.TO", "T.TO"]  
    patterns = ["CDLENGULFING", "CDL3WHITESOLDIERS", "CDLSHOOTINGSTAR"] 
    
    start = "2023-01-01"
    end = "2025-01-01"
    
    # Initialize account with a starting balance
    account = Account(balance=10000)

     # Loop over each stock and each pattern
    for stock in stocks:
        for pattern in patterns:
            signal = Signal(stock, pattern, start, end)
            trade = Trade(signal)
            account.add_trade(trade)  # Add trade to account

    account.run()

    return 0


if __name__ == "__main__":
    main()

## TODO ##
# We need to have so that once we create and give a signal, the account will run and initiate a trade everytime the signal is hit. 
# A new trade is created when a buy indicator is created