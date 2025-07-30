from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.live import StockDataStream
from alpaca.common.exceptions import APIError

import asyncio
from functools import partial
import sys
import pandas as pd
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) 
import models.classes as classes
import models.strategies as strategies
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path('../../') / '.env'
load_dotenv(dotenv_path=env_path)
API_KEY = os.getenv('YOUR_API_KEY_ID')
API_SECRET = os.getenv('YOUR_SECRET_KEY')

BASE_URL = "https://paper-api.alpaca.markets"  

DB_CONFIG = {
    "host": "localhost",
    "user": "postgres",
    "password": "1234",
    "dbname": "tradingbot"
}



symbol = "LCID"

async def handle_trade(trade, custom_arg1):

    """Run the sell() method for each active trade"""
    for row in custom_arg1:
        pl_percent = (trade.price - row.buy_position) / row.buy_position * 100
        print(row.sell(trade.price))
        print(f"P/L: {pl_percent:.2f}%")

    """Process each incoming trade"""
    print(f"{trade.symbol} ${trade.price} (Size: {trade.size})")
    
    try:
        # Initialize database connection
        db = classes.Database(**DB_CONFIG)
        
        # Update current price for INACTIVE trades (note: is_active = FALSE)
        db.query(
            "UPDATE trades SET current_price = %s, updated_at = CURRENT_TIMESTAMP "
            "WHERE stock_id = (SELECT stock_id FROM stocks WHERE symbol = %s) "
            "AND is_active = FALSE",
            (trade.price, trade.symbol),
            fetch=False
        )

    except Exception as e:
        print(f"Database error: {e}")
    finally:
        # Ensure database connection is closed
        if 'db' in locals():
            del db

def main():
    active_trades = []
    db = classes.Database(**DB_CONFIG)

    try: 
        # Get ALL ACTIVE trades for this symbol (should be is_active = TRUE)
        trades_data = db.query(
            "SELECT t.*, s.symbol as stock_name FROM trades t JOIN stocks s ON t.stock_id = s.stock_id WHERE s.symbol = %s AND t.is_active = TRUE",
            (symbol,),
            fetch=True
        )
        
        # Create Trade objects for each active position
        for trade_data in trades_data:
            print(f"Loading trade: {trade_data}")
            trade = classes.Trade(
                buy_position=float(trade_data['buy_position']),
                num_of_stocks=trade_data['num_of_stocks'],
                stock_name=trade_data['stock_name'],
                date=trade_data['buy_date']
            )
            active_trades.append(trade)
        
        print(f"Loaded {len(active_trades)} active trades for {symbol}")
            
    except Exception as e:
        print(f"Error: {e}")


    # Initialize stream
    stream = StockDataStream('PKGJHW34RDC8CG26S1OB', 'NDu8J0yVPfo7enmZaTFBPSUpbCVdw1ccWYWExNxI')
    stream.subscribe_trades(
        partial(handle_trade, custom_arg1=active_trades), 
        symbol
    )   
    

    try:
        print(f"Starting {symbol} trade stream...")
        asyncio.run(stream.run())
    except KeyboardInterrupt:
        print("\nStream stopped by user")

if __name__ == "__main__":
    main()
