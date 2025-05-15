import asyncio
from oauth2 import QuestradeAuth
import models.classes as classes
import update_trades
import pandas as pd
import models.strategies as strategies
import warnings
from dotenv import load_dotenv
from pathlib import Path
import os
warnings.filterwarnings("ignore")

env_path = Path('..') / '.env'
load_dotenv(dotenv_path=env_path)
QUESTRADE_REFRESH_TOKEN = os.getenv("QUESTRADE_REFRESH_TOKEN")
account_id = "53455775"

def get_single_stock_data(db, symbol):
    """
    Query database and return all historical data in yfinance-style DataFrame
    Args:
        db: Database connection object
        symbol: Stock symbol (e.g., 'RY.TO')
    Returns:
        pd.DataFrame with all historical OHLCV + adjusted_close data
    """
    try:
        # Query all historical data with parameterized SQL
        query = """
        SELECT date, open, high, low, close, adjusted_close, volume
        FROM daily_prices dp
        JOIN stocks s ON dp.stock_id = s.stock_id
        WHERE s.symbol = %s
        ORDER BY date
        """
        data = db.query(query, (symbol,))
        if not data:
            print(f"No data found for {symbol}")
            return None
            
        # Convert Decimal objects to float and date to datetime
        processed_data = []
        for row in data:
            processed_data.append({
                'Date': pd.to_datetime(row['date']),
                'Open': float(row['open']),
                'High': float(row['high']),
                'Low': float(row['low']),
                'Close': float(row['close']),
                'Adj Close': float(row['adjusted_close']),
                'Volume': int(row['volume'])
            })
        
        # Create DataFrame
        df = pd.DataFrame(processed_data)
        df.set_index('Date', inplace=True)
        
        # Forward-fill any missing values
        df.ffill(inplace=True)
        
        # Validate data
        if df.empty:
            print(f"Empty DataFrame for {symbol}")
            return None
        return df
    
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")
        return None

def analyze_account_positions(account_positions):
    print(f"\n{'='*50}")
    print("Trades")
    print(f"{'='*50}\n")
    for position in account_positions['positions']:
        update_trades.insert_trade(position)
        update_trades.update_trade(position)
        # Extract position data
        symbol = position['symbol']
        current_price = position['currentPrice']
        cost_basis = position['averageEntryPrice']
        quantity = position['openQuantity']
        
        # Create Trade instance with default risk parameters
        trade = classes.Trade(
            buy_position=cost_basis,
            num_of_stocks=quantity,
            stock_name=symbol,
            date="N/A"  # Date not available in the position data
        )
        
        # Check if we should sell
        should_sell = trade.sell(current_price)
        
        # Calculate percentage change
        price_diff = current_price - cost_basis
        percent_change = (price_diff / cost_basis) * 100
        
        # Determine sign symbols
        pnl_sign = "+" if position['openPnl'] >= 0 else ""
        percent_sign = "+" if percent_change >= 0 else ""
        trade_id = update_trades.generate_trade_id(position)
        # Print position analysis
        print(f"\n{symbol}:")
        print(f"  Shares: {quantity}")
        print(f"  Current Price: ${current_price:.2f}")
        print(f"  Original Price: ${cost_basis:.2f}")
        print(f"  Price Change: {percent_sign}{price_diff:.2f} ({percent_sign}{percent_change:.2f}%)")
        print(f"  Position Value: ${position['currentMarketValue']:.2f}")
        print(f"  P/L: {pnl_sign}${position['openPnl']:.2f}")
        print(f"  Trade ID: {trade_id}")
        
        # Recommendation based on sell logic
        if should_sell:
            print("  RECOMMENDATION: SELL")
            print(f"  Reason: Price has triggered one of the sell conditions")
            print(f"  Current risk parameters:")
            print(f"    - Trailing start: {trade.trailing_start*100-100:.1f}% above buy")
            print(f"    - Initial trail: {100-trade.initial_trail_percent*100:.1f}% below peak")
            print(f"    - Tight trail threshold: {trade.tight_trail_threshold*100-100:.1f}% above buy")
            print(f"    - Tight trail: {100-trade.tight_trail_percent*100:.1f}% below peak")
            print(f"    - Hard stop loss: {100-trade.hard_stop_loss*100:.1f}% below buy")
            print(f"    - Profit target: {trade.profit_target*100-100:.1f}% above buy")
        else:
            print("  RECOMMENDATION: HOLD")
            print("  Reason: Price has not triggered any sell conditions")
    print(f"\n{'='*50}")
    print("Trades")
    print(f"{'='*50}\n")

def trade_recommendations():
    """Analyze TSX stocks and generate buy recommendations"""
    # Initialize database
    db = classes.Database(host="localhost", user="postgres", 
                         password="1234", dbname="tradingbot")
    
    # Get all active TSX stocks from database
    stocks = db.query("""
        SELECT stock_id, symbol, company_name 
        FROM stocks 
        WHERE is_active = TRUE 
        ORDER BY symbol
    """)
    print(f"\n{'='*50}")
    print("Analyzing Stocks...")
    print(f"{'='*50}\n")
    
    buy_recommendations = []
    
    for stock in stocks:
        stock_id, symbol, name = stock
        # Get data for single stock
        data = get_single_stock_data(db, stock['symbol'])
        
        if data is None or data.empty:
            print(f"No data for {symbol} - Skipping")
            continue
        
        # Get strategy signal (100 = buy, 0 = hold/sell)
        if strategies.volume_sma_buy_signal(data) or strategies.check_volume_candlestick_buy_signal(data) == 100:
            ma_50 = data['Close'].rolling(50).mean()[-1]
            ma_200 = data['Close'].rolling(200).mean()[-1]
            
            recommendation = {
                'symbol': symbol,
                'name': name,
                'last_close': data['Close'][-1],
                'ma_50': ma_50,
                'ma_200': ma_200
            }
            buy_recommendations.append(recommendation)
            
            print(f"BUY {stock['symbol']} ({stock['company_name']})")
            print(f"Price: ${data['Close'][-1]:.2f} | 50MA: ${ma_50:.2f} | 200MA: ${ma_200:.2f}")
            print("---")
    
    print(f"\n{'='*50}")
    print(f"Analysis Complete - {len(buy_recommendations)} TSX Buy Recommendations")
    print(f"{'='*50}")
    
    return buy_recommendations
def main():

    qt = QuestradeAuth()

    try:
        accounts = qt.make_request("/v1/accounts")
        print("Accounts:", accounts)
    except Exception as e:
        print(f"API Error: {e}")

    try:
        positions = qt.make_request(f"/v1/accounts/{account_id}/positions")
        print("Positions:", positions) 
    except Exception as e:
        print(f"API Error: {e}")

    # analyze_account_positions(positions)
    
    trade_recommendations()

    return 0
    
    

if __name__ == "__main__":
    main()