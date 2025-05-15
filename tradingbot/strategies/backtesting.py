import yfinance as yf
import talib as ta
import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, Generator, Tuple, Union
from alpaca.trading.client import TradingClient
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent  # Adjust based on your file location
sys.path.append(str(project_root))

from models.classes import Trade, Account, Database
from models.strategies import cdl_hammer_bot, get_engulfing, get_hammer, get_sma_indicator, get_volume_indicator, get_green_candle, detect_dip_signal, detect_supports, check_for_sma_dip, smart_money_trend_strategy, volume_sma_buy_signal, check_volume_candlestick_buy_signal, swing_trade_signal, ma_crossover_strategy, check_volume_spike

start="2023-01-01"
end="2025-01-01"

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

def get_single_stock_data_intraday(db, symbol):
    """
    Query database and return all historical data in yfinance-style DataFrame
    Args:
        db: Database connection object
        symbol: Stock symbol (e.g., 'RY.TO')
    Returns:
        pd.DataFrame with all historical OHLCV + adjusted_close data
    """
    try:
        myQuery = """
        SELECT 
            datetime as date,
            open,
            high,
            low,
            close,
            adjusted_close,
            volume
        FROM intraday_60min_prices ip
        JOIN stocks s ON ip.stock_id = s.stock_id
        WHERE s.symbol = %s
        ORDER BY datetime
        """
        df = db.query(myQuery, (symbol,), fetch=True)
        processed_data = []
        for row in df:
            processed_data.append({
                'Date': pd.to_datetime(row['date']),
                'Open': float(row['open']),
                'High': float(row['high']),
                'Low': float(row['low']),
                'Close': float(row['close']),
                'Adj Close': 0,
                'Volume': int(row['volume'])
            })
        
        df = pd.DataFrame(processed_data)
        df.set_index('Date', inplace=True)

        df.ffill(inplace=True)

        if df.empty:
            print(f"Empty DataFrame for {symbol}")
            return None
        
        return df
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")
        return None
    
def fetch_data(db, symbols: Union[str, list]) -> Dict[str, pd.DataFrame]:
    """
    Fetch intraday data for multiple stocks and return in yfinance-style format
    
    Args:
        db: Database connection object
        symbols: Single symbol (str) or list of symbols
        
    Returns:
        Dictionary where keys are symbols and values are DataFrames with OHLCV data
        (identical format to yfinance's tickers.history() output)
    """
    # Convert single symbol to list for uniform processing
    if isinstance(symbols, str):
        symbols = [symbols]
    
    data_dict = {}
    
    for symbol in symbols:
        try:
            df = get_single_stock_data_intraday(db, symbol)
            if df is not None and not df.empty:
                data_dict[symbol] = df
        except Exception as e:
            print(f"Skipping {symbol} due to error: {str(e)}")
            continue
    
    return data_dict
        
def fetch_data_daily(db, symbols: Union[str, list]) -> Dict[str, pd.DataFrame]:
    # Convert single symbol to list for uniform processing
    if isinstance(symbols, str):
        symbols = [symbols]
    
    data_dict = {}
    
    for symbol in symbols:
        try:
            df = get_single_stock_data(db, symbol)
            if df is not None and not df.empty:
                data_dict[symbol] = df
        except Exception as e:
            print(f"Skipping {symbol} due to error: {str(e)}")
            continue
    
    return data_dict

def run_simulation(full_data, account, lookback_period=252, base_position_percentage=0.10):
    """
    OHLCV Simulation with Simplified Position Management
    - Maintains complete OHLCV data
    - Uses account.check_trades() for position management
    - Progressive position sizing
    """
    
    # 1. Configuration
    BALANCE_TIERS = {
        500: 1.00,    # All-in below $500
        1000: 0.50,   # 50% of balance $500-$1000
        2000: 0.30,   # 30% of balance $1000-$2000
        5000: 0.15,   # 25% of balance $1000-$5000
        10000: 0.1,  # 15% of balance $5000-$10000
        float('inf'): base_position_percentage
    }

    # 2. Data Preparation with Flexible Lookback
    symbol_data = {}
    min_start_date = None
    
    for symbol, data in full_data.items():
        # Determine available data range
        available_days = len(data)
        usable_days = min(available_days, lookback_period)
        
        # Split into warmup and simulation periods
        simulation_data = data.iloc[-usable_days:].copy()
        warmup_data = data.iloc[:-usable_days] if available_days > usable_days else pd.DataFrame()
        
        # Track earliest simulation start date across all symbols
        current_start = simulation_data.index[0]
        min_start_date = current_start if (min_start_date is None or current_start < min_start_date) else min_start_date
        
        # Initialize rolling OHLCV window
        rolling_window = deque(maxlen=lookback_period)
        for _, row in warmup_data.iterrows():
            rolling_window.append((
                row['Open'], row['High'],
                row['Low'], row['Close'],
                row['Volume']
            ))
        
        symbol_data[symbol] = {
            'simulation_data': simulation_data,
            'rolling_window': rolling_window,
            'start_date': current_start
        }

   # 3. Daily Processing
    all_dates = sorted(set().union(*[
        data['simulation_data'].index 
        for data in symbol_data.values()
    ]))
    
    # Modified to use index-based access for next-day execution
    for i in range(len(all_dates) - 1):  # Stop one day earlier
        current_date = all_dates[i]
        next_date = all_dates[i + 1]
        
        print(f"\n{current_date.strftime('%Y-%m-%d')} | Balance: ${account.balance:,.2f}")
        # FIRST: Check existing positions
        for symbol, data in symbol_data.items():
            if current_date in data['simulation_data'].index:
                current_close = data['simulation_data'].loc[current_date]['Close'].item()
                account.check_trades(current_close, symbol, current_date)

        # SECOND: Generate signals for NEXT DAY execution
        for symbol, data in symbol_data.items():
            if current_date not in data['simulation_data'].index:
                continue
                
            row = data['simulation_data'].loc[current_date]
            data['rolling_window'].append((
                row['Open'], row['High'],
                row['Low'], row['Close'],
                row['Volume']
            ))
            
            # Generate signals using CURRENT day's data
            df = pd.DataFrame(data['rolling_window'], 
                           columns=["Open", "High", "Low", "Close", "Volume"])
            
            if ma_crossover_strategy(df) == 100:
                # Verify next day exists for this symbol
                if next_date in data['simulation_data'].index:
                    next_row = data['simulation_data'].loc[next_date]
                    
                    # Position sizing using current balance
                    current_balance = account.balance
                    for threshold, percentage in sorted(BALANCE_TIERS.items()):
                        if current_balance < threshold:
                            position_value = current_balance * percentage
                            break
                    
                    shares = int(position_value // next_row['Open'])  # Use next day's close price
                    if shares >= 1:
                        account.add_trade(
                            position=next_row['Open'],  # Execute at next day's close
                            num_of_stocks=shares,
                            stock_name=symbol,
                            date=next_date,
                            profit_target=1.15,
                            hard_stop_loss=0.82,
                            tight_trail_percent=0,
                            tight_trail_threshold=0,
                            trailing_start=0,
                            initial_trail_percent=0

                        )
                        print(f"Signal on {current_date.date()}: "
                              f"Bought {shares} {symbol} @ {next_row['Open']:.2f} on {next_date.date()}")

    # 4. Results
    print("\n=== Simulation Complete ===")
    print(f"Final Balance: ${account.balance:,.2f}")
    print(f"Total Trades: {len(account.sold_trades)}")

def simulation(intraday_data, daily_data, account, lookback_period=252*2, position_percentage=0.33):
    """
    Simulate trading through all available intraday data hour by hour
    
    Args:
        data_dict: Dictionary of {symbol: DataFrame} with 1-hour OHLCV data
        account: Account object with balance and positions
        
    Returns:
        Modified account object after simulation
    """
    # Combine and sort all data
    all_intraday_data = []
    for symbol, df in intraday_data.items():
        df['Symbol'] = symbol
        all_intraday_data.append(df)

    all_daily_data = []
    for symbol, df in daily_data.items():
        df['Symbol'] = symbol
        all_daily_data.append(df)
    
    combined_intraday_data = pd.concat(all_intraday_data).sort_index()
    combined_daily_data = pd.concat(all_daily_data).sort_index()
    
    combined = pd.concat([combined_intraday_data, combined_daily_data])

    # Calculate available days by checking date range
    available_days = (combined_intraday_data.index[-1] - combined_intraday_data.index[0]).days
    
    if available_days < lookback_period:
        print(f"Error: Requested lookback ({lookback_period} days) exceeds available intraday data ({available_days} days)")
        return account
    
    # Apply lookback period to the COMBINED data
    # To this:
    # Combine all intraday data first
    combined_intraday_data = pd.concat(all_intraday_data).sort_index()

    # Then get dates and filter
    end_date = combined_intraday_data.index[-1]
    start_date = end_date - pd.Timedelta(days=lookback_period)
    filtered_intraday = combined_intraday_data.loc[start_date:end_date]  # Filter the concatenated DataFrame

    for timestamp, hour_data in filtered_intraday.groupby(filtered_intraday.index):
            # Print every timestamp being processed
            print(f"\nProcessing {timestamp}")
            
            # Check if this is the first trading hour of the day (typically 9:30 AM)
            if timestamp.time() == pd.Timestamp('09:30:00').time():
                
                # Get previous day's data
                prev_day = timestamp.date() - pd.Timedelta(days=1)
                daily_mask = (combined_daily_data.index.date == prev_day)
                daily_prices = combined_daily_data[daily_mask]
                
                for symbol, symbol_data in hour_data.groupby('Symbol'):

                    current_price = symbol_data['Close'].iloc[0]
                    # Get number of shares to buy
                    position_value = account.balance * position_percentage
                    shares = int(position_value // current_price)
                    # Get full daily history
                    symbol_full_daily = combined_daily_data[
                        (combined_daily_data['Symbol'] == symbol) & 
                        (combined_daily_data.index.date <= prev_day)
                    ]
                    
                    # Generate signal (only at market open)
                    signal = check_volume_candlestick_buy_signal(
                        symbol_full_daily, 
                        volume_multiplier=1.3, 
                        lookback_days=10, 
                        min_bullish_patterns=2
                    )

                    signal2 = volume_sma_buy_signal(symbol_full_daily, volume_multiplier=1.5)
                    
                    if signal == 100 or signal2 == 100:
                        print(f"BUY {symbol} at {current_price}")
                        account.add_trade(
                            buy_position=current_price,
                            num_of_stocks=shares,
                            stock_name=symbol,
                            date=timestamp,
                            profit_target=1.1,
                            trailing_start=1.05,
                            initial_trail_percent=0.98,
                            tight_trail_threshold=1.10,
                            tight_trail_percent=0.99,
                            hard_stop_loss=0.90,
                            volatility_threshold=1.0
                        )
            
            # Still check positions every hour
            for symbol, symbol_data in hour_data.groupby('Symbol'):
                current_price = symbol_data['Close'].iloc[0]
                account.check_trades(current_price, symbol, timestamp)
        
    return account

    
    
def main():
    db = Database(host="localhost", user="postgres", 
                password="1234", dbname="tradingbot")
    stock_list = db.query("SELECT * FROM stocks WHERE is_active = true")
    all_symbols = [stock['symbol'] for stock in stock_list]
    account = Account(2000)

    intraday_data = fetch_data(db, all_symbols)
    daily_data = fetch_data_daily(db, all_symbols)

    # run_simulation(running_data, account, lookback_period=252, base_position_percentage=0.05) 
    simulation(intraday_data, daily_data, account, lookback_period=100)
    account.print_sold_trades()
    account.print_balance()
    account.print_existing_trades()

if __name__ == "__main__":
    main()


        