import yfinance as yf
import talib as ta
import numpy as np
import pandas as pd
from collections import Counter
# import extra

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

# Hammer function for the trading bot
# the lower shadow and upper shadown multiplier can be adjusted
def cdl_hammer_bot(open, high, low, close, lower_shadow_multiplier=2, upper_shadow_multiplier=1):
    """
    Identifies Hammer and Inverted Hammer candlestick patterns with adjustable size parameters.
    
    Converts Pandas Series to NumPy arrays before processing.

    Parameters:
    - open (pd.Series): Open prices.
    - high (pd.Series): High prices.
    - low (pd.Series): Low prices.
    - close (pd.Series): Close prices.
    - lower_shadow_multiplier (float): Multiplier for the lower shadow size.
    - upper_shadow_multiplier (float): Multiplier for the upper shadow size.

    Returns:
    - pd.Series: 100 for Hammer, -100 for Inverted Hammer, 0 otherwise.
    """
    # Convert Pandas Series to NumPy arrays
    open_values = np.array([day.item() for day in open])
    high_values = np.array([day.item() for day in high])
    low_values = np.array([day.item() for day in low])
    close_values = np.array([day.item() for day in close])

    # Calculate the body and shadow sizes
    body_size = np.abs(close_values - open_values)  # Real body size
    upper_shadow = high_values - np.maximum(open_values, close_values)  # Upper wick
    lower_shadow = np.minimum(open_values, close_values) - low_values  # Lower wick
    candle_range = high_values - low_values  # Total candle range

    # Hammer condition: Small body, long lower wick, small upper wick
    is_hammer = (lower_shadow >= lower_shadow_multiplier * body_size) & (upper_shadow < upper_shadow_multiplier * body_size)

    # Inverted Hammer condition: Small body, long upper wick, small lower wick
    is_inverted_hammer = (upper_shadow >= upper_shadow_multiplier * body_size) & (lower_shadow < lower_shadow_multiplier * body_size)

    # Assign values: 100 for Hammer, -100 for Inverted Hammer, 0 otherwise
    pattern = np.where(is_hammer, 100, np.where(is_inverted_hammer, -100, 0))

    return pd.Series(pattern, index=open.index)

# Hammer function for the web app
# the lower shadow and upper shadown multiplier can be adjusted
def cdl_hammer_web(open_series, high_series, low_series, close_series, lower_shadow_multiplier=2, upper_shadow_multiplier=1):
    """
    Identifies Hammer and Inverted Hammer candlestick patterns using individual Pandas Series.

    Parameters:
    - open_series (pd.Series): Open prices.
    - high_series (pd.Series): High prices.
    - low_series (pd.Series): Low prices.
    - close_series (pd.Series): Close prices.
    - lower_shadow_multiplier (float): Multiplier for the lower shadow size.
    - upper_shadow_multiplier (float): Multiplier for the upper shadow size.

    Returns:
    - pd.Series: 100 for Hammer, -100 for Inverted Hammer, 0 otherwise.
    """

    # Calculate body and shadow sizes
    body_size = (close_series - open_series).abs()  # Real body size
    upper_shadow = high_series - pd.concat([open_series, close_series], axis=1).max(axis=1)  # Upper wick
    lower_shadow = pd.concat([open_series, close_series], axis=1).min(axis=1) - low_series  # Lower wick

    # Apply multipliers
    body_size_scaled_lower = lower_shadow_multiplier * body_size
    body_size_scaled_upper = upper_shadow_multiplier * body_size

    # Define hammer conditions
    is_hammer = (lower_shadow >= body_size_scaled_lower) & (upper_shadow < body_size_scaled_upper)
    is_inverted_hammer = (upper_shadow >= body_size_scaled_upper) & (lower_shadow < body_size_scaled_lower)

    # Assign values: 100 for Hammer, -100 for Inverted Hammer, 0 otherwise
    pattern = 100 * is_hammer - 100 * is_inverted_hammer

    return pattern


def get_engulfing(df):
    """Calculate the engulfing candlestick pattern for the last row."""
    return float(ta.CDLENGULFING(df["Open"], df["High"], df["Low"], df["Close"]).iloc[-1])

def get_hammer(df):
    """Calculate the hammer candlestick pattern for the last row."""
    return float(ta.CDLHAMMER(df["Open"], df["High"], df["Low"], df["Close"]).iloc[-1])

def get_volume_indicator(volume, average_volume, threshold=1.4):
    """Return 100 if volume is significantly higher than average."""
    return 100 if volume > (threshold * average_volume) else 0

def get_sma_indicator(close_price, sma_50):
    """Return 100 if close is above 50-day SMA."""
    return 100 if close_price > sma_50 else 0

def get_green_candle(close_price, open_price):
    """Return 100 if the last candle is green."""
    return 100 if close_price > open_price else 0

def detect_dip_signal(df):
    # Ensure required columns exist
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("DataFrame must contain 'Open', 'High', 'Low', 'Close' columns.")

    # Calculate indicators
    sma_50 = ta.SMA(df['Close'], timeperiod=50)
    rsi = ta.RSI(df['Close'], timeperiod=14)
    upper, middle, lower = ta.BBANDS(df['Close'], timeperiod=20)

    # Extract the latest scalar values
    close = df['Close'].iloc[-1].item()
    lower_band = lower.iloc[-1].item()
    rsi_value = rsi.iloc[-1].item()
    sma_value = sma_50.iloc[-1].item()

    # Dip signal condition
    if close < lower_band and rsi_value < 35 and close > (0.9 * sma_value):
        return 100  # Dip detected, possible buy signal
    else:
        return 0  # No dip
    

def detect_supports(df):
    df = restore_yfinance_structure(df)
    if 'Low' not in df.columns or 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Low' and 'Close' columns.")

    # Detect support levels (local minima)
    supports = df[df['Low'] == df['Low'].rolling(5, center=True).min()]['Low']

    # Filter out levels that are too close to each other (within $2 difference)
    supports = supports[abs(supports.diff()) > 2]

    # Get the last detected support level
    if not supports.empty:
        last_support_price = supports.iloc[-1]
        current_close = df['Close'].iloc[-1]

        # Check if current price is at least 5% above the last support
        if current_close >= last_support_price * 1.05:
            return 100  # Buy signal
    return 0  # No signal

def check_for_sma_dip(df, short_window=20, long_window=50):
    df = restore_yfinance_structure(df)
    df = df.copy()
    df['SMA_short'] = df['Close'].rolling(window=short_window).mean()
    df['SMA_long'] = df['Close'].rolling(window=long_window).mean()

    # Get the last row and the one before for comparison
    last = df.iloc[-1]
    prev = df.iloc[-2]

    # Dip condition:
    # - Current close below short SMA
    # - Previous close was above short SMA
    # - Long SMA is trending upward
    if (
        last['Close'] < last['SMA_short']
        and prev['Close'] > prev['SMA_short']
        and last['SMA_long'] > prev['SMA_long']
    ):
        return 100
    else:
        return 0

def smart_money_trend_strategy(df, use_volume=True, use_green_prev=True, use_sma_alignment=True):
    df = restore_yfinance_structure(df)
    df = df.copy()

    df['SMA10'] = df['Close'].rolling(10).mean()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['Volume_avg'] = df['Volume'].rolling(20).mean()

    if len(df) < 60:
        return 0  # Not enough data

    last = df.iloc[-1]
    prev = df.iloc[-2]

    # 🔹 Buy on strength, not weakness
    breakout = last['Close'] > last['SMA20'] and last['SMA20'] > last['SMA50']
    strong_candle = last['Close'] > last['Open']  # Green candle
    volume_ok = last['Volume'] >= 1.2 * last['Volume_avg']
    green_prev = prev['Close'] > prev['Open']
    sma_trending_up = df['SMA50'].iloc[-1] > df['SMA50'].iloc[-5]

    # 🔹 New entry conditions (reversed logic)
    conditions = [breakout, strong_candle]
    if use_volume:
        conditions.append(volume_ok)
    if use_green_prev:
        conditions.append(green_prev)
    if use_sma_alignment:
        conditions.append(sma_trending_up)

    if all(conditions):
        return 100  # Buy Signal
    return 0  # No trade

def volume_sma_buy_signal(df, volume_multiplier=1.5):
    """
    Buy Signal Conditions:
    - Price is above 50-SMA (strong trend)
    - Price crosses above 20-SMA (momentum shift)
    - Volume is at least 1.5x the average 20-day volume (high interest)

    Returns:
    - 100 for a Buy Signal
    - 0 for No Trade
    """
    # df = extra.restore_yfinance_structure(df)
    df = df.copy()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['Volume_avg'] = df['Volume'].rolling(20).mean()

    if len(df) < 60:
        return 0  # Not enough data

    last = df.iloc[-1]
    prev = df.iloc[-2]

    buy_signal = (
        last['Close'] > last['SMA50'] and  # Uptrend confirmation
        prev['Close'] < prev['SMA20'] and last['Close'] > last['SMA20'] and  # Breakout above 20-SMA
        last['Volume'] >= volume_multiplier * last['Volume_avg']  # Strong volume surge
    )

    return 100 if buy_signal else 0

def check_volume_candlestick_buy_signal(df, volume_multiplier=1.5, lookback_days=10, min_bullish_patterns=2):
    """
    Enhanced version with multi-day candlestick pattern confirmation
    
    Args:
        df: DataFrame with OHLCV data
        volume_multiplier: Volume spike threshold (default 1.5x average)
        lookback_days: Number of days to analyze for patterns (default 5)
        min_bullish_patterns: Minimum bullish patterns required (default 2)
    
    Returns:
        100 for strong buy signal, 0 otherwise
    """
    # Error handling: Check for sufficient data
    if df.empty or len(df) < lookback_days + 20:
        return 0
        
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_columns):
        return 0

    df = restore_yfinance_structure(df)
    df = df.copy().iloc[-lookback_days-20:]  # Ensure enough data for averages
    # 1. Volume Analysis (more robust with std dev)
    avg_volume = df['Volume'].rolling(window=20).mean().iloc[-1]
    std_volume = df['Volume'].rolling(window=20).std().iloc[-1]
    current_volume = df['Volume'].iloc[-1]
    
    # Volume spike = current > avg + 1std (configurable via multiplier)
    volume_spike = current_volume > (avg_volume + (std_volume * volume_multiplier))
    
    # 2. Multi-Day Candlestick Pattern Detection
    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    
    # Analyze last N days (excluding today)
    pattern_counts = Counter()
    for i in range(-lookback_days-1, -1):
        day_slice = slice(i-3, i+1) if i < -1 else slice(i-3, None)
        
        # Check all major bullish patterns
        if ta.CDLENGULFING(opens[day_slice], highs[day_slice], lows[day_slice], closes[day_slice])[-1] > 0:
            pattern_counts['engulfing'] += 1
        if ta.CDLHAMMER(opens[day_slice], highs[day_slice], lows[day_slice], closes[day_slice])[-1] > 0:
            pattern_counts['hammer'] += 1
        if ta.CDLMORNINGSTAR(opens[day_slice], highs[day_slice], lows[day_slice], closes[day_slice])[-1] > 0:
            pattern_counts['morning_star'] += 1
        if ta.CDLPIERCING(opens[day_slice], highs[day_slice], lows[day_slice], closes[day_slice])[-1] > 0:
            pattern_counts['piercing'] += 1
        if ta.CDL3WHITESOLDIERS(opens[day_slice], highs[day_slice], lows[day_slice], closes[day_slice])[-1] > 0:
            pattern_counts['3_white_soldiers'] += 1
    
    # 3. Trend Confirmation (5-day SMA rising)
    df['SMA5'] = df['Close'].rolling(window=5).mean()
    sma_rising = df['SMA5'].iloc[-1] > df['SMA5'].iloc[-2]
    
    # 4. Generate Signal
    if (volume_spike and 
        sum(pattern_counts.values()) >= min_bullish_patterns and 
        sma_rising):
        return 100
    return 0

def swing_trade_signal(df):
    """
    Generate swing trade signals with looser requirements (returns 100 or 0 only)
    Args:
        df (DataFrame): OHLCV data with columns ['Open','High','Low','Close','Volume']
    Returns:
        int: 100 (buy) or 0 (no trade)
    """
    df = restore_yfinance_structure(df)
    if len(df) < 7:  # Reduced minimum data requirement
        return 0
    
    close = df['Close'].iloc[-1]
    atr = df['High'].iloc[-1] - df['Low'].iloc[-1]  # Simple volatility measure
    
    # Loosened Signal Conditions (Only need 3 out of 5 to trigger)
    conditions_met = sum([
        close > df['Close'].rolling(5).mean().iloc[-1],  # Price above 5-day MA
        df['Volume'].iloc[-1] > df['Volume'].rolling(5).mean().iloc[-1] * 0.8,  # Volume not weak
        (close - df['Low'].iloc[-1]) < (0.4 * atr),  # Closed in top 60% of range
        df['Close'].iloc[-1] > df['Open'].iloc[-1] * 0.995,  # Not a strong red candle
        df['Close'].iloc[-1] > df['Close'].rolling(3).mean().iloc[-2]  # Upward momentum
    ])
    
    return 100 if conditions_met >= 3 else 0

def ma_crossover_strategy(df):
    """
    Generates buy signals (100) when:
    1. 20SMA crosses above 50SMA
    2. Volume > 1.5x 20-day average volume
    3. Price closes above both MAs (confirmation)
    
    Args:
        df (DataFrame): OHLCV data with columns ['Open','High','Low','Close','Volume']
    Returns:
        int: 100 (buy) or 0 (no trade)
    """
    if len(df) < 50:  # Need at least 50 periods for 50SMA
        return 0
    
    # Calculate indicators
    df['20sma'] = df['Close'].rolling(20).mean()
    df['50sma'] = df['Close'].rolling(50).mean()
    df['vol_20ma'] = df['Volume'].rolling(20).mean()
    
    current = df.iloc[-1]
    prev = df.iloc[-2]
    
    # 1. MA Crossover Condition
    ma_crossover = (
        (current['20sma'] > current['50sma']) and 
        (prev['20sma'] <= prev['50sma'])  # Cross just happened
    )
    
    # 2. Volume Spike Condition (1.5x average)
    volume_condition = current['Volume'] > 1.3 * current['vol_20ma']
    
    # 3. Price Confirmation
    price_condition = current['Close'] > current['20sma']
    
    # All conditions must be met
    if ma_crossover and volume_condition and price_condition:
        return 100
    
    return 0



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

def check_volume_spike(df):
    """
    Returns 100 if current volume > 1.2x the 5-day average volume,
    otherwise returns 0.
    Handles edge cases (NaN values, insufficient data).
    """
    # Get current volume (most recent)
    current_volume = df['Volume'].iloc[-1]
    
    # Calculate 5-day average (excluding current day)
    avg_5day = df['Volume'].iloc[-6:-1].mean()  # Looks at previous 5 days
    
    # If not enough data (NaN), return 0
    if pd.isna(avg_5day):
        return 0
    
    # Check if current volume > 1.2x average
    if current_volume > (1.9 * avg_5day):
        return 100
    else:
        return 0
   


