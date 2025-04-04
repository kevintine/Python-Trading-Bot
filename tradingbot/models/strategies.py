import yfinance as yf
import talib as ta
import numpy as np
import pandas as pd
from collections import deque
from scipy.signal import argrelextrema
# import extra

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
    df = extra.restore_yfinance_structure(df)
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
    df = extra.restore_yfinance_structure(df)
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
    df = extra.restore_yfinance_structure(df)
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
    df = extra.restore_yfinance_structure(df)
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
    df = extra.restore_yfinance_structure(df)
    df = df.copy()
    
    # Moving averages
    df['SMA50_Vol'] = df['Volume'].rolling(window=50).mean()
    df['SMA50_Price'] = ta.SMA(df['Close'], timeperiod=50)

    # Loosened Volume Surge Condition (5% instead of 10%)
    df['Volume_Surge'] = df['Volume'] > (volume_multiplier * df['SMA50_Vol'])

    # Weaker Close Allowed (top 30% of the day's range instead of 40%)
    df['Close_Strong'] = (df['Close'] > df['Open']) | ((df['Close'] - df['Low']) / (df['High'] - df['Low']) > 0.1)

    # Allowing Price to Be Further From SMA (5% instead of 2%)
    df['Near_SMA50'] = (df['Close'] > df['SMA50_Price'] * 0.85)

    # Buy Signal
    buy_signal = df['Volume_Surge'] & df['Close_Strong'] & df['Near_SMA50']

    return 100 if buy_signal.iloc[-1] else 0
