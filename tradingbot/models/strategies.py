import yfinance as yf
import talib as ta
import numpy as np
import pandas as pd
from collections import deque
    

# Takes a Pandas DataFrame and returns a Pandas Series
import numpy as np
import pandas as pd

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


import pandas as pd

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



