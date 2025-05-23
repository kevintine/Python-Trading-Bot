import models.classes as classes

def generate_trade_id(trade_dict):
    """
    Creates a unique trade ID integer by combining:
    1. symbolId
    2. openQuantity 
    3. averageEntryPrice (with decimal removed)
    """
    symbol_id = str(trade_dict['symbolId'])
    quantity = str(trade_dict['openQuantity'])
    entry_price = str(trade_dict['averageEntryPrice']).replace('.', '')  # Remove decimal
    
    combined = symbol_id + quantity + entry_price
    return int(combined)

def insert_trade(trade_dict: dict) -> dict:
    """
    Inserts a new trade using generate_trade_id() for unique IDs and 
    currentMarketValue as current_price.
    
    Args:
        trade_dict: {
            'symbol': str,               # Required
            'symbolId': int,             # Required for trade ID
            'openQuantity': int,         # -> num_of_stocks
            'averageEntryPrice': float,  # -> buy_position
            'currentMarketValue': float, # -> current_price
            'currentPrice': float,       # Optional fallback
            'totalCost': float           # Optional for notes
        }
    
    Returns:
        Complete trade record with all columns
    
    Raises:
        ValueError: For invalid data or duplicate trades
    """
    # Validate required fields
    if trade_dict['openQuantity'] <= 0:
        raise ValueError("Quantity must be positive")
    
    db = classes.Database(host="localhost", user="postgres",
                         password="1234", dbname="tradingbot")
    
    # 1. Get stock_id
    stock_result = db.query(
        "SELECT stock_id FROM stocks WHERE symbol = %s",
        (trade_dict['symbol'],),
        fetch=True
    )
    if not stock_result:
        raise ValueError(f"Stock {trade_dict['symbol']} not found")
    stock_id = stock_result[0]['stock_id']
    
    # 2. Generate unique trade ID
    trade_id = generate_trade_id(trade_dict)
    
    # 3. Check for existing trade ID (not just same stock/date)
    existing = db.query(
        "SELECT 1 FROM trades WHERE trade_id = %s",
        (trade_id,),
        fetch=True
    )
    if existing:
        return None
        raise ValueError(f"Trade ID {trade_id} already exists") 
    
    # 4. Calculate current_price per share
    current_price = (
        trade_dict.get('currentMarketValue') or  # Primary source
        trade_dict.get('currentPrice')           # Fallback
    )
    if current_price and trade_dict['openQuantity'] > 0:
        current_price = round(current_price / trade_dict['openQuantity'], 4)
    
    # 5. Insert trade with custom ID
    result = db.query(
        """
        INSERT INTO trades (
            trade_id,
            stock_id,
            buy_position,
            num_of_stocks,
            buy_date,
            current_price,
            notes
        ) VALUES (
            %s, %s, %s, %s, CURRENT_DATE, %s, %s
        )
        RETURNING *
        """,
        (
            trade_id,
            stock_id,
            trade_dict['averageEntryPrice'],
            trade_dict['openQuantity'],
            current_price,
            f"Market value: {trade_dict.get('currentMarketValue')} | "
            f"Entry cost: {trade_dict.get('totalCost')}"
        ),
        fetch=True
    )
    
    if not result:
        raise RuntimeError("Failed to insert trade")
    return result[0]

def update_trade(trade_id: int, trade_dict: dict) -> dict:
    pass