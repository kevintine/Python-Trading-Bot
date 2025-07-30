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
            'symbol': str,
            'symbolId': int,
            'openQuantity': int,
            'averageEntryPrice': float,
            'currentMarketValue': float,
            'currentPrice': float,
            'totalCost': float
        }

    Returns:
        Complete trade record with all columns

    Raises:
        ValueError: For invalid data or duplicate trades
    """
    if trade_dict['openQuantity'] <= 0:
        raise ValueError("Quantity must be positive")

    db = classes.Database(
        host="localhost", user="postgres",
        password="1234", dbname="tradingbot"
    )

    # 1. Get stock_id
    stock_result = db.query(
        "SELECT stock_id FROM stocks WHERE symbol = %s",
        (trade_dict['symbol'],),
        fetch=True
    )
    if not stock_result:
        raise ValueError(f"Stock {trade_dict['symbol']} not found")
    stock_id = stock_result[0]['stock_id']

    # 2. Generate trade_id
    trade_id = generate_trade_id(trade_dict)

    # 3. Check for existing trade
    existing = db.query(
        "SELECT 1 FROM trades WHERE trade_id = %s",
        (trade_id,),
        fetch=True
    )
    if existing:
        return None

    # 4. Calculate current price per share
    current_price = (
        trade_dict.get('currentMarketValue') or
        trade_dict.get('currentPrice')
    )
    if current_price and trade_dict['openQuantity'] > 0:
        current_price = round(current_price / trade_dict['openQuantity'], 4)

    # 5. Insert new trade (no 'notes' column)
    result = db.query(
        """
        INSERT INTO trades (
            trade_id,
            stock_id,
            buy_position,
            num_of_stocks,
            buy_date,
            current_price
        ) VALUES (
            %s, %s, %s, %s, CURRENT_DATE, %s
        )
        RETURNING *
        """,
        (
            trade_id,
            stock_id,
            trade_dict['averageEntryPrice'],
            trade_dict['openQuantity'],
            current_price
        ),
        fetch=True
    )


    if not result:
        raise RuntimeError("Failed to insert trade")
    return result[0]


def update_trade(trade_dict: dict) -> dict:
    trade_id = generate_trade_id(trade_dict)
    db = classes.Database(
        host="localhost", user="postgres",
        password="1234", dbname="tradingbot"
    )
    db.query(
        """
        UPDATE trades SET
            current_price = %s
        WHERE trade_id = %s
        """,
        (
            trade_dict['currentPrice'],
            trade_id
        ),
        fetch=False
    )
    return 0

