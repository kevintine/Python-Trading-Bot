import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import psycopg2
import psycopg2.extras
from typing import List, Dict, Optional

class Database:
    def __init__(self, host: str, user: str, password: str, dbname: str, port: int = 5432):
        """
        Simple PostgreSQL database wrapper
        
        Args:
            host: Database host
            user: Database username
            password: Database password
            dbname: Database name
            port: Database port (default: 5432)
        """
        self.connection_params = {
            'host': host,
            'user': user,
            'password': password,
            'dbname': dbname,
            'port': port
        }
    
    def query(self, sql: str, params: Optional[tuple] = None, fetch: bool = True) -> Optional[List[Dict]]:
        """
        Execute a SQL query and return results
        
        Args:
            sql: SQL query string
            params: Tuple of query parameters
            fetch: Whether to return results (use False for write operations without RETURNING)
            
        Returns:
            List of dictionaries (for SELECT/RETURNING) or None
        """
        conn = None
        try:
            conn = psycopg2.connect(**self.connection_params)
            with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute(sql, params)
                
                if fetch:
                    result = [dict(row) for row in cursor.fetchall()]
                    conn.commit()  # MUST commit before returning for RETURNING queries
                    return result
                else:
                    conn.commit()  # Commit write operations
                    return None
                    
        except Exception as e:
            print(f"Database error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

def plot_balance_over_time(trade_log):
    df = pd.DataFrame(trade_log)
    df = df.sort_values(by='sell_date')
    df['cumulative_balance'] = df['profit/loss'].cumsum() + 50000  # Starting balance
    plt.figure(figsize=(12, 6))
    plt.plot(df['sell_date'], df['cumulative_balance'], marker='o')
    plt.title("Account Balance Over Time")
    plt.xlabel("Date")
    plt.ylabel("Balance ($)")
    plt.grid(True)
    plt.show()

def plot_trade_candlestick(stock, buy_date, sell_date, buy_price, sell_price):
    # Create DB connection
    db = Database(
        host="localhost",
        user="postgres",
        password="1234",
        dbname="tradingbot"
    )

    query = """
        SELECT dp.date, dp.open, dp.high, dp.low, dp.close 
        FROM daily_prices dp
        JOIN stocks s ON s.stock_id = dp.stock_id
        WHERE s.symbol = %s
        AND dp.date BETWEEN %s AND %s
        ORDER BY dp.date
    """
    rows = db.query(query, (stock, buy_date, sell_date))
    if not rows:
        print(f"No data found for {stock} between {buy_date} and {sell_date}")
        return

    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close"])

    fig = go.Figure(data=[go.Candlestick(
        x=df['date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name="Price"
    )])

    fig.add_trace(go.Scatter(
        x=[buy_date],
        y=[buy_price],
        mode='markers',
        marker=dict(color='green', size=12),
        name='Buy'
    ))
    fig.add_trace(go.Scatter(
        x=[sell_date],
        y=[sell_price],
        mode='markers',
        marker=dict(color='red', size=12),
        name='Sell'
    ))

    fig.update_layout(
        title=f"{stock} Trade from {buy_date} to {sell_date}",
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )

    fig.show()