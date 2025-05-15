import yfinance as yf
import talib as ta
import pandas as pd
import psycopg2
import psycopg2.extras
from typing import List, Dict, Optional

class Trade:
    def __init__(
        self,
        buy_position: float,
        num_of_stocks: int,
        stock_name: str,
        date: str,
        # Risk management parameters (defaults can be overridden)
        trailing_start: float = 1.05,  # Start trailing after 5% gain
        initial_trail_percent: float = 0.98,  # 2% drop from peak triggers sell
        tight_trail_threshold: float = 1.10,  # Tighten trail after 10% gain
        tight_trail_percent: float = 0.99,  # 1% drop triggers sell after threshold
        hard_stop_loss: float = 0.90,  # 10% max loss
        profit_target: float = 1.12,  # 12% profit target
        volatility_threshold: float = 1.0  
    ):
        self.buy_position = buy_position
        self.sell_position = None
        self.current_position = buy_position
        self.num_of_stocks = num_of_stocks
        self.stock_name = stock_name
        self.date = date
        self.highest_position = buy_position
        
        # Risk management parameters
        self.trailing_start = trailing_start
        self.initial_trail_percent = initial_trail_percent
        self.tight_trail_threshold = tight_trail_threshold
        self.tight_trail_percent = tight_trail_percent
        self.hard_stop_loss = hard_stop_loss
        self.profit_target = profit_target
        self.volatility_threshold = volatility_threshold
    
    def buy(self):
        pass  # Your existing buy logic

    def sell(self, new_position: float) -> bool:
        """
        Enhanced multi-condition exit strategy with:
        - Cleaner price tracking
        - Configurable thresholds
        - Removed print statements (let parent handle output)
        - Added validation
        """
        # Validate inputs
        if not isinstance(new_position, (int, float)) or new_position <= 0:
            return False

        # Track current position
        self.current_position = new_position

        # Track highest price seen (for trailing stops)
        self.highest_position = max(self.current_position, self.highest_position)

        # Calculate dynamic thresholds
        stop_loss_price = self.buy_position * self.hard_stop_loss * self.volatility_threshold
        profit_target_price = self.buy_position * self.profit_target
        loose_trail_trigger = self.highest_position * self.initial_trail_percent
        tight_trail_trigger = self.highest_position * self.tight_trail_percent

        # Exit conditions (in order of priority)
        # 1. Hard stop loss
        if self.current_position <= stop_loss_price:
            print(f"Hard stop loss triggered for {self.stock_name} at {self.current_position:.2f} per share")
            self.sell_position = self.current_position
            return True

        # 2. Profit target
        if self.current_position >= profit_target_price:
            print(f"Profit target triggered for {self.stock_name} at {self.current_position:.2f} per share")
            self.sell_position = self.current_position
            return True

        # 3. Tight trailing stop (after reaching larger threshold)
        if (self.highest_position >= self.buy_position * self.tight_trail_threshold and
                self.current_position <= tight_trail_trigger):
            print(f"Tight trailing stop triggered for {self.stock_name} at {self.current_position:.2f} per share")
            self.sell_position = self.current_position
            return True

        # 4. Regular trailing stop (after reaching smaller threshold)
        if (self.highest_position >= self.buy_position * self.trailing_start and
                self.current_position <= loose_trail_trigger):
            print(f"Regular trailing stop triggered for {self.stock_name} at {self.current_position:.2f} per share")
            self.sell_position = self.current_position
            return True

        return False


# Created at the beginning of the program to keep track of all trades and overall position
class Account:
    def __init__(self, balance):
        self.balance = balance
        self.trades = []
        self.position_total = 0
        self.sold_trades = []
        self.pending_trades = []

    def add_trade(self, buy_position, num_of_stocks, stock_name, date, trailing_start, initial_trail_percent, tight_trail_threshold, tight_trail_percent, hard_stop_loss, profit_target, volatility_threshold):
        # check if the balance of the account is greater than the cost of the trade
        if self.balance < buy_position * num_of_stocks:
            return
        # create a trade object and add it to the trades list
        trade = Trade(buy_position=buy_position, 
                      num_of_stocks=num_of_stocks, 
                      stock_name=stock_name, 
                      date=date, 
                      trailing_start=trailing_start, 
                      initial_trail_percent=initial_trail_percent, 
                      tight_trail_threshold=tight_trail_threshold, 
                      tight_trail_percent=tight_trail_percent, 
                      hard_stop_loss=hard_stop_loss, 
                      profit_target=profit_target, 
                      volatility_threshold=volatility_threshold)
        self.trades.append(trade)
        # subtract the cost of the trade from the balance
        self.balance -= buy_position * num_of_stocks
        # print the transaction
        print(f"Bought {num_of_stocks} shares of {stock_name} at {buy_position} per share")
        return
    def add_pending_trade(self, num_of_stocks, stock_name, date):
        self.pending_trades.append((num_of_stocks, stock_name, date))
        return
    def check_trades(self, sell_position, symbol, date):
        for trade in self.trades[:]:  # Iterate over a copy of the list to avoid modification issues
            if trade.stock_name == symbol:  # Added holding period check
                if trade.sell(sell_position) is True:
                    # Update balance
                    self.balance += trade.sell_position * trade.num_of_stocks
                    
                    # Store trade details in sold_trades
                    self.sold_trades.append({
                        "date": trade.date,
                        "sell_date": date,
                        "stock": trade.stock_name,
                        "buy_position": trade.buy_position,
                        "sell_position": trade.sell_position,
                        "num_of_stocks": trade.num_of_stocks,
                        "profit/loss": (trade.sell_position - trade.buy_position) * trade.num_of_stocks
                    })
                    
                    # Print trade details
                    print(f"Sold {trade.num_of_stocks} shares of {trade.stock_name} at {trade.sell_position:.2f} per share. "
                        f"Bought at {trade.buy_position:.2f}. "
                        f"Profit/Loss: {(trade.sell_position - trade.buy_position) * trade.num_of_stocks:.2f}")

                    # Remove the trade from active trades
                    self.trades.remove(trade)
        
        return 0

    
    def print_balance(self):
        # any existing trades, add them to the balance
        for trade in self.trades:
            self.balance += trade.buy_position * trade.num_of_stocks
        print(f"Balance: {self.balance}")
        return
    
    def print_existing_trades(self):
        for trade in self.trades:
            print(f"Trade: {trade.buy_position}, {trade.num_of_stocks}, {trade.stock_name}")
        return
    
    def print_sold_trades(self):
        for trade in self.sold_trades:
            print(
                f"Trade Details: Stock: {trade['stock']}, "
                f"B-Date: {trade['date']}, "
                f"B-Price: {trade['buy_position']:.2f}, "
                f"S-Date: {trade['sell_date']}, "
                f"S-Price: {trade['sell_position']:.2f}, "
                f"Shares: {trade['num_of_stocks']}, "
                f"Profit/Loss: {trade['profit/loss']:.2f}"
            )

        return

    def get_balance(self):
        return self.balance
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