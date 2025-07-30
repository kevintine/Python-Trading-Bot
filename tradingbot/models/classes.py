import yfinance as yf
import talib as ta
import pandas as pd
import psycopg2
import psycopg2.extras
from typing import List, Dict, Optional
from .account_visualization import plot_trade_candlestick
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import mplfinance as mpf
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import mplfinance as mpf
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc 

class Trade:
    HARD_STOP = 1
    PROFIT_TARGET = 2
    TIGHT_TRAILING = 3
    REGULAR_TRAILING = 4
    NO_SELL = 0

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
            return (False, Trade.NO_SELL)

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
            return (True, Trade.HARD_STOP)

        # 2. Profit target
        if self.current_position >= profit_target_price:
            print(f"Profit target triggered for {self.stock_name} at {self.current_position:.2f} per share")
            self.sell_position = self.current_position
            return (True, Trade.PROFIT_TARGET)

        # 3. Tight trailing stop (after reaching larger threshold)
        if (self.highest_position >= self.buy_position * self.tight_trail_threshold and
                self.current_position <= tight_trail_trigger):
            print(f"Tight trailing stop triggered for {self.stock_name} at {self.current_position:.2f} per share")
            self.sell_position = self.current_position
            return (True, Trade.TIGHT_TRAILING)

        # 4. Regular trailing stop (after reaching smaller threshold)
        if (self.highest_position >= self.buy_position * self.trailing_start and
                self.current_position <= loose_trail_trigger):
            print(f"Regular trailing stop triggered for {self.stock_name} at {self.current_position:.2f} per share")
            self.sell_position = self.current_position
            return (True, Trade.REGULAR_TRAILING)

        return (False, Trade.NO_SELL)


# Created at the beginning of the program to keep track of all trades and overall position
class Account:
    def __init__(self, balance):
        self.balance = balance
        self.trades = []
        self.position_total = 0
        self.sold_trades = []
        self.pending_trades = []
        self.hard_stops = 0
        self.profit_targets = 0
        self.tight_trails = 0
        self.regular_trails = 0
        self.account_positioning = 0.25

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
                sold, signal_type = trade.sell(sell_position)
                if sold:  # Equivalent to 'if sold is True'
                    # add sell type here Deepseek
                    if signal_type == Trade.HARD_STOP:
                        self.hard_stops += 1
                    elif signal_type == Trade.PROFIT_TARGET:
                        self.profit_targets += 1
                    elif signal_type == Trade.TIGHT_TRAILING:
                        self.tight_trails += 1
                    elif signal_type == Trade.REGULAR_TRAILING:
                        self.regular_trails += 1

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
    
    def print_metrics(self):
        if not self.sold_trades:
            print("No closed trades to analyze.")
            return

        total_trades = len(self.sold_trades)
        wins = [t for t in self.sold_trades if t["profit/loss"] > 0]
        losses = [t for t in self.sold_trades if t["profit/loss"] <= 0]

        total_profit = sum(t["profit/loss"] for t in self.sold_trades)
        total_wins = sum(t["profit/loss"] for t in wins)
        total_losses = abs(sum(t["profit/loss"] for t in losses))  # Abs to keep positive

        avg_profit = total_wins / len(wins) if wins else 0
        avg_loss = -total_losses / len(losses) if losses else 0  # Negative sign for clarity
        win_rate = (len(wins) / total_trades) * 100 if total_trades > 0 else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        expectancy = ((win_rate / 100) * avg_profit) + ((1 - (win_rate / 100)) * avg_loss)

        max_gain = max(t["profit/loss"] for t in self.sold_trades)
        max_loss = min(t["profit/loss"] for t in self.sold_trades)

        print("===== Account Performance Metrics =====")
        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {len(wins)}")
        print(f"Losing Trades: {len(losses)}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total Net Profit: ${total_profit:.2f}")
        print(f"Average Profit (Winners): ${avg_profit:.2f}")
        print(f"Average Loss (Losers): ${avg_loss:.2f}")
        print(f"Max Gain: ${max_gain:.2f}")
        print(f"Max Loss: ${max_loss:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")
        print(f"Expectancy per Trade: ${expectancy:.2f}")
        print(f"Num of Hard Stop Losses: {self.hard_stops}")
        print(f"Num of Profit Targets: {self.profit_targets}")
        print(f"Num of Tight Trailing Stops: {self.tight_trails}")
        print(f"Num of Regular Trailing Stops: {self.regular_trails}")
        print("=======================================")

    def show_trading_timeline(self):
        if not self.sold_trades:
            print("No trades to display")
            return

        # Create figure with adjusted layout for buttons
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(12, 1)  # More rows for better button placement
        ax1 = fig.add_subplot(gs[:3])  # Trade performance
        ax2 = fig.add_subplot(gs[4:10])  # Price chart (more space)
        
        fig.suptitle('Trading Timeline', fontsize=16, y=0.95)
        
        # Prepare timeline data
        dates = [trade['date'] for trade in self.sold_trades]
        profits = [trade['sell_position'] - trade['buy_position'] for trade in self.sold_trades]
        colors = ['g' if p > 0 else 'r' for p in profits]
        percentages = [p/trade['buy_position']*100 for p, trade in zip(profits, self.sold_trades)]
        durations = [(trade['sell_date'] - trade['date']).days for trade in self.sold_trades]
        
        # Plot timeline with duration labels
        bars = ax1.bar(dates, percentages, color=colors, width=2, alpha=0.7)
        ax1.set_ylabel('Return (%)')
        ax1.set_title('Trade Performance (Green=Profit, Red=Loss)')
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Add duration labels to bars
        for bar, duration in zip(bars, durations):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{duration}d', ha='center', va='center', 
                    color='white', weight='bold')

        # Create a list to store button references
        buttons = []
        
        def on_button_clicked(trade_idx, event=None):
            """Handle button click events with candlestick chart"""
            try:
                trade = self.sold_trades[trade_idx]
                
                # Highlight selected trade
                for bar in bars:
                    bar.set_alpha(0.3)
                bars[trade_idx].set_alpha(1)
                
                # Get historical data with buffer period
                buffer_days = pd.Timedelta(days=5)
                stock_data = self.get_historical_data(
                    trade['stock'],
                    trade['date'] - buffer_days,
                    trade['sell_date'] + buffer_days
                )
                
                if stock_data is None or stock_data.empty:
                    ax2.clear()
                    ax2.text(0.5, 0.5, "Data unavailable", ha='center', va='center')
                    ax2.set_title(f"{trade['stock']} - Missing Data")
                    plt.draw()
                    return
                
                # Ensure proper datetime index and numeric values
                if not isinstance(stock_data.index, pd.DatetimeIndex):
                    stock_data.index = pd.to_datetime(stock_data.index)
                
                ohlc_columns = ['Open', 'High', 'Low', 'Close']
                for col in ohlc_columns:
                    if col in stock_data.columns:
                        stock_data[col] = pd.to_numeric(stock_data[col], errors='coerce')
                
                stock_data = stock_data.dropna(subset=ohlc_columns)
                
                if stock_data.empty:
                    ax2.clear()
                    ax2.text(0.5, 0.5, "Invalid price data", ha='center', va='center')
                    ax2.set_title(f"{trade['stock']} - Data Error")
                    plt.draw()
                    return
                
                # Clear previous plot
                ax2.clear()
                
                # Prepare data for candlestick chart
                ohlc_data = stock_data[ohlc_columns].copy()
                ohlc_data['Date'] = mdates.date2num(stock_data.index.to_pydatetime())
                ohlc_data = ohlc_data[['Date', 'Open', 'High', 'Low', 'Close']].values
                
                # Plot candlestick chart
                candlestick_ohlc(ax2, ohlc_data, width=0.6, colorup='g', colordown='r', alpha=0.8)
                
                # Plot entry and exit lines
                ax2.axhline(trade['buy_position'], color='blue', linestyle='--', label='Entry')
                ax2.axhline(trade['sell_position'], color='purple', linestyle='--', label='Exit')
                
                # Mark holding period
                ax2.axvspan(trade['date'], trade['sell_date'], color='gray', alpha=0.1)
                
                # Set title and legend
                ax2.set_title(f"{trade['stock']} Trade {trade_idx+1} | "
                            f"Buy: {trade['buy_position']:.2f} | "
                            f"Sell: {trade['sell_position']:.2f} | "
                            f"Return: {percentages[trade_idx]:.1f}%")
                ax2.legend()
                
                # Format x-axis with dates
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
                
                plt.draw()
                
            except Exception as e:
                print(f"Error processing trade click: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Create buttons - positioned lower in the figure
        button_height = 0.02  # Reduced button height
        button_bottom = 0.01   # Positioned at the very bottom
        for i in range(len(self.sold_trades)):
            ax_button = plt.axes([0.1 + i*0.8/len(self.sold_trades), 
                                button_bottom, 
                                0.8/len(self.sold_trades), 
                                button_height])
            button = Button(ax_button, f"Trade {i+1}")
            button.on_clicked(lambda event, idx=i: on_button_clicked(idx))
            buttons.append(button)
        
        # Show first trade by default
        if buttons:
            try:
                on_button_clicked(0)
            except Exception as e:
                print(f"Error showing initial trade: {str(e)}")
        
        # Adjust layout to prevent overlap
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15, hspace=0.4)
        plt.show()
    def get_historical_data(self, symbol, start_date=None, end_date=None):
        """Retrieve OHLCV data from PostgreSQL database"""
        conn = None
        try:
            conn = psycopg2.connect(
                host='localhost',
                user='postgres',
                password='1234',
                dbname='tradingbot',
                port=5432
            )
            cur = conn.cursor()
            
            # Get stock_id
            cur.execute("""
                SELECT stock_id, symbol, company_name 
                FROM stocks 
                WHERE symbol = %s OR symbol = %s
            """, (symbol, symbol.replace('.TO', '')))
            
            stock = cur.fetchone()
            if not stock:
                raise ValueError(f"Symbol {symbol} not found in database")
            
            stock_id, symbol, company = stock
            
            # Build query
            query = """
                SELECT date, open, high, low, close, volume 
                FROM daily_prices 
                WHERE stock_id = %s
            """
            params = [stock_id]
            
            if start_date:
                query += " AND date >= %s"
                params.append(start_date)
            if end_date:
                query += " AND date <= %s"
                params.append(end_date)
                
            query += " ORDER BY date ASC"
            
            # Execute query
            cur.execute(query, params)
            data = cur.fetchall()
            
            if not data:
                return None
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df.set_index('Date', inplace=True)
            
            # Add metadata
            df.attrs['symbol'] = symbol
            df.attrs['company'] = company
            
            return df
            
        except Exception as e:
            print(f"Database error for {symbol}: {str(e)}")
            return None
        finally:
            if conn:
                conn.close()








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



    def show_trading_timeline(self):
        """Display interactive timeline of all trades with candlestick details"""
        if not self.sold_trades:
            print("No trades to display")
            return
        
        # Create figure with two subplots
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(10, 1)
        ax1 = fig.add_subplot(gs[:2])
        ax2 = fig.add_subplot(gs[3:])
        
        fig.suptitle('Trading Timeline', fontsize=16, y=0.95)
        
        # Prepare timeline data
        dates = [trade['date'] for trade in self.sold_trades]
        profits = [trade['sell_position'] - trade['buy_position'] for trade in self.sold_trades]
        colors = ['g' if p > 0 else 'r' for p in profits]
        percentages = [p/trade['buy_position']*100 for p, trade in zip(profits, self.sold_trades)]
        
        # Plot timeline
        bars = ax1.bar(dates, percentages, color=colors, width=2)
        ax1.set_ylabel('Return (%)')
        ax1.set_title('Trade Performance (Green=Profit, Red=Loss)')
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Create interactive buttons
        button_axes = [plt.axes([0.1 + i*0.8/len(self.sold_trades), 0.01, 0.8/len(self.sold_trades), 0.03]) 
                      for i in range(len(self.sold_trades))]
        buttons = [Button(ax, f"{i+1}") for i, ax in enumerate(button_axes)]
        
        def show_trade(event):
            trade_idx = buttons.index(event.inaxes.buttons[0])
            trade = self.sold_trades[trade_idx]
            
            # Highlight selected trade in timeline
            for bar in bars:
                bar.set_alpha(0.3)
            bars[trade_idx].set_alpha(1)
            
            # Get price data with buffer period
            start_date = trade['date'] - timedelta(days=10)
            end_date = trade['sell_date'] + timedelta(days=10)
            stock_data = self.get_historical_data(trade['stock'], start_date, end_date)
            
            if stock_data is None:
                ax2.clear()
                ax2.text(0.5, 0.5, "Could not load price data", 
                         ha='center', va='center')
                ax2.set_title(f"{trade['stock']} Trade {trade_idx+1} - Data Unavailable")
                plt.draw()
                return
            
            # Prepare plot data
            trade_period = stock_data.loc[trade['date']:trade['sell_date']]
            
            # Clear and plot
            ax2.clear()
            mpf.plot(stock_data, type='candle', style='charles',
                     title=f"{trade['stock']} Trade {trade_idx+1} | "
                           f"Bought: {trade['date'].date()} @ ${trade['buy_position']:.2f} | "
                           f"Sold: {trade['sell_date'].date()} @ ${trade['sell_position']:.2f}",
                     ax=ax2,
                     addplot=[
                         mpf.make_addplot([trade['buy_position']]*len(stock_data), color='blue'),
                         mpf.make_addplot([trade['sell_position']]*len(stock_data), color='purple')
                     ])
            
            # Mark entry and exit periods
            ax2.axvspan(trade['date'], trade['sell_date'], color='gray', alpha=0.1)
            ax2.axhline(trade['buy_position'], color='b', linestyle='--', alpha=0.7, label='Entry')
            ax2.axhline(trade['sell_position'], color='p', linestyle='--', alpha=0.7, label='Exit')
            ax2.legend()
            
            plt.draw()
        
        # Connect buttons
        for button in buttons:
            button.on_clicked(show_trade)
        
        # Show first trade by default
        show_trade(buttons[0])
        
        plt.tight_layout()
        plt.show()

    def get_historical_data(self, symbol, start_date=None, end_date=None):
        """Retrieve OHLCV data from PostgreSQL database"""
        conn = None
        try:
            conn = psycopg2.connect(
                host='localhost',
                user='postgres',
                password='1234',
                dbname='tradingbot',
                port=5432
            )
            cur = conn.cursor()
            
            # Get stock_id
            cur.execute("""
                SELECT stock_id, symbol, company_name 
                FROM stocks 
                WHERE symbol = %s OR symbol = %s
            """, (symbol, symbol.replace('.TO', '')))
            
            stock = cur.fetchone()
            if not stock:
                raise ValueError(f"Symbol {symbol} not found in database")
            
            stock_id, symbol, company = stock
            
            # Build query
            query = """
                SELECT date, open, high, low, close, volume 
                FROM daily_prices 
                WHERE stock_id = %s
            """
            params = [stock_id]
            
            if start_date:
                query += " AND date >= %s"
                params.append(start_date)
            if end_date:
                query += " AND date <= %s"
                params.append(end_date)
                
            query += " ORDER BY date ASC"
            
            # Execute query
            cur.execute(query, params)
            data = cur.fetchall()
            
            if not data:
                return None
            
            # Create DataFrame
            df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df.set_index('Date', inplace=True)
            
            # Add metadata
            df.attrs['symbol'] = symbol
            df.attrs['company'] = company
            
            return df
            
        except Exception as e:
            print(f"Database error for {symbol}: {str(e)}")
            return None
        finally:
            if conn:
                conn.close()