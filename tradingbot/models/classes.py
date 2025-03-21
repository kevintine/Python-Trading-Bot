import yfinance as yf
import talib as ta
import pandas as pd

class Trade:
    def __init__(self, buy_position, num_of_stocks, stock_name, date):
        self.buy_position = buy_position
        self.sell_position = None
        self.current_position = buy_position
        self.num_of_stocks = num_of_stocks
        self.stock_name = stock_name
        self.date = date
    
    def buy(self):
        pass

    def sell(self, new_position):
        # Update the most recent high price
        if new_position > self.current_position:
            self.current_position = new_position  # Set new high price

        # Condition 1: Sell if price drops 5% from most recent high
        if new_position < self.current_position * 0.90:
            self.sell_position = new_position
            return True

        # Condition 2: Sell if price drops 20% below buy price
        if new_position < self.buy_position * 0.80:
            self.sell_position = new_position
            return True
        
        return False  # No sell triggered




# Created at the beginning of the program to keep track of all trades and overall position
class Account:
    def __init__(self, balance):
        self.balance = balance
        self.trades = []
        self.position_total = 0
        self.sold_trades = []

    def add_trade(self, position, num_of_stocks, stock_name, date):
        # check if the balance of the account is greater than the cost of the trade
        if self.balance < position * num_of_stocks:
            return
        # create a trade object and add it to the trades list
        trade = Trade(position, num_of_stocks, stock_name, date)
        self.trades.append(trade)
        # subtract the cost of the trade from the balance
        self.balance -= position * num_of_stocks
        # print the transaction
        print(f"Bought {num_of_stocks} shares of {stock_name} at {position} per share")
        return
    
    def check_trades(self, sell_position, symbol):
        for trade in self.trades[:]:  # Iterate over a copy of the list to avoid modification issues
            if trade.sell(sell_position) is True and trade.stock_name == symbol:
                # Update balance
                self.balance += trade.sell_position * trade.num_of_stocks
                
                # Store trade details in sold_trades
                self.sold_trades.append({
                    "date": trade.date,
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
            print(f"Trade Details: Date: {trade['date']}, "
            f"Stock: {trade['stock']}, "
            f"Buy Price: {trade['buy_position']:.2f}, "
            f"Sell Price: {trade['sell_position']:.2f}, "
            f"Shares: {trade['num_of_stocks']}, "
            f"Profit/Loss: {trade['profit/loss']:.2f}")


        return


# this class JUST ONE of many different types of signals i will be creating. 
# This signal will send a signal based off of a candlestick pattern and volume
class Signal:
    def __init__(self, stock, pattern, start, end):
        self.volume_chart = None
        self.cdl_chart = None
        self.data_chart = None
        self.stock = stock
        self.pattern = pattern
        self.start = start
        self.end = end

    # FUNCTION: to get ta-lib cdl pattern and sets self.cdl_pattern attribute
    def initialize_cdl_pattern(self):
        # get cdl pattern
        # takes a stock string and pattern string
        # some issues:
        # maybe two years ago, yfinance used to return data a certain way and now its changed where ta-lib wasnt able to take in its values 
        # i had to change the code for ta-lib to continue to accept a pandas series where yfinance was give it a pandas dataframe. 
        # yfinance issue
        data = yf.download(self.stock, start=self.start,end=self.end)
        pattern_function = getattr(ta, self.pattern)
        integer = pattern_function(data['Open'].squeeze(), data['High'].squeeze(), data['Low'].squeeze(), data['Close'].squeeze())
        self.cdl_chart = integer
        return 
    
    # FUNCTION: get instances of above average volume and sets self.volume_chart attribute
    def initialize_above_avg_volume(self):
        # calculate the average volume and pick out days where the volume is 1.4 times higher than that
        # return something accurate like the ta-lib pattern with 0, 100 and -100
        data = yf.download(self.stock, start=self.start,end=self.end)
        volume = data['Volume'].squeeze().mean()
        threshold = volume * 1.7
        volume_indicator = data['Volume'].squeeze().apply(lambda x: 100 if x > threshold else 0)
        volume_df = pd.DataFrame({
            'Date': data.index,
            'Volume Indicator': volume_indicator
        })
        volume_df = volume_df["Volume Indicator"]
        self.volume_chart = volume_df
        return 
   
    # FUNCTION: combine both cdl and volume charts to set data_chart attribute
    def get_data_chart(self):
        if self.cdl_chart is None or self.volume_chart is None:
            print("cdl_chart or volume chart missing to get data_chart")
            return
        self.data_chart = yf.download(self.stock, start=self.start,end=self.end)
        self.data_chart["Talib_Pattern"] = self.cdl_chart
        self.data_chart["Volume_Pattern"] = self.volume_chart
        return self.data_chart
    
    def __init__(self, stock, pattern, start, end):
        self.volume_chart = None
        self.cdl_chart = None
        self.data_chart = None
        self.stock = stock
        self.pattern = pattern
        self.start = start
        self.end = end
    def action():
        pass