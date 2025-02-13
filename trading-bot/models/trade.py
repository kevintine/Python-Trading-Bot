import yfinance as yf
import talib as ta
import pandas as pd

# when a buy is initiated, keep track of the trade with this class
class Trade:
    # when a buy happens, get the price and enter that number in an attribute here
    # monitor the position each day to the stock and sell when price is too high or low
    def __init__(self, signal = "Signal"):
        self.position = None
        self.signal = signal  
        self.sell_position = None

        # Get the trading signals
        self.signal.initialize_cdl_pattern()
        self.signal.initialize_above_avg_volume()
        
        # Combine all data into a single DataFrame
        self.data = self.signal.get_data_chart()

    def buy(self):
        data = self.data
        # if talib is 100 and volume is 100 and buy
        # set the position to the price of the buy
        # ISSUE: will want to create a signals class to determine the buy process
        for index, row in data.iterrows():
            if row["Talib_Pattern"].item() != 0 and row["Volume_Pattern"].item() != 0:
                self.position = row["Close"]
                print(f"Buy! Date: {index.date()}, Price: {row['Close'].item()}") 
                return 0
        print("No buys!")
        return 
    
    def sell(self):
        if self.position is None:
            print("No active position!")
            return
        # # get the current position
        date_bought = self.position.name.date()
        price_bought = self.position.item()

        # get the 10% + and - threshold
        lower = price_bought * 0.9
        higher = price_bought * 1.06

        # loop from date bought to date end
        # if the date bought price goes higher or lower 10%, sell
        data = yf.download(self.signal.stock, start=date_bought,end=self.signal.end)
        for index, row in data.iterrows():
            # check if the price has moved 10% up or down from the buy price
            if row["Close"].item() >= higher:
                print(f"Sell! Date: {index.date()}, Price: {row['Close'].item()}")
                self.sell_position = row["Close"]
                break 
            elif row["Close"].item() <= lower:
                print(f"Sell! Date: {index.date()}, Price: {row['Close'].item()}")
                self.sell_position = row["Close"]
                break  
        else:
            print("No sell condition met.")
        return
    
    def print_position(self):
        if self.position is None:
            print("No active position!")
        else:
            print(f"Current Position: {self.position}")

    def get_position(self):
        return self.position
        

# this signal class sends trade signals based on two factors, volume and cdl patterns
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
        
# this class will take trade objects and keep track of profits
class Account:
    def __init__(self, balance, signal = "Signal"):
        self.balance = balance
        self.trades = []
        self.profit = 0    
        self.signal = signal

        # Get the trading signals
        self.signal.initialize_cdl_pattern()
        self.signal.initialize_above_avg_volume()
        
        # Combine all data into a single DataFrame
        self.data = self.signal.get_data_chart() 

    def add_trade(self, trade):
        self.trades.append(trade)
        return
    
    def run(self):
        # run through the trades and buy 500$ worth of stock for each trade
        # at position
        # then selll them at sell_position
        # update profit accordingly
        """Execute all trades, buying $500 worth and selling when conditions are met."""
        for trade in self.trades:
            # Initiate buy
            trade.buy()
            
            if trade.position is None:
                print(f"No buy signal for {trade.signal.stock}, skipping...")
                continue
            
            # Buy $500 worth of stock
            buy_price = trade.position.item()
            num_shares = 500 / buy_price
            self.balance -= num_shares * buy_price  # Deduct investment
            print(f"Bought {num_shares:.2f} shares of {trade.signal.stock} at ${buy_price:.2f}")

            # Check for selling opportunity
            trade.sell()
            
            if trade.sell_position is None:
                print(f"No sell signal for {trade.signal.stock}, holding position...")
                continue
            
            # Sell the stock
            sell_price = trade.sell_position.item()
            total_sell_value = num_shares * sell_price
            self.balance += total_sell_value  # Add sale proceeds

            print(f"Sold {num_shares:.2f} shares of {trade.signal.stock} at ${sell_price:.2f}, "
                  f"New Balance: ${self.balance:.2f}")

        print(f"\nFinal Account Balance: ${self.balance:.2f}")
        pass 

