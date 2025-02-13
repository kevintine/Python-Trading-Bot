import yfinance as yf
import talib as ta
import datetime as dt
class Strategy:
    def __init__(self, data, data_one_day, position):
        self.id = id
        self.data = data
        self.data_one_day = data_one_day
        self.position = position

    def entry(self):
        # check if one day has the candlestick pattern  
        # if so, enter position
        # if not, do nothing
        result = ta.CDLENGULFING(self.data_one_day['Open'], self.data_one_day['High'], self.data_one_day['Low'], self.data_one_day['Close'])
        print(result)
        return 0
    def exit(self):
        # check if position is greater or less then 10%
        pass

# TESTING

# Date Calculations
current_date = dt.datetime.now()
one_year_ago = current_date - dt.timedelta(days=365)
date_one_year_ago = one_year_ago.strftime("%Y-%m-%d")
current_date = dt.datetime.now().strftime("%Y-%m-%d")
# Sample stock data
data = yf.download("AC.TO", start="2020-01-01", end="2020-08-01")
# get data for one day
stockData = yf.Ticker("AC.TO").history(period='1d', start=date_one_year_ago, end=current_date)





        