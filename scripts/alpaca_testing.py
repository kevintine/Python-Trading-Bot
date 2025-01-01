from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime

# Takes in a stock symbol and returns a dataframe from 2018-07-18 to present day
# The dataframe gets converted from a 9 column table to 6 column
def get_stock_bars(symbol):
    client = StockHistoricalDataClient("PKRKXLD9ZT0BZUD6WRS3", "vHmp76htHcrzkCVVmNlHwYwuz2hAeWQFecKGtlDS")

    request_params = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=TimeFrame.Day,
        start=datetime.strptime("2018-07-18", '%Y-%m-%d')
    )

    bars = client.get_stock_bars(request_params)
    stock_bars = bars.df
    stock_bars = stock_bars.reset_index()
    stock_bars = stock_bars.rename(columns={'timestamp': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
    stock_bars.set_index('Date', inplace=True)
    return stock_bars

symbol = 'AMD'

print(get_stock_bars(symbol))



