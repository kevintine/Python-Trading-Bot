# This is where the bot will run

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
from data.fetch import getStockData

# I want this main function to be run contiually. Have trades be saved in a database and be pulled everytime it is run.
# {   'close': 30.59,
#     'high': 30.6582,
#     'low': 30.285,
#     'open': 30.42,
#     'symbol': 'INTC',
#     'timestamp': datetime.datetime(2024, 6, 27, 4, 0, tzinfo=TzInfo(UTC)),
#     'trade_count': 148007.0,
#     'volume': 30184280.0,
#     'vwap': 30.529181}
def main():
    data = getStockData("INTC")
    

    
    return 0

if __name__ == "__main__":
    main()