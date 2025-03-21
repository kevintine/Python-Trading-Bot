from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta

# Date Calculations
current_date = datetime.now()
one_year_ago = current_date - timedelta(days=365)
date_one_year_ago = one_year_ago.strftime("%Y-%m-%d")
current_date = datetime.now().strftime("%Y-%m-%d")

def getStockData(symbol):
    client = StockHistoricalDataClient("PKRKXLD9ZT0BZUD6WRS3", "vHmp76htHcrzkCVVmNlHwYwuz2hAeWQFecKGtlDS")

    request_params = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Day,
        start=date_one_year_ago
    )

    data = client.get_stock_bars(request_params)

    return data

