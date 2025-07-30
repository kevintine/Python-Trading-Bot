import yfinance as yf

# Define the ticker symbol
ticker = 'AAPL'

# Fetch historical data
data = yf.download(ticker, start='2025-07-01', end='2025-07-19', interval='1d')

# Show the first few rows
print(data.tail())