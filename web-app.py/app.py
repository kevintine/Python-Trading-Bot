from flask import Flask, render_template, request, jsonify, send_from_directory
import json
from alpaca.data.live import StockDataStream 
import yfinance as yf
import threading
from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path('..') / '.env'
load_dotenv(dotenv_path=env_path)
API_KEY = os.getenv('YOUR_API_KEY_ID')
API_SECRET = os.getenv('YOUR_SECRET_KEY')

app = Flask(__name__, template_folder='templates')

stream = None  # Global stream handle

DEFAULT_STOCKS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
subscribed_symbols = set(DEFAULT_STOCKS)
# Store live prices
live_prices = {symbol: None for symbol in DEFAULT_STOCKS}

def start_stream():
    global stream 
    # Correct initialization (no keyword arguments)
    stream = StockDataStream('PKGJHW34RDC8CG26S1OB', 'NDu8J0yVPfo7enmZaTFBPSUpbCVdw1ccWYWExNxI')
    
    async def on_trade(trade):
        symbol = trade.symbol
        price = trade.price
        print(f"TRADE UPDATE: {symbol} @ {price}")
        live_prices[symbol] = price
    
    # Subscribe to all default stocks
    for symbol in DEFAULT_STOCKS:
        stream.subscribe_trades(on_trade, symbol)
    
    print("Starting WebSocket connection...")
    try:
        stream.run()
    except Exception as e:
        print(f"WebSocket error: {e}")

# Start WebSocket in background
threading.Thread(target=start_stream, daemon=True).start()

@app.route('/live_prices')
def get_live_prices():
    return jsonify({
        'prices': live_prices,
        'status': 'connected' if any(live_prices.values()) else 'waiting_for_data'
    })

@app.route('/search_stock', methods=['POST'])
def search_stock():
    global stream, DEFAULT_STOCKS, live_prices, subscribed_symbols

    symbol = request.form.get('symbol', '').upper()

    if not symbol:
        return jsonify({'error': 'Symbol not provided'}), 400

    try:
        stock = yf.Ticker(symbol)
        info = stock.info

        if not info or ('regularMarketPrice' not in info and 'currentPrice' not in info):
            return jsonify({'error': 'Stock not found'}), 404

        current_price = info.get('currentPrice', info.get('regularMarketPrice'))
        hist = stock.history(period="1d", interval="1m")
        day_high = hist['High'].max()
        day_low = hist['Low'].min()
        stock_data = {
            'symbol': symbol,
            'name': info.get('shortName', info.get('longName', 'N/A')),
            'current_price': info.get("previousClose"),
            'currency': info.get('currency', 'N/A'),
            'previous_close': info.get('previousClose', 'N/A'),
            'day_high': day_high,
            'day_low': day_low
        }

        # Add to default list if not already there
        if symbol not in DEFAULT_STOCKS:
            DEFAULT_STOCKS.append(symbol)

        # Add to live_prices if not tracked
        if symbol not in live_prices:
            live_prices[symbol] = current_price

        # Subscribe to live data if not already subscribed
        if symbol not in subscribed_symbols:
            subscribed_symbols.add(symbol)

            async def on_trade(trade):
                if trade.symbol == symbol:
                    live_prices[trade.symbol] = trade.price
                    # print(f"TRADE UPDATE (dynamic): {trade.symbol} @ {trade.price}")

            if stream:
                stream.subscribe_trades(on_trade, symbol)

        return jsonify(stock_data)

    except Exception as e:
        import traceback
        traceback.print_exc()  # Shows full error in console
        return jsonify({
            'error': f'Error searching for symbol: {symbol}',
            'details': str(e)
        }), 500

@app.route('/')
def home():
    # Just send the file - we'll handle defaults in JavaScript
    return send_from_directory('templates', 'index.html')

@app.route('/backtest')
def backtest():
    # Just send the file - we'll handle defaults in JavaScript
    return send_from_directory('templates', 'backtest.html')

@app.route('/get_default_stocks')
def get_default_stocks():
    return json.dumps(DEFAULT_STOCKS)

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('templates', filename)

if __name__ == '__main__':
    app.run(debug=True)