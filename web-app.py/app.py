from flask import Flask, render_template
from flask_socketio import SocketIO
import alpaca_trade_api as tradeapi
import asyncio
import threading
import queue
from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path('..') / '.env'
load_dotenv(dotenv_path=env_path)
API_KEY = os.getenv('YOUR_API_KEY_ID')
API_SECRET = os.getenv('YOUR_SECRET_KEY')

app = Flask(__name__)
socketio = SocketIO(app, async_mode='threading')

# Alpaca Configuration
BASE_URL = 'https://paper-api.alpaca.markets'

# Stocks to watch
WATCHLIST = ['AAPL', 'MSFT', 'TSLA', 'AMZN', 'GOOGL']

# Store latest prices
latest_prices = {symbol: {'price': 0, 'change': 0} for symbol in WATCHLIST}

# Queue for thread-safe communication
update_queue = queue.Queue()

def alpaca_stream_thread():
    async def run_stream():
        stream = tradeapi.stream2.Stream(API_KEY, API_SECRET, base_url=BASE_URL, data_feed='iex')
        
        async def on_bar(bar):
            try:
                # Get previous close
                prev_close = api.get_barset(bar.symbol, 'day', limit=2)[bar.symbol][0].c
                change_pct = ((bar.close - prev_close) / prev_close) * 100
                
                update_queue.put({
                    'symbol': bar.symbol,
                    'price': bar.close,
                    'change': change_pct
                })
            except Exception as e:
                print(f"Error processing bar: {e}")

        # Subscribe to bars
        stream.subscribe_bars(on_bar, *WATCHLIST)
        
        # Run stream
        await stream._run_forever()

    # Create new event loop for the thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_stream())

def process_updates():
    while True:
        try:
            update = update_queue.get_nowait()
            symbol = update['symbol']
            latest_prices[symbol] = {
                'price': update['price'],
                'change': update['change']
            }
            
            socketio.emit('stock_update', {
                'symbol': symbol,
                'price': update['price'],
                'change': f"{update['change']:.2f}%",
                'color': 'green' if update['change'] >= 0 else 'red'
            })
        except queue.Empty:
            socketio.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html', watchlist=WATCHLIST)

@socketio.on('connect')
def handle_connect():
    # Send initial data when client connects
    for symbol, data in latest_prices.items():
        socketio.emit('stock_update', {
            'symbol': symbol,
            'price': data['price'],
            'change': f"{data['change']:.2f}%" if data['price'] != 0 else '0.00%',
            'color': 'green' if data['change'] >= 0 else 'red'
        })

if __name__ == '__main__':
    # Initialize Alpaca REST client
    api = tradeapi.REST(API_KEY, API_SECRET, BASE_URL)
    
    # Start Alpaca stream thread
    threading.Thread(target=alpaca_stream_thread, daemon=True).start()
    
    # Start update processing thread
    threading.Thread(target=process_updates, daemon=True).start()
    
    # Run Flask app
    socketio.run(app, debug=True)