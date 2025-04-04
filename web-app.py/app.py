import os
import sys
from pathlib import Path
import threading
import asyncio
from decouple import config
from flask import Flask, jsonify, request, render_template
from flask_socketio import SocketIO

# Get the absolute path to the project root (Python Trading Bot folder)
project_root = Path(__file__).resolve().parent.parent

# Add the scripts directory to Python path
scripts_path = str(project_root / 'tradingbot')
sys.path.insert(0, scripts_path)  # Insert at start of path

app = Flask(__name__)
socketio = SocketIO(app)

# Configuration (add before route definitions)
API_KEY = config('ALPACA_API_KEY', default='')  # Will read from .env file
API_SECRET = config('ALPACA_API_SECRET', default='') 

# Streaming control
stream_active = False
streamer = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream')
def stream_dashboard():
    return render_template('stream.html')

@app.route('/start_stream', methods=['POST'])
def start_stream():
    global stream_active, streamer
    
    if not stream_active:
        try:
            # Now this import should work
            from alpaca_stream import AlpacaStreamer
            
            symbols = request.json.get('symbols', ['AAPL', 'MSFT'])
            api_key = 'PKGJHW34RDC8CG26S1OB'  # Consider moving to config/environment variables
            api_secret = 'NDu8J0yVPfo7enmZaTFBPSUpbCVdw1ccWYWExNxI'
            
            streamer = AlpacaStreamer(api_key, api_secret, symbols, socketio)
            
            def run_stream():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(streamer.run())
                loop.close()
                
            thread = threading.Thread(target=run_stream, daemon=True)
            thread.start()
            stream_active = True
            return jsonify({'status': 'started'})
        
        except ImportError as e:
            return jsonify({'status': 'error', 'message': str(e), 'path': sys.path}), 500
    
    return jsonify({'status': 'already running'})

@app.route('/stop_stream')
def stop_stream():
    global stream_active, streamer
    if streamer and stream_active:
        streamer.stop()
        stream_active = False
        return jsonify({'status': 'stopped'})
    return jsonify({'status': 'not running'})

# Add initialization function
def create_streamer():
    global streamer
    if streamer is None:
        from alpaca_stream import AlpacaStreamer
        if not API_KEY or not API_SECRET:
            raise ValueError("Alpaca API credentials not configured")
        streamer = AlpacaStreamer(API_KEY, API_SECRET, [], socketio)
    return streamer

# Modified add_stock endpoint
@app.route('/add_stock', methods=['POST'])
def add_stock():
    try:
        if not API_KEY or not API_SECRET:
            return jsonify({
                'success': False,
                'message': 'API credentials not configured'
            }), 500
            
        symbol = request.json.get('symbol', '').strip().upper()
        if not symbol:
            return jsonify({'success': False, 'message': 'No symbol provided'})
        
        streamer = create_streamer()
        
        # Initialize symbols list if not exists
        if not hasattr(streamer, 'symbols'):
            streamer.symbols = []
        
        if symbol in streamer.symbols:
            return jsonify({
                'success': False,
                'message': f'{symbol} is already being tracked'
            })
            
        streamer.symbols.append(symbol)
        
        # Start or update stream
        if not stream_active:
            thread = threading.Thread(target=run_stream, daemon=True)
            thread.start()
            stream_active = True
        else:
            asyncio.run_coroutine_threadsafe(streamer.resubscribe(), streamer._loop)
        
        return jsonify({
            'success': True,
            'symbol': symbol,
            'message': f'Started tracking {symbol}'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

def run_stream():
    streamer = create_streamer()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(streamer.run())
    finally:
        loop.close()
    
@socketio.on('connect')
def handle_connect():
    print('Client connected')

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5001)
    app.run(debug=True, port=5001)