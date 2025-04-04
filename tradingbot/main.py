import asyncio
from alpaca_stream import AlpacaStreamer

API_KEY = "PKGJHW34RDC8CG26S1OB"
API_SECRET = "NDu8J0yVPfo7enmZaTFBPSUpbCVdw1ccWYWExNxI"
BASE_URL = "https://paper-api.alpaca.markets"

async def main():
    # Configuration
    SYMBOLS = ["AAPL", "TSLA", "MSFT"]  # Stocks to track
    
    # Create and run the streamer
    streamer = AlpacaStreamer(API_KEY, API_SECRET, SYMBOLS)
    await streamer.run()

if __name__ == "__main__":
    asyncio.run(main())