import asyncio
from alpaca_trade_api.stream import Stream
from datetime import datetime

class AlpacaStreamer:
    def __init__(self, api_key, api_secret, symbols, socketio):
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbols = symbols
        self.socketio = socketio
        self.stream = None
        self._running = False

    async def _handle_trade(self, trade):
        if self._running:
            data = {
                'symbol': trade.symbol,
                'price': trade.price,
                'size': trade.size,
                'timestamp': datetime.now().isoformat()
            }
            self.socketio.emit('trade_update', data)

    async def run(self):
        self._running = True
        self.stream = Stream(
            self.api_key,
            self.api_secret,
            base_url='https://paper-api.alpaca.markets',
            data_feed='iex'
        )
        
        self.stream.subscribe_trades(self._handle_trade, *self.symbols)
        
        try:
            await self.stream._run_forever()
        except Exception as e:
            print(f"Stream error: {e}")
        finally:
            await self.close()

    async def resubscribe(self):
        """Resubscribe to current symbols"""
        if self.stream:
            await self.stream.unsubscribe_trades(*self.symbols)
            await self.stream.subscribe_trades(self._handle_trade, *self.symbols)
            self._logger.info(f"Resubscribed to symbols: {self.symbols}")

    async def close(self):
        if self.stream:
            await self.stream.close()
        self._running = False

    async def resubscribe(self):
        """Handle subscription updates"""
        if self.stream:
            try:
                # Unsubscribe first if needed
                if hasattr(self, 'symbols') and self.symbols:
                    await self.stream.unsubscribe_trades(*self.symbols)
                    
                # Re-subscribe with updated list
                await self.stream.subscribe_trades(self._handle_trade, *self.symbols)
                self._logger.info(f"Updated subscriptions: {self.symbols}")
            except Exception as e:
                self._logger.error(f"Resubscribe error: {e}")
                raise

    def stop(self):
        """Proper cleanup"""
        self._running = False
        if hasattr(self, '_loop') and self._loop:
            asyncio.run_coroutine_threadsafe(self._cleanup(), self._loop)