from datetime import datetime
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# Trade Class which will contain one transsaction from start to finish.
# The bot must create one of these to initaite a trade and call the sell function to end it.

# Before this class is designed, I must first understand how the alpaca trading works.
class Trade:
    def __init__(self, symbol, quantity, entryPrice, entryTime=None, price = 0):
        self.symbol = symbol
        self.quantity = quantity
        self.entryPrice = entryPrice
        self.entryTime = datetime.now()

    def sell(self):
        sellOrder = MarketOrderRequest(
            symbol = self.symbol,
            qty = self.quantity,
            side = OrderSide.SELL,
            time_in_force = TimeInForce.DAY
        )

        
    def display(self):
        print(self.symbol)
        print(self.quantity)
        print(self.entryTime)
        