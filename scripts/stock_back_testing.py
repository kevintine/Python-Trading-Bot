from backtesting import Backtest, Strategy
from alpaca_testing import get_stock_bars
from backtesting.lib import crossover

import talib

AMD = get_stock_bars("AMD")

class RsiOscillator(Strategy):

    upper_bound = 70
    lower_bound = 30

    def init(self):
        self.rsi = self.I(talib.RSI, self.data.Close, 14)

    def next(self):

        if crossover(self.rsi, self.upper_bound):
            self.position.close()
            
        elif crossover(self.lower_bound, self.rsi):
            self.buy()

bt = Backtest(AMD, RsiOscillator, cash = 10_000)
stats = bt.run()

print(stats)

bt.plot()