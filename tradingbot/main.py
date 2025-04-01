from alpaca.trading.client import TradingClient

trading_client = TradingClient("PKGJHW34RDC8CG26S1OB", "NDu8J0yVPfo7enmZaTFBPSUpbCVdw1ccWYWExNxI")
def main():
    print(trading_client.get_clock())

if __name__ == "__main__":
    main()