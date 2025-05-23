import yfinance as yf

patterns = {
    "CDLDOJI": "Doji",
    "CDLENGULFING": "Engulfing Pattern",
    "CDLHAMMER": "Hammer",
    "CDLINVERTEDHAMMER": "Inverted Hammer",
    "CDL3WHITESOLDIERS": "Three Advancing White Soldiers",
    "CDLMORNINGSTAR": "Morning Star",
    "HAMMER": "Kevin Hammer"
}

stocks = {
    "T.TO": "Telus Corporation",
    "SU.TO": "Suncor Energy Inc.",
    "RY.TO": "Royal Bank of Canada",
    "TD.TO": "TD Bank",
    "AC.TO": "Air Canada",
    "CM.TO": "Canadian Imperial Bank of Commerce",
    "AW.TO": "A & W Food Services of Canada Inc.",
    "INTC": "Intel Corporation",
    "AMD": "Advanced Micro Devices, Inc.",
    "MSFT": "Microsoft Corporation",
    "AAPL": "Apple Inc."
}

pattern_descriptions = {
    "CDLDOJI": {
        "description": "A Doji is a candlestick pattern where the open and close prices are nearly equal, indicating market indecision. It often signals a potential reversal or continuation, depending on the context of the trend.",
        "image_path": "static/images/cdl_doji.jpg"
    },
    "CDLENGULFING": {
        "description": "The Engulfing pattern consists of a smaller candlestick followed by a larger one that completely engulfs it. A bullish engulfing pattern suggests a potential upward reversal, while a bearish engulfing indicates possible downside.",
        "image_path": "static/images/cdl_engulfing.jpg"
    },
    "CDLHAMMER": {
        "description": "The Hammer is a single candlestick with a small body and a long lower wick, appearing after a downtrend. It signals potential bullish reversal as buyers step in at lower levels.",
        "image_path": "static/images/cdl_hammer.jpg"
    },
    "CDLINVERTEDHAMMER": {
        "description": "The Inverted Hammer is similar to the Hammer but has a long upper wick and appears after a downtrend. It indicates a potential bullish reversal but requires confirmation.",
        "image_path": "static/images/cdl_inverted_hammer.jpg"
    },
    "CDLMORNINGSTAR": {
        "description": "The Morning Star is a three-candlestick pattern that forms after a downtrend. It consists of a large bearish candle, a smaller indecisive candle, and a bullish candle, signaling a reversal to the upside.",
        "image_path": "static/images/cdl_morning_star.jpg"
    }
}


# patterns = {
#     "CDL2CROWS": "Two Crows",
#     "CDL3BLACKCROWS": "Three Black Crows",
#     "CDL3INSIDE": "Three Inside Up/Down",
#     "CDL3LINESTRIKE": "Three-Line Strike",
#     "CDL3OUTSIDE": "Three Outside Up/Down",
#     "CDL3STARSINSOUTH": "Three Stars In The South",
#     "CDL3WHITESOLDIERS": "Three Advancing White Soldiers",
#     "CDLABANDONEDBABY": "Abandoned Baby",
#     "CDLADVANCEBLOCK": "Advance Block",
#     "CDLBELTHOLD": "Belt-hold",
#     "CDLBREAKAWAY": "Breakaway",
#     "CDLCLOSINGMARUBOZU": "Closing Marubozu",
#     "CDLCONCEALBABYSWALL": "Concealing Baby Swallow",
#     "CDLCOUNTERATTACK": "Counterattack",
#     "CDLDARKCLOUDCOVER": "Dark Cloud Cover",
#     "CDLDOJI": "Doji",
#     "CDLDOJISTAR": "Doji Star",
#     "CDLDRAGONFLYDOJI": "Dragonfly Doji",
#     "CDLENGULFING": "Engulfing Pattern",
#     "CDLEVENINGDOJISTAR": "Evening Doji Star",
#     "CDLEVENINGSTAR": "Evening Star",
#     "CDLGAPSIDESIDEWHITE": "Up/Down-gap side-by-side white lines",
#     "CDLGRAVESTONEDOJI": "Gravestone Doji",
#     "CDLHAMMER": "Hammer",
#     "CDLHANGINGMAN": "Hanging Man",
#     "CDLHARAMI": "Harami Pattern",
#     "CDLHARAMICROSS": "Harami Cross Pattern",
#     "CDLHIGHWAVE": "High-Wave Candle",
#     "CDLHIKKAKE": "Hikkake Pattern",
#     "CDLHIKKAKEMOD": "Modified Hikkake Pattern",
#     "CDLHOMINGPIGEON": "Homing Pigeon",
#     "CDLIDENTICAL3CROWS": "Identical Three Crows",
#     "CDLINNECK": "In-Neck Pattern",
#     "CDLINVERTEDHAMMER": "Inverted Hammer",
#     "CDLKICKING": "Kicking",
#     "CDLKICKINGBYLENGTH": "Kicking - bull/bear determined by the longer marubozu",
#     "CDLLADDERBOTTOM": "Ladder Bottom",
#     "CDLLONGLEGGEDDOJI": "Long Legged Doji",
#     "CDLLONGLINE": "Long Line Candle",
#     "CDLMARUBOZU": "Marubozu",
#     "CDLMATCHINGLOW": "Matching Low",
#     "CDLMATHOLD": "Mat Hold",
#     "CDLMORNINGDOJISTAR": "Morning Doji Star",
#     "CDLMORNINGSTAR": "Morning Star",
#     "CDLONNECK": "On-Neck Pattern",
#     "CDLPIERCING": "Piercing Pattern",
#     "CDLRICKSHAWMAN": "Rickshaw Man",
#     "CDLRISEFALL3METHODS": "Rising/Falling Three Methods",
#     "CDLSEPARATINGLINES": "Separating Lines",
#     "CDLSHOOTINGSTAR": "Shooting Star",
#     "CDLSHORTLINE": "Short Line Candle",
#     "CDLSPINNINGTOP": "Spinning Top",
#     "CDLSTALLEDPATTERN": "Stalled Pattern",
#     "CDLSTICKSANDWICH": "Stick Sandwich",
#     "CDLTAKURI": "Takuri (Dragonfly Doji with very long lower shadow)",
#     "CDLTASUKIGAP": "Tasuki Gap",
#     "CDLTHRUSTING": "Thrusting Pattern",
#     "CDLTRISTAR": "Tristar Pattern",
#     "CDLUNIQUE3RIVER": "Unique 3 River",
#     "CDLUPSIDEGAP2CROWS": "Upside Gap Two Crows",
#     "CDLXSIDEGAP3METHODS": "Upside/Downside Gap Three Methods"
# }

