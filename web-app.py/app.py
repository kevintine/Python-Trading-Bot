import os
import sys
sys.path.append(os.path.abspath('../scripts'))
from pattern_detect import get_stock_with_symbol, get_candlestick_pattern_from_stock, get_current_date, get_past_date_by_day
import charts

from flask import Flask, render_template, request
from flask_scss import Scss
from static.patterns import patterns, stocks, pattern_descriptions
from flask import jsonify
import mplfinance as mpf


app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", patterns=patterns, stocks=stocks)

@app.route("/candlestick-chart", methods=["GET", "POST"])
def candlestick_chart():
    stock = request.args.get("stock")
    pattern = request.args.get("pattern")
    candlestick_chart_path = charts.get_chart_with_pattern(stock, pattern)
    pattern_image_path = charts.get_pattern_descriptions(pattern)
    return render_template("index.html", pattern_image_path=pattern_image_path, candlestick_chart_path=candlestick_chart_path,patterns=patterns, 
                           stocks=stocks)

if __name__ in "__main__":
    app.run(debug=True, port=5001)