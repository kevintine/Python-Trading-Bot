import psycopg2
import datetime as dt
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

hostname = 'localhost'
username = 'postgres'
pwd = '1234'
database = 'tradingbot'
port_id = 5432

try:
    conn = psycopg2.connect(
        host=hostname,
        user=username,
        password=pwd,
        dbname=database,
        port=port_id
    )
    cur = conn.cursor()

    # 1. Get all active stock_ids and symbols
    cur.execute("""
        SELECT s.stock_id, s.symbol 
        FROM stocks s
        WHERE s.is_active = TRUE
    """)
    stocks = cur.fetchall()
    print(f"Found {len(stocks)} active stocks")

    for stock_id, symbol in stocks:
        cur.execute("""
            SELECT MAX(date) 
            FROM daily_prices 
            WHERE stock_id = %s
        """, (stock_id,))
        result = cur.fetchone()
        most_recent_date = result[0] if result[0] else None

        if most_recent_date:
            print(f"\n{symbol} last updated on {most_recent_date}")
            start_date = most_recent_date + dt.timedelta(days=1)
        else:
            print(f"\n{symbol} has no data, fetching 4 years")
            start_date = dt.date.today() - dt.timedelta(days=4 * 365)

        end_date = dt.date.today() 

        if start_date > end_date:
            print("  Already up to date")
            continue

        print(f"  Downloading from {start_date} to {end_date}")
        try:
            data = yf.download(symbol, start=start_date, end=end_date)

            if data.empty:
                print("  No new data returned from yfinance.")
                continue

            print(f"  Retrieved {len(data)} rows from yfinance.")
            for index, row in data.iterrows():
                trade_date = index.date()
                cur.execute("""
                    INSERT INTO daily_prices 
                    (stock_id, date, open, high, low, close, adjusted_close, volume)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (stock_id, date) DO NOTHING
                """, (
                    stock_id,
                    trade_date,
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    float(row.get('Adj Close', 0.0)),
                    int(row['Volume'])
                ))

            conn.commit()
            print(f"  Inserted data for dates: {', '.join(str(d.date()) for d in data.index)}")

        except Exception as e:
            print(f"Error updating {symbol}: {e}")
            conn.rollback()

    conn.close()

except Exception as error:
    print(f"Connection error: {error}")
