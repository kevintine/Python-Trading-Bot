# Usage: 
# Update all stocks with 1-hour intraday data. Run multiple times per day

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

    # 1. Get all active stock_ids and symbols (UNCHANGED)
    cur.execute("""
        SELECT s.stock_id, s.symbol 
        FROM stocks s
        WHERE s.is_active = TRUE
    """)
    stocks = cur.fetchall()
    print(f"Found {len(stocks)} active stocks")

    for stock_id, symbol in stocks:
        # 2. Get most recent datetime for this stock (UNCHANGED)
        cur.execute("""
            SELECT MAX(datetime) 
            FROM intraday_60min_prices 
            WHERE stock_id = %s
        """, (stock_id,))
        result = cur.fetchone()
        
        most_recent_datetime = result[0] if result[0] else None
        
        if most_recent_datetime:
            print(f"Stock ID {stock_id} last updated at {most_recent_datetime}")
            # Calculate period needed to update (UNCHANGED)
            start_date = most_recent_datetime + dt.timedelta(hours=1)
            end_date = dt.datetime.now()
            
            if start_date <= end_date:
                print(f"  Needs update from {start_date} to {end_date}")
                try:
                    data = yf.download(symbol, start=start_date, end=end_date, interval='1h')

                    for index, row in data.iterrows():
                        # Get the corresponding daily price_id
                        cur.execute("""
                            SELECT price_id FROM daily_prices
                            WHERE stock_id = %s AND date = %s
                        """, (stock_id, index.date()))
                        
                        price_row = cur.fetchone()
                        if not price_row:
                            print(f"  Skipping {symbol} @ {index} — no daily price found for {index.date()}")
                            continue  # Skip this row if no daily price entry exists

                        price_id = price_row[0]

                        cur.execute("""
                            INSERT INTO intraday_60min_prices 
                            (price_id, stock_id, datetime, open, high, low, close, adjusted_close, volume)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (stock_id, datetime) DO NOTHING
                        """, (
                            price_id,
                            stock_id,
                            index.to_pydatetime(),
                            float(row['Open']),
                            float(row['High']),
                            float(row['Low']),
                            float(row['Close']),
                            0.0,
                            int(row['Volume'])
                        ))

                    conn.commit()
                    print(f"Successfully updated {len(data)} records for {symbol}")

                except Exception as e:
                    print(f"Error updating {symbol}: {str(e)}")
                    conn.rollback()

            else:
                print("  Already up to date")
        else:
            print(f"\n{symbol} (ID: {stock_id}) has no intraday data - fetching 60 days")
            try:
                # Fetch 60 days of historical intraday data (UNCHANGED)
                end_date = dt.datetime.now()
                start_date = end_date - dt.timedelta(days=60)
                
                data = yf.download(symbol, start=start_date, end=end_date, interval='1h')
                
                # Add all historical data to database (ONLY CHANGE: Added price_id lookup)
                for index, row in data.iterrows():
                    # Get the corresponding daily price_id
                    cur.execute("""
                        SELECT price_id FROM daily_prices
                        WHERE stock_id = %s AND date = %s
                    """, (stock_id, index.date()))
                    price_id = cur.fetchone()[0]
                    
                    cur.execute("""
                        INSERT INTO intraday_60min_prices 
                        (price_id, stock_id, datetime, open, high, low, close, adjusted_close, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (stock_id, datetime) DO NOTHING
                    """, (
                        price_id,  # Now properly included
                        stock_id, 
                        index.to_pydatetime(), 
                        float(row['Open']), 
                        float(row['High']), 
                        float(row['Low']), 
                        float(row['Close']), 
                        0.0,  # Kept as original
                        int(row['Volume'])
                    ))
                
                conn.commit()
                print(f"  Added {len(data)} historical records for {symbol} (from {start_date} to {end_date})")
            except Exception as e:
                print(f"Error updating {symbol}: {str(e)}")
                conn.rollback()

    conn.close()
except Exception as error:
    print(error)