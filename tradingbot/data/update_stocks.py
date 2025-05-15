# Usage:
# Update all stocks to most recent day. Run once a day

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
        query2 = """
        SELECT MAX(date) 
        FROM daily_prices 
        WHERE stock_id = %s
        """
        cur.execute(query2, (stock_id,))
        result = cur.fetchone()
        
        most_recent_date = result[0] if result[0] else None
        print(most_recent_date)
        if most_recent_date:
            print(f"Stock ID {stock_id} last updated on {most_recent_date}")
            # Calculate days needed to update (from most_recent_date + 1 to today)
            start_date = most_recent_date + dt.timedelta(days=1)
            end_date =dt.datetime.now().date()
            
            if start_date <= end_date:
                print(f"  Needs update from {start_date} to {end_date}")
                try:
                    data = yf.download(symbol, start=start_date, end=end_date)
                    
                    # Add data to database
                    for index, row in data.iterrows():
                        cur.execute("""
                            INSERT INTO daily_prices 
                            (stock_id, date, open, high, low, close, adjusted_close, volume)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (stock_id, date) DO NOTHING
                        """, (
                            stock_id, 
                            index.date(), 
                            float(row['Open']), 
                            float(row['High']), 
                            float(row['Low']), 
                            float(row['Close']), 
                            0.0,  # Set adjusted_close to 0
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
            print(f"\n{symbol} (ID: {stock_id}) has no historical data - fetching 4 years")
            try:
                # Fetch 4 years of historical data
                end_date = dt.datetime.now().date()
                start_date = end_date - dt.timedelta(days=4*365)
                
                data = yf.download(symbol, start=start_date, end=end_date)
                
                # Add all historical data to database
                for index, row in data.iterrows():
                    cur.execute("""
                        INSERT INTO daily_prices 
                        (stock_id, date, open, high, low, close, adjusted_close, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (stock_id, date) DO NOTHING
                    """, (
                        stock_id, 
                        index.date(), 
                        float(row['Open']), 
                        float(row['High']), 
                        float(row['Low']), 
                        float(row['Close']), 
                        float(row['Adj Close']), 
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