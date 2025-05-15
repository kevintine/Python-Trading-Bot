import psycopg2
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

def add_stock_with_history():
    hostname = 'localhost'
    username = 'postgres'
    pwd = '1234'
    database = 'tradingbot'
    port_id = 5432

    try:
        # Get symbol from user input
        symbol = input("Enter the stock symbol (e.g., RY.TO): ").strip().upper()
        
        # Connect to database
        conn = psycopg2.connect(
            host=hostname,
            user=username,
            password=pwd,
            dbname=database,
            port=port_id
        )
        cur = conn.cursor()

        # Check if symbol exists in stocks table
        cur.execute("SELECT stock_id, company_name FROM stocks WHERE symbol = %s", (symbol,))
        existing_stock = cur.fetchone()

        if not existing_stock:
            # Fetch stock info from Yahoo Finance
            print(f"\nAdding new stock: {symbol}")
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            company_name = info.get('shortName', symbol)
            exchange = info.get('exchange', 'TSX')
            
            # Insert new stock
            cur.execute("""
                INSERT INTO stocks 
                (symbol, company_name, exchange, is_active)
                VALUES (%s, %s, %s, %s)
                RETURNING stock_id
            """, (symbol, company_name, exchange, True))
            
            stock_id = cur.fetchone()[0]
            conn.commit()
            print(f"Added new stock: {company_name} ({symbol})")
        else:
            stock_id, company_name = existing_stock
            print(f"\nFound existing stock: {company_name} ({symbol})")
        
        # Check if symbol already has price data
        cur.execute("SELECT 1 FROM daily_prices WHERE stock_id = %s LIMIT 1", (stock_id,))
        if cur.fetchone():
            print(f"Warning: {symbol} already has price data in database")
            overwrite = input("Overwrite existing data? (y/n): ").lower()
            if overwrite != 'y':
                return

        # Fetch 4 years of historical data
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=4*365)
        
        print(f"\nFetching historical data for {symbol} from {start_date} to {end_date}...")
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            print("Error: No data found for this symbol")
            return

        # Insert data into daily_prices
        print(f"Inserting {len(data)} records...")
        inserted = 0
        
        for index, row in data.iterrows():
            try:
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
                inserted += 1
            except Exception as e:
                print(f"Error inserting {index.date()}: {str(e)}")
                conn.rollback()
                return

        conn.commit()
        print(f"\nSuccessfully inserted {inserted} records for {symbol}")

    except Exception as error:
        print(f"Error: {str(error)}")
        if 'conn' in locals():
            conn.rollback()
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    add_stock_with_history()