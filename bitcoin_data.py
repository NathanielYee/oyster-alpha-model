import yfinance as yf
import pandas as pd

def fetch_historical_data(symbol='BTC-USD', start_date='2020-01-01', end_date=None, interval='1d'):
    """
    Fetch historical data for a given symbol from Yahoo Finance.
    :param symbol: Ticker symbol, e.g., 'BTC-USD' for Bitcoin in USD.
    :param start_date: Start date for the data in 'YYYY-MM-DD' format.
    :param end_date: End date for the data in 'YYYY-MM-DD' format. Default is None (fetches up to today).
    :param interval: Data interval ('1d', '1h', '1wk', etc.).
    :return: DataFrame with historical data.
    """
    # Download data
    data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
    
    # Reset index to make timestamp a column
    data.reset_index(inplace=True)
    
    return data

# Fetch Bitcoin historical data
if __name__ == '__main__':
    symbol = 'BTC-USD'  # Bitcoin/USD ticker on Yahoo Finance
    start_date = '2020-01-01'  # Start date
    interval = '1d'  # Daily data
    
    data = fetch_historical_data(symbol, start_date, interval=interval)

    # Save to CSV
    filename = f'{symbol}_historical_data.csv'
    data.to_csv(filename, index=False)
    print(f"Historical data saved to {filename}")

    # Display sample
    print(data.head())

