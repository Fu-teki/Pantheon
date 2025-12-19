### FETCH ETFS PRICES (FROM YF)

# %% RUN

import yfinance as yf
import pandas as pd

def fetch_etf_prices(etf_symbols, start_date, end_date):
    """
    Fetches closing prices for a list of ETFs from Yahoo Finance.

    Parameters:
        etf_symbols (list): List of ETF ticker symbols (e.g., ['SPY', 'QQQ', 'IWM'])
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
        pd.DataFrame: DataFrame containing the closing prices for each ETF
    """
    # Download historical data
    data = yf.download(etf_symbols, start=start_date, end=end_date)['Close']
    
    # Rename DataFrame for clarity
    df_etf_price = pd.DataFrame(data)
    
    return df_etf_price

if __name__ == "__main__":
    etf_list = ['SPY', 'QQQ', 'VTI']
    l_dates = ['2024-01-01', '2024-12-31']
    df_etf_price = fetch_etf_prices(etf_list, l_dates[0], l_dates[1])
    print(df_etf_price.head())

# %%
