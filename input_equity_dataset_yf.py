# %% ### EQUITY FINANCIALS — FETCH + PRIORITY + DIRECTION MAP (yfinance version)

import yfinance as yf
import pandas as pd
import numpy as np

# NOTE: API_KEY and period parameters are kept for compatibility but are NOT used by yfinance.
def fetch_financial_data(tickers, api_key=None, period="annual", years=5):
    """
    Fetch financial data (Income Statement, Balance Sheet, Key Ratios) 
    using yfinance only.
    
    Returns:
        df_financials       → full metric matrix (historical data is limited by yfinance)
        df_priority_coeff   → priority weights
        df_direction_map    → direct / inverse scoring rules
        df_extraction_report → per-ticker extraction diagnostics
    """

    # --- YFINANCE FIELD MAPPING ---
    
    # 1. INCOME STATEMENT FIELDS (Directly from .financials)
    # yfinance uses descriptive names (e.g., 'Total Revenue')
    INCOME_STATEMENT_MAP = {
        "Total Revenue"     : "Revenue",
        "Net Income"        : "Net Income",
        "Operating Income"  : "Operating Income",
        "Gross Profit"      : "Gross Profit",
    }

    # 2. RATIOS + KEY METRICS MAP (From .info, .financials, .balance_sheet)
    # yfinance.info often provides the most recent value for these
    YF_INFO_MAP = {
        "forwardPE"         : "P/E ratio",
        "priceToSalesTrailing12Months"      : "P/S ratio",
        "priceToBook"       : "P/B ratio",
        "enterpriseToEbitda": "EV/EBITDA",
        "returnOnEquity"    : "ROE",
        "returnOnAssets"    : "ROA",
        "freeCashflow"      : "FCF", # FCF is a raw number, not yield
    }
    
    # We will manually calculate margins and balance sheet ratios from the statements.

    # ============================================================
    # 3. PRIORITY SCORES (Kept the same for desired output structure)
    # ============================================================

    PRIORITY_SCORES = {
        # Profitability — HIGH
        "ROE": 3,
        "ROA": 3,
        "Gross Profit Margin": 2,
        "Operating Profit Margin": 2,
        "Net Profit Margin": 3,

        # Solvency — MEDIUM
        "Interest Coverage Ratio": 2,
        "Current Ratio": 2,
        "Quick Ratio": 2,
        "Debt/Asset Ratio": 2,
        "Debt/Equity Ratio": 3,

        # Efficiency — MEDIUM
        "Assets Turnover Ratio": 2,

        # Valuation
        "P/E ratio": 2,
        "P/Sales ratio": 1,

        # Growth (income statement)
        "Net Income": 3,
        "Revenue": 2,
        "Operating Income": 2,
        "Gross Profit": 2
    }

    # ============================================================
    # 4. DIRECTION MAP (Kept the same)
    # ============================================================

    DIRECTION_MAP = {
        # High = GOOD
        "ROE": "direct",
        "ROA": "direct",
        "Gross Profit Margin": "direct",
        "Operating Profit Margin": "direct",
        "Net Profit Margin": "direct",
        "Assets Turnover Ratio": "direct",
        "Interest Coverage Ratio": "direct",

        # Growth metrics (income statement)
        "Revenue": "direct",
        "Operating Income": "direct",
        "Net Income": "direct",
        "Composite Growth": "direct",

        # Low = GOOD (inverse)
        "Debt/Equity Ratio": "inverse",
        "Debt/Asset Ratio": "inverse",
        "Current Ratio": "inverse",
        "Quick Ratio": "inverse",
        "P/E ratio": "inverse",
        "P/S ratio": "inverse",
    }

    # ============================================================
    # 5. FETCHING LOOP (Rewritten for yfinance)
    # ============================================================

    if isinstance(tickers, str):
        tickers = [tickers]

    all_rows = []
    extraction_report = [] 

    for ticker in tickers:
        report = {
            "Ticker": ticker,
            "Status": "OK",
            "Error": None,
            "MissingIncomeFields": [],
            "MissingRatioFields": [],
            "MissingMetricFields": [],
            "ReturnedYears": 0,
        }

        print(f"\n📥 Fetching: {ticker} (using yfinance)")
        
        try:
            # 1. Get Ticker Object
            stock = yf.Ticker(ticker)

            # 2. Get Raw Statements and Info (yfinance is typically the last 4 years)
            df_income_raw = stock.financials.T 
            df_balance_raw = stock.balance_sheet.T
            # df_cashflow_raw = stock.cashflow.T # Not strictly needed for the mapped metrics
            info = stock.info # Contains most recent valuation and profitability ratios

            # Check if data exists
            if df_income_raw.empty or df_balance_raw.empty or not info:
                 report["Status"] = "NO_DATA"
                 report["Error"] = "yfinance returned empty dataframes or info"
                 extraction_report.append(report)
                 print(f"⚠️ Missing financial data for {ticker}")
                 continue

            # Limit to the requested number of years
            df_income_raw = df_income_raw.head(years)
            df_balance_raw = df_balance_raw.head(years)
            
            # Record returned years
            years_available = df_income_raw.index.tolist()
            report["ReturnedYears"] = len(years_available)
            
            # --- MERGE YEAR BY YEAR ---
            for date_index in years_available:
                year = date_index.year
                
                # Extract data for the current year
                inc_row = df_income_raw.loc[date_index].to_dict()
                bal_row = df_balance_raw.loc[date_index].to_dict()
                
                merged = {"Ticker": ticker, "Year": year}

                # a) INCOME STATEMENT (Revenue, Net Income, Operating Income)
                for yf_field, output_name in INCOME_STATEMENT_MAP.items():
                    value = inc_row.get(yf_field)
                    merged[output_name] = value

                # b) CALCULATED RATIOS (Margins and Solvency)
                try:
                    # Profitability Margins (requires Gross Profit, Operating Expenses, Net Income, Revenue)
                    merged["Gross Profit Margin"] = inc_row.get("Gross Profit", np.nan) / inc_row.get("Total Revenue", np.nan)
                    merged["Operating Profit Margin"] = inc_row.get("Operating Income", np.nan) / inc_row.get("Total Revenue", np.nan)
                    merged["Net Profit Margin"] = inc_row.get("Net Income", np.nan) / inc_row.get("Total Revenue", np.nan)
                    
                    # Solvency/Liquidity Ratios (requires Balance Sheet items)
                    total_debt = bal_row.get("Total Debt", bal_row.get("Long Term Debt", 0) + bal_row.get("Short Term Debt", 0))                    

                    # Ratios
                    merged["Current Ratio"] = bal_row.get("Current Assets", np.nan) / bal_row.get("Current Liabilities", np.nan)
                    merged["Interest Coverage Ratio"] = inc_row.get("EBIT", np.nan) / inc_row.get("Interest Expense", np.nan)
                    merged["Debt/Equity Ratio"] = total_debt / bal_row.get("Total Equity Gross Minority Interest", np.nan)
                    merged["Debt/Asset Ratio"] = total_debt / bal_row.get("Total Assets", np.nan)

                except ZeroDivisionError:
                    print(f"⚠ Skipping ratio calculations for {ticker} in {year} due to zero denominator.")
                    pass # Keep as NaN if division by zero occurs
                except Exception as e:
                    print(f"❌ Error calculating ratios for {ticker} in {year}: {e}")
                    pass
                
                # c) LATEST PERIOD RATIOS from .info (Valuation and Pre-Calculated Profitability)
                # These are usually only the *most recent* data, so we fill them only for the latest year available
                if year == years_available[0].year:
                    for yf_field, output_name in YF_INFO_MAP.items():
                        merged[output_name] = info.get(yf_field)
                        
                all_rows.append(merged)

        except Exception as e:
            report["Status"] = "YFINANCE_ERROR"
            report["Error"] = str(e)
            print(f"❌ General yfinance error for {ticker}: {e}")
            extraction_report.append(report)
            continue

        # append final ok report for ticker
        if report["Status"] == "OK":
            extraction_report.append(report)

    # ============================================================
    # 6. FINAL OUTPUT DATAFRAMES
    # ============================================================
    
    # We must explicitly define all columns that may be missing in the merged data
    all_metric_names = list(INCOME_STATEMENT_MAP.values()) + list(YF_INFO_MAP.values()) + [
        "Gross Profit Margin", "Operating Profit Margin", "Net Profit Margin",
        "Current Ratio", "Debt/Equity Ratio", "Debt/Asset Ratio", "Interest Coverage Ratio"
    ]
    
    # Ensure columns exist and order them
    df_financials = pd.DataFrame(all_rows, columns=["Ticker", "Year"] + all_metric_names)

    df_priority_coeff = pd.DataFrame(
        {"Metric": list(PRIORITY_SCORES.keys()),
         "Priority": list(PRIORITY_SCORES.values())}
    )

    df_direction_map = pd.DataFrame(
        {"Metric": list(DIRECTION_MAP.keys()),
         "Direction": list(DIRECTION_MAP.values())}
    )

    df_extraction_report = pd.DataFrame(extraction_report)

    return df_financials, df_priority_coeff, df_direction_map, df_extraction_report

# EXAMPLE USAGE
if __name__ == "__main__":

    tickers = ["AAPL", "MSFT", "GOOGL", "DG.PA", "BN.PA"]

    df_financials, df_priority_coeff, df_direction_map, df_extraction_report = fetch_financial_data(tickers, years=3)

    print("\n--- FINANCIAL DATA SAMPLE (yfinance) ---")
    print(df_financials.head(6).transpose()) # Use transpose to better view many columns

    print("\n--- PRIORITY COEFFICIENTS ---")
    print(df_priority_coeff.head())

    print("\n--- EXTRACTION REPORT ---")
    print(df_extraction_report)

    # # to excel test
    # df_financials.to_excel(r"C:\Users\amint\Desktop\PANTHEON RESEARCH\PYTHON\1_code\text.xlsx")

# %% PRICE HISTORY FETCHER (for Momentum & Risk)

import yfinance as yf
import pandas as pd

def fetch_price_history(tickers, start="2015-01-01", end=None, interval="1d"):
    """
    Fetch historical price data for Momentum & Risk factor calculations.

    Parameters:
        tickers (list or str): list of tickers or single ticker
        start (str): start date (YYYY-MM-DD)
        end   (str): end date (YYYY-MM-DD)
        interval (str): "1d", "1wk", "1mo"

    Returns:
        df_prices (DataFrame): Adjusted Close price history
        df_returns (DataFrame): daily returns
    """

    if isinstance(tickers, str):
        tickers = [tickers]

    print(f"📥 Downloading price history for: {tickers}")

    df = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    # Fix multi-index for single ticker case
    if isinstance(df.columns, pd.MultiIndex):
        df_prices = df["Adj Close"].copy()
    else:
        df_prices = df.rename(columns={"Adj Close": tickers[0]})

    # Drop missing columns
    df_prices = df_prices.dropna(axis=1, how="all")

    # Compute returns
    df_returns = df_prices.pct_change().dropna()

    return df_prices, df_returns

# USAGE
if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "DG.PA", "BN.PA"]
    prices, returns = fetch_price_history(tickers, start="2018-01-01")

    print(prices.tail())
    print(returns.tail())

# %% TEST YF FETCH

# import yfinance as yf
# import pandas as pd

# # 1. Define the ticker symbol and the desired item
# ticker_symbol = "DG.PA"
# item = "Interest Expense" # Item to search for

# # 2. Create the Ticker object
# stock = yf.Ticker(ticker_symbol)

# # 3. Access the 'balance_sheet' and retrieve the item
# try:
#     # Get the balance sheet data (lines are items, columns are dates)
#     df_data = stock.income_stmt
    
#     # Select the specific item (row) from the balance sheet DataFrame
#     item_data_series = df_data.loc[item]
    
#     # Get the latest value, which is the first element (index 0) of the Series
#     latest_item_value = item_data_series.iloc[0] 

#     # 4. Print the result
#     print(f"💰 {item} for {ticker_symbol} (Latest Annual):")
#     # Format for better readability (with comma separators and $ sign)
#     print(f"${latest_item_value:,.0f}")
    
# except KeyError:
#     # This exception catches if the exact item name is not found in the balance sheet
#     print(f"Error: '{item}' key not found in the balance sheet for {ticker_symbol}. Check the exact spelling/naming in the Yahoo Finance data.")
#     print("Available balance sheet items (head):")
#     print(df_data.head().index.tolist())
# except Exception as e:
#     print(f"An unexpected error occurred: {e}")
    
# %%
