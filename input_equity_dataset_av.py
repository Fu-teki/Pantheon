# %% ### EQUITY FINANCIALS — FETCH + PRIORITY + DIRECTION MAP (Alpha Vantage Version)

import requests
import pandas as pd
import numpy as np
import time

# --- Alpha Vantage Endpoints ---
BASE_URL = "https://www.alphavantage.co/query?"
FUNCTION_OVERVIEW = "OVERVIEW"
FUNCTION_INCOME = "INCOME_STATEMENT"
FUNCTION_BALANCE = "BALANCE_SHEET"

# --- Alpha Vantage Field Mapping ---
# Maps Alpha Vantage JSON key to your desired output name

# 1. INCOME STATEMENT FIELDS
AV_INCOME_MAP = {
    "totalRevenue": "Revenue",
    "netIncome": "Net Income",
    "operatingIncome": "Operating Income",
    "grossProfit": "Gross Profit", # Used for margin calculation
}

# 2. OVERVIEW/BALANCE SHEET FIELDS (For direct ratios and raw values)
AV_OVERVIEW_RATIOS_MAP = {
    "PERatio": "P/E ratio",
    "PriceToSalesRatio": "P/S ratio",
    "PriceToBookRatio": "P/B ratio",
    "EVToEBITDA": "EV/EBITDA",
    "ReturnOnEquityTTM": "ROE",
    "ReturnOnAssetsTTM": "ROA",
    # Note: FCF Yield is not directly available and must be calculated or approximated
}
# We will calculate margins and debt ratios manually using data from statements/overview

def fetch_financial_ratios(tickers, api_key, period="annual", years=5):
    """
    Fetch financial data (Income Statement, Ratios) using Alpha Vantage.
    
    Parameters:
        tickers (list or str): list of tickers
        api_key (str): Alpha Vantage API key
        period (str): 'annual' (default in AV) or 'quarterly'
        years (int): Number of historical periods to retrieve (max 5 in AV)
        
    Returns:
        df_financials       → full metric matrix
        df_priority_coeff   → priority weights
        df_direction_map    → direct / inverse scoring rules
        df_extraction_report → per-ticker extraction diagnostics
    """
    
    # ============================================================
    # 1. CONSTANT MAPS (Ratios, Priority, Direction)
    # ============================================================

    # NOTE: These maps are kept consistent with your original structure
    
    # Ratios used for manual calculation/placeholder names
    RATIOS_MAP = {
        "P/E ratio": "P/E ratio",
        "P/S ratio": "P/S ratio",
        "P/B ratio": "P/B ratio",
        "EV/EBITDA": "EV/EBITDA",

        "Current Ratio": "Current Ratio",
        "Quick Ratio": "Quick Ratio",
        "Debt/Asset Ratio": "Debt/Asset Ratio",
        "Debt/Equity Ratio": "Debt/Equity Ratio",
        "Interest Coverage Ratio": "Interest Coverage Ratio",

        "Assets Turnover Ratio": "Assets Turnover Ratio",
        "Gross Profit Margin": "Gross Profit Margin",
        "Operating Profit Margin": "Operating Profit Margin",
        "Net Profit Margin": "Net Profit Margin",
    }

    KEY_METRICS_MAP = {
        "ROA": "ROA",
        "ROE": "ROE",
        "FCF Yield": "FCF Yield"
    }

    PRIORITY_SCORES = {
        "ROE": 3, "ROA": 3, "Gross Profit Margin": 2, "Operating Profit Margin": 2, "Net Profit Margin": 3,
        "Interest Coverage Ratio": 2, "Current Ratio": 2, "Quick Ratio": 2, "Debt/Asset Ratio": 2, "Debt/Equity Ratio": 3,
        "Assets Turnover Ratio": 2, "P/E ratio": 2, "P/S ratio": 1,
        "Revenue": 2, "Operating Income": 2, "Net Income": 3,
    }

    DIRECTION_MAP = {
        "ROE": "direct", "ROA": "direct", "Gross Profit Margin": "direct", "Operating Profit Margin": "direct", "Net Profit Margin": "direct", 
        "Assets Turnover Ratio": "direct", "Interest Coverage Ratio": "direct", "Revenue": "direct", "Operating Income": "direct", 
        "Net Income": "direct", "Composite Growth": "direct",
        "Debt/Equity Ratio": "inverse", "Debt/Asset Ratio": "inverse", "Current Ratio": "inverse", "Quick Ratio": "inverse", 
        "P/E ratio": "inverse", "P/S ratio": "inverse",
    }

    # ============================================================
    # 2. FETCHING LOOP
    # ============================================================

    if isinstance(tickers, str):
        tickers = [tickers]

    all_rows = []
    extraction_report = [] 

    for i, ticker in enumerate(tickers):
        # Alpha Vantage free tier limits: 5 calls per minute, 500 per day
        if i > 0:
            print("⏳ Waiting 15 seconds to respect Alpha Vantage API limit (5 calls/min)...")
            time.sleep(15) 
            
        report = {
            "Ticker": ticker,
            "Status": "OK",
            "Error": None,
            "MissingIncomeFields": [],
            "MissingRatioFields": [],
            "MissingMetricFields": [],
            "ReturnedYears": 0,
        }

        print(f"\n📥 Fetching: {ticker} (Alpha Vantage)")

        try:
            # --- API calls ---
            # NOTE: Alpha Vantage does not have separate ratio/key metric endpoints.
            # Overview contains latest ratios; Statements contain historical data.
            
            # 1. Overview (Latest Ratios)
            url_overview = f"{BASE_URL}function={FUNCTION_OVERVIEW}&symbol={ticker}&apikey={api_key}"
            overview_resp = requests.get(url_overview)
            overview_json = overview_resp.json()
            
            # 2. Income Statement (Historical Income Data)
            # Alpha Vantage period filter is not exposed for statements; always annual/quarterly via function name
            url_income = f"{BASE_URL}function={FUNCTION_INCOME}&symbol={ticker}&apikey={api_key}"
            inc_resp = requests.get(url_income)
            inc_json = inc_resp.json()
            
            # 3. Balance Sheet (Historical Balance Data for ratios)
            url_balance = f"{BASE_URL}function={FUNCTION_BALANCE}&symbol={ticker}&apikey={api_key}"
            bal_resp = requests.get(url_balance)
            bal_json = bal_resp.json()


            # --- Check and Prepare DataFrames ---
            
            # Error check (Alpha Vantage often returns {"Note": "..."} or {"Error Message": "..."})
            if "Error Message" in overview_json or "Note" in overview_json:
                 report["Status"] = "API_ERROR"
                 report["Error"] = overview_json.get("Error Message") or overview_json.get("Note")
                 extraction_report.append(report)
                 print(f"❌ API Error for {ticker}: {report['Error']}")
                 continue
            
            
            # Convert financial statements to DataFrames
            
            # Statement function to handle Alpha Vantage JSON format (list of dicts)
            def av_json_to_df(json_data, data_key):
                if data_key in json_data and isinstance(json_data[data_key], list):
                    df = pd.DataFrame(json_data[data_key])
                    df = df.set_index("fiscalDateEnding").rename_axis("Date")
                    # Convert all values from string to numeric
                    for col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    return df.head(years)
                return pd.DataFrame()

            df_income_raw = av_json_to_df(inc_json, "annualReports")
            df_balance_raw = av_json_to_df(bal_json, "annualReports")
            
            if df_income_raw.empty or df_balance_raw.empty:
                report["Status"] = "NO_FINANCIALS_DATA"
                report["Error"] = "Income or Balance Sheet data missing/invalid."
                extraction_report.append(report)
                print(f"⚠️ Missing statement data for {ticker}")
                continue
                
            years_available = df_income_raw.index.tolist()
            report["ReturnedYears"] = len(years_available)
            
            # --- MERGE YEAR BY YEAR ---
            for date_index in years_available:
                year = pd.to_datetime(date_index).year
                
                # Extract data for the current year
                inc_row = df_income_raw.loc[date_index].to_dict()
                bal_row = df_balance_raw.loc[date_index].to_dict()
                
                merged = {"Ticker": ticker, "Year": year}

                # a) INCOME STATEMENT (Raw Values)
                for av_field, output_name in AV_INCOME_MAP.items():
                    merged[output_name] = inc_row.get(av_field)

                # b) CALCULATED RATIOS (Margins and Solvency)
                
                # i. Margins (calculated from Income Statement)
                merged["Gross Profit Margin"] = inc_row.get("grossProfit") / inc_row.get("totalRevenue", np.nan)
                merged["Operating Profit Margin"] = inc_row.get("operatingIncome") / inc_row.get("totalRevenue", np.nan)
                merged["Net Profit Margin"] = inc_row.get("netIncome") / inc_row.get("totalRevenue", np.nan)

                # ii. Solvency/Liquidity (calculated from Balance Sheet)
                try:
                    merged["Current Ratio"] = bal_row.get("totalCurrentAssets") / bal_row.get("totalCurrentLiabilities", np.nan)
                    # Debt/Equity = Total Liabilities / Total Shareholder Equity (AV field names used)
                    merged["Debt/Equity Ratio"] = bal_row.get("totalLiabilities") / bal_row.get("totalShareholderEquity", np.nan)
                    # Debt/Asset = Total Liabilities / Total Assets
                    merged["Debt/Asset Ratio"] = bal_row.get("totalLiabilities") / bal_row.get("totalAssets", np.nan)
                except Exception:
                    pass # Ignore ZeroDivisionError/NaN issues for ratios

                # c) LATEST PERIOD RATIOS from OVERVIEW
                # Since Overview only gives TTM/latest static ratios, we only fill for the latest year available
                if year == pd.to_datetime(years_available[0]).year:
                    for av_field, output_name in AV_OVERVIEW_RATIOS_MAP.items():
                        merged[output_name] = pd.to_numeric(overview_json.get(av_field), errors='coerce')

                # d) Add placeholdes for metrics not easily available (or set to NaN)
                merged["FCF Yield"] = np.nan # Not a direct field in AV
                merged["Quick Ratio"] = np.nan
                merged["Interest Coverage Ratio"] = np.nan
                merged["Assets Turnover Ratio"] = np.nan

                all_rows.append(merged)

            # append final ok report for ticker
            if report["Status"] == "OK":
                extraction_report.append(report)

        except Exception as e:
            report["Status"] = "GENERAL_ERROR"
            report["Error"] = str(e)
            print(f"❌ General error for {ticker}: {e}")
            extraction_report.append(report)
            continue

    # ============================================================
    # 3. FINAL OUTPUT DATAFRAMES
    # ============================================================

    # Compile the list of all expected output metric names
    all_metric_names = list(AV_INCOME_MAP.values()) + list(RATIOS_MAP.keys()) + list(KEY_METRICS_MAP.keys())
    # Remove duplicates
    all_metric_names = sorted(list(set(all_metric_names)))

    df_financials = pd.DataFrame(all_rows, columns=["Ticker", "Year"] + all_metric_names)
    
    # Ensure correct column order for statements which should be raw values (not ratios)
    financial_statement_cols = ['Revenue', 'Operating Income', 'Net Income']
    
    # Re-order the columns to place raw statement data first
    other_cols = [col for col in df_financials.columns if col not in ['Ticker', 'Year'] + financial_statement_cols]
    df_financials = df_financials[['Ticker', 'Year'] + financial_statement_cols + other_cols]


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
    # --- IMPORTANT: Replace with a real Alpha Vantage API key ---
    AV_API_KEY = "8BCVI9D96Z865P5K" 
    
    companies = ["AAPL", "MSFT", "GOOGL"]
    companies = ["CAT", "DG.PA"]

    # The API key must be functional for the code to execute successfully.
    if AV_API_KEY == "YOUR_ALPHA_VANTAGE_API_KEY":
        print("\n🛑 ERROR: Please replace 'YOUR_ALPHA_VANTAGE_API_KEY' with a valid Alpha Vantage key to run this example.")
    else:
        df_financials, df_priority_coeff, df_direction_map, df_extraction_report = fetch_financial_ratios(companies, AV_API_KEY, years=5)

        print("\n--- FINANCIAL DATA SAMPLE (Alpha Vantage) ---")
        print(df_financials.head(len(companies) * 2).transpose())

        print("\n--- PRIORITY COEFFICIENTS ---")
        print(df_priority_coeff.head())
        
        print("\n--- EXTRACTION REPORT ---")
        print(df_extraction_report)
        
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
    tickers = ["AAPL", "MSFT", "GOOGL"]
    prices, returns = fetch_price_history(tickers, start="2018-01-01")

    print(prices.tail())
    print(returns.tail())

# %%
