# %% ### EQUITY FINANCIALS  — FETCH + PRIORITY + DIRECTION MAP

import requests
import pandas as pd

def fetch_financial_ratios(tickers, api_key, period="annual", years=5):
    """
    Fetch ratios + key metrics + income statement fields.
    Returns:
        df_financials        → full metric matrix
        df_priority_coeff    → priority weights
        df_direction_map     → direct / inverse scoring rules
        df_extraction_report → per-ticker extraction diagnostics (added)
    """

    # ============================================================
    # 1. INCOME STATEMENT MAP (new)
    # ============================================================

    INCOME_STATEMENT_MAP = {
        "revenue"         : "Revenue",
        "netIncome"       : "Net Income",
        "operatingIncome" : "Operating Income",
    }

    # ============================================================
    # 2. RATIOS + KEY METRICS MAP
    # ============================================================

    RATIOS_MAP = {
        "priceToEarningsRatio": "P/E ratio",
        "priceToSalesRatio": "P/S ratio",
        "priceToBookRatio" : "P/B ratio",
        "enterpriseValueMultiple" : "EV/EBITDA",

        "currentRatio": "Current Ratio",
        "quickRatio": "Quick Ratio",
        "debtToAssetsRatio": "Debt/Asset Ratio",
        "debtToEquityRatio": "Debt/Equity Ratio",
        "interestCoverageRatio": "Interest Coverage Ratio",

        "assetTurnover": "Assets Turnover Ratio",
        "grossProfitMargin": "Gross Profit Margin",
        "operatingProfitMargin": "Operating Profit Margin",
        "netProfitMargin": "Net Profit Margin",
    }

    KEY_METRICS_MAP = {
        "returnOnAssets": "ROA",
        "returnOnEquity": "ROE",
        "freeCashFlowYield" : "FCF Yield"
    }

    # ============================================================
    # 3. PRIORITY SCORES
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
        "P/S ratio": 1,

        # Growth (income statement)
        "Revenue": 2,
        "Operating Income": 2,
        "Net Income": 3,
    }

    # ============================================================
    # 4. DIRECTION MAP (direct / inverse)
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
    # 5. FETCHING LOOP
    # ============================================================

    if isinstance(tickers, str):
        tickers = [tickers]

    all_rows = []
    extraction_report = []  # <-- collect per-ticker diagnostics

    for ticker in tickers:
        # Initialize a report record for this ticker
        report = {
            "Ticker": ticker,
            "Status": "OK",
            "Error": None,
            "MissingIncomeFields": [],
            "MissingRatioFields": [],
            "MissingMetricFields": [],
            "ReturnedYears": 0,
        }

        print(f"\n📥 Fetching: {ticker}")

        # --- API endpoints ---
        url_income = (
            f"https://financialmodelingprep.com/stable/income-statement?"
            f"symbol={ticker}&period={period}&apikey={api_key}"
        )
        url_ratios = (
            f"https://financialmodelingprep.com/stable/ratios?"
            f"symbol={ticker}&period={period}&apikey={api_key}"
        )
        url_metrics = (
            f"https://financialmodelingprep.com/stable/key-metrics?"
            f"symbol={ticker}&period={period}&apikey={api_key}"
        )

        # --- fetch JSON ---
        try:
            inc_resp = requests.get(url_income)
            ratios_resp = requests.get(url_ratios)
            metrics_resp = requests.get(url_metrics)
        except Exception as e:
            report["Status"] = "REQUEST_ERROR"
            report["Error"] = str(e)
            print(f"❌ Error fetching {ticker}: {e}")
            extraction_report.append(report)
            continue

        # check HTTP statuses
        if inc_resp.status_code != 200:
            report["Status"] = "INCOME_HTTP_ERROR"
            report["Error"] = f"Income HTTP {inc_resp.status_code}"
            print(f"⚠ {ticker} income HTTP {inc_resp.status_code}")
            extraction_report.append(report)
            continue

        if ratios_resp.status_code != 200:
            report["Status"] = "RATIOS_HTTP_ERROR"
            report["Error"] = f"Ratios HTTP {ratios_resp.status_code}"
            print(f"⚠ {ticker} ratios HTTP {ratios_resp.status_code}")
            extraction_report.append(report)
            continue

        if metrics_resp.status_code != 200:
            report["Status"] = "METRICS_HTTP_ERROR"
            report["Error"] = f"Metrics HTTP {metrics_resp.status_code}"
            print(f"⚠ {ticker} key-metrics HTTP {metrics_resp.status_code}")
            extraction_report.append(report)
            continue

        try:
            inc_json = inc_resp.json()
            ratios_json = ratios_resp.json()
            metrics_json = metrics_resp.json()
        except Exception as e:
            report["Status"] = "JSON_DECODE_ERROR"
            report["Error"] = str(e)
            print(f"❌ JSON decode error for {ticker}: {e}")
            extraction_report.append(report)
            continue

        # --- check missing data ---
        if not isinstance(inc_json, list) or len(inc_json) == 0:
            report["Status"] = "NO_INCOME_DATA"
            report["Error"] = "income-statement empty or invalid"
            print(f"⚠️ Missing income-statement for {ticker}")
            extraction_report.append(report)
            continue

        if not isinstance(ratios_json, list) or len(ratios_json) == 0:
            report["Status"] = "NO_RATIOS_DATA"
            report["Error"] = "ratios empty or invalid"
            print(f"⚠️ Missing ratios for {ticker}")
            extraction_report.append(report)
            continue

        if not isinstance(metrics_json, list) or len(metrics_json) == 0:
            report["Status"] = "NO_KEYMETRICS_DATA"
            report["Error"] = "key-metrics empty or invalid"
            print(f"⚠️ Missing key-metrics for {ticker}")
            extraction_report.append(report)
            continue

        # Trim to last N years
        inc_json     = inc_json[:years]
        ratios_json  = ratios_json[:years]
        metrics_json = metrics_json[:years]

        # DataFrames
        df_income_raw = pd.DataFrame(inc_json)
        df_ratios_raw = pd.DataFrame(ratios_json)
        df_metrics_raw = pd.DataFrame(metrics_json)

        # helper: normalize year column
        def extract_year(df):
            if "calendarYear" in df.columns:
                df["Year"] = df["calendarYear"]
            elif "date" in df.columns:
                # some endpoints use 'date' like "2021-12-31"
                df["Year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
            else:
                df["Year"] = None
            return df[["Year"] + [c for c in df.columns if c != "Year"]]

        df_income_raw  = extract_year(df_income_raw)
        df_ratios_raw  = extract_year(df_ratios_raw)
        df_metrics_raw = extract_year(df_metrics_raw)

        # record returned years
        years_in_ratios = sorted(df_ratios_raw["Year"].dropna().unique().tolist())
        report["ReturnedYears"] = len(years_in_ratios)

        # --- MERGE YEAR BY YEAR ---
        for year in df_ratios_raw["Year"].unique():

            inc_row = (
                df_income_raw[df_income_raw["Year"] == year].iloc[0].to_dict()
                if year in df_income_raw["Year"].values else {}
            )
            ratios_row = df_ratios_raw[df_ratios_raw["Year"] == year].iloc[0].to_dict()
            metrics_row = (
                df_metrics_raw[df_metrics_raw["Year"] == year].iloc[0].to_dict()
                if year in df_metrics_raw["Year"].values else {}
            )

            merged = {"Ticker": ticker, "Year": year}

            # income statement
            for f, name in INCOME_STATEMENT_MAP.items():
                # if field missing in raw income, note it
                if f not in df_income_raw.columns:
                    if name not in report["MissingIncomeFields"]:
                        report["MissingIncomeFields"].append(name)
                merged[name] = inc_row.get(f)

            # ratios
            for f, name in RATIOS_MAP.items():
                if f not in df_ratios_raw.columns:
                    if name not in report["MissingRatioFields"]:
                        report["MissingRatioFields"].append(name)
                merged[name] = ratios_row.get(f)

            # key metrics
            for f, name in KEY_METRICS_MAP.items():
                if f not in df_metrics_raw.columns:
                    if name not in report["MissingMetricFields"]:
                        report["MissingMetricFields"].append(name)
                merged[name] = metrics_row.get(f)

            all_rows.append(merged)

        # append final ok report for ticker
        if report["Status"] == "OK":
            extraction_report.append(report)

    # ============================================================
    # 6. FINAL OUTPUT DATAFRAMES
    # ============================================================

    df_financials = pd.DataFrame(all_rows)

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
    API_KEY = "qKAL5tYjzDYub5SoUK98SrX8RbD4OJUV" # key 1
    API_KEY = "eStobmCXEMQ3XXAPdIhN7oycsELjuOta" # key 2
    companies = ["AAPL", "MSFT", "GOOGL", "CAT"]

    df_financials, df_priority_coeff, df_direction_map, df_extraction_report = fetch_financial_ratios(companies, API_KEY, years=5)

    print("\n--- FINANCIAL DATA ---")
    print(df_financials)

    print("\n--- PRIORITY COEFFICIENTS ---")
    print(df_priority_coeff)

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
