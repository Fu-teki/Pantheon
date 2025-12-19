# %% FULL QUANT MODEL — Growth, Quality, Value, Momentum, Risk

import numpy as np
import pandas as pd

from input_equity_dataset_yf import (
    fetch_financial_data,
    fetch_price_history,
)

# --------------------------------------------------------------
# Helper: percentile ranking
# --------------------------------------------------------------
def _percentile_rank(series: pd.Series) -> pd.Series:
    return series.rank(method="average", pct=True) * 100

# ==============================================================
# GROWTH FACTOR
# ==============================================================
class GrowthFactor:
    """
    Growth Factor using *priority-weighted* CAGR percentiles.
    """

    # All metrics that may be used for CAGR
    GROWTH_METRICS = [
        "Net Income",
        "Revenue",
        "Operating Income",
        "Gross Profit"

    ]

    def __init__(self, df_financials: pd.DataFrame, df_priority: pd.DataFrame):
        """
        df_financials: output of fetch_financial_data()
        df_priority: DataFrame with [Metric, Priority]
        """
        self.df = df_financials.copy()
        self.df_priority = df_priority.copy()

        # convert priority table to mapping:
        #   e.g. {"Revenue": 3, "Net Income": 2, ... }
        self.priority_map = dict(
            zip(self.df_priority["Metric"], self.df_priority["Priority"])
        )

        self.df_cagr = {}    # ticker -> {metric: CAGR%}
        self.scores = None   # final DataFrame with Growth factor scores

        # ensure sorted by year
        self.df.sort_values(["Ticker", "Year"], inplace=True)

    # --------------------------------------------------------------
    def _compute_cagr_per_ticker(self, df_t: pd.DataFrame) -> dict:
        """
        Compute CAGR (%) for each growth metric available for one ticker.
        CAGR = (last / first)^(1/years) - 1
        """
        cagr_dict = {}
        df_t = df_t.set_index("Year").sort_index()

        for metric in self.GROWTH_METRICS:

            if metric not in df_t.columns:
                continue

            series = df_t[metric].dropna()
            if len(series) < 2:
                continue

            first, last = series.iloc[0], series.iloc[-1]
            years = len(series) - 1

            try:
                cagr = (last / first) ** (1 / years) - 1
                cagr_dict[metric] = round(cagr * 100, 2)
            except Exception:
                continue

        return cagr_dict

    # --------------------------------------------------------------
    def compute(self):
        """
        Growth Score = PRIORITY-WEIGHTED average of CAGR percentiles.
        """

        # ---- Step 1: compute CAGR per ticker ----
        for ticker in self.df["Ticker"].unique():
            df_t = self.df[self.df["Ticker"] == ticker].copy()
            self.df_cagr[ticker] = self._compute_cagr_per_ticker(df_t)

        # Flatten to long format
        rows = []
        for ticker, metric_dict in self.df_cagr.items():
            for metric, val in metric_dict.items():
                rows.append({
                    "Ticker": ticker,
                    "Metric": metric,
                    "CAGR": val,
                    "Priority": self.priority_map.get(metric, 1)  # default priority = 1 (when don't exist)
                })

        df_cagr_all = pd.DataFrame(rows)

        if df_cagr_all.empty:
            self.scores = pd.DataFrame(columns=["Growth"])
            return self.scores

        # ---- Step 2: percentile score per metric ----
        df_cagr_all["Percentile"] = (
            df_cagr_all.groupby("Metric")["CAGR"]
            .transform(_percentile_rank)
        )

        # ---- Step 3: Apply priority weighting ----
        df_cagr_all["WeightedPct"] = (
            df_cagr_all["Percentile"] * df_cagr_all["Priority"]
        )

        # ---- Step 4: Final Growth Score = weighted average ----
        df_scores = (
            df_cagr_all.groupby("Ticker")
            .apply(lambda g: g["WeightedPct"].sum() / g["Priority"].sum())
            .to_frame(name="Growth")
        )

        self.scores = df_scores.round(2)
        return self.scores


# ==============================================================
# QUALITY FACTOR
# ==============================================================

class QualityFactor:
    """
    Quality = profitability + efficiency + financial strength.
    Uses level metrics (median over years) and direction map.
    """

    QUALITY_METRICS = [

        # Profit
        "Net Profit Margin",
        "Operating Profit Margin",
        "Gross Profit Margin",

        # Return
        "ROE",
        "ROA",
        "Assets Turnover Ratio",

        # Solvency
        "Interest Coverage Ratio",
        "Debt/Equity Ratio",
        "Debt/Asset Ratio",
        "Current Ratio",
        "Quick Ratio",
    ]

    def __init__(
        self,
        df_financials: pd.DataFrame,
        df_direction_map: pd.DataFrame,
    ):
        self.df = df_financials.copy()
        self.df_direction = df_direction_map.copy()
        self.scores = None

        self.df.sort_values(["Ticker", "Year"], inplace=True)

        # dict: metric -> "direct" or "inverse"
        self.direction_map = dict(
            zip(self.df_direction["Metric"], self.df_direction["Direction"])
        )

    def _metric_direction(self, metric: str) -> str:
        return self.direction_map.get(metric, "direct")

    def compute(self):
        rows = []

        for ticker in self.df["Ticker"].unique():
            df_t = (
                self.df[self.df["Ticker"] == ticker]
                .set_index("Year")
                .sort_index()
            )

            for metric in self.QUALITY_METRICS:
                if metric not in df_t.columns:
                    continue

                median_val = df_t[metric].median()
                if pd.isna(median_val):
                    continue

                rows.append(
                    {
                        "Ticker": ticker,
                        "Metric": metric,
                        "Value": median_val,
                    }
                )

        df_q = pd.DataFrame(rows)
        if df_q.empty:
            self.scores = pd.DataFrame(columns=["Quality"])
            return self.scores

        # raw percentile per metric
        df_q["Percentile"] = (
            df_q.groupby("Metric")["Value"]
            .transform(_percentile_rank)
        )

        # direction-adjusted percentile
        def _adjust(row):
            direction = self._metric_direction(row["Metric"])
            return 100 - row["Percentile"] if direction == "inverse" else row["Percentile"]

        df_q["AdjPercentile"] = df_q.apply(_adjust, axis=1)

        # simple avg of all quality metrics per ticker
        df_scores = (
            df_q.groupby("Ticker")["AdjPercentile"]
            .mean()
            .to_frame(name="Quality")
        )

        self.scores = df_scores.round(2)
        return self.scores


# ==============================================================
# VALUE FACTOR
# ==============================================================

class ValueFactor:
    """
    Value = cheapness based on valuation ratios.
    Higher value = cheaper on average.
    """

    VALUE_METRICS = [
        "P/E ratio",
        "P/B ratio",
        "P/S ratio",
        "EV/EBITDA",
        "FCF Yield",
    ]

    def __init__(self, df_financials: pd.DataFrame, df_direction_map: pd.DataFrame):
        self.df = df_financials.copy()
        self.df_direction = df_direction_map.copy()
        self.scores = None

        self.df.sort_values(["Ticker", "Year"], inplace=True)
        self.direction_map = dict(
            zip(self.df_direction["Metric"], self.df_direction["Direction"])
        )

    def _metric_direction(self, metric: str) -> str:
        return self.direction_map.get(metric, "direct")

    def compute(self):
        rows = []

        for ticker in self.df["Ticker"].unique():
            df_t = (
                self.df[self.df["Ticker"] == ticker]
                .set_index("Year")
                .sort_index()
            )

            for metric in self.VALUE_METRICS:
                if metric not in df_t.columns:
                    continue

                median_val = df_t[metric].median()
                if pd.isna(median_val):
                    continue

                rows.append(
                    {
                        "Ticker": ticker,
                        "Metric": metric,
                        "Value": median_val,
                    }
                )

        df_v = pd.DataFrame(rows)
        if df_v.empty:
            self.scores = pd.DataFrame(columns=["Value"])
            return self.scores

        # raw percentile per metric
        df_v["Percentile"] = (
            df_v.groupby("Metric")["Value"]
            .transform(_percentile_rank)
        )

        # apply direction (cheap = good)
        def _adjust(row):
            direction = self._metric_direction(row["Metric"])
            return 100 - row["Percentile"] if direction == "inverse" else row["Percentile"]

        df_v["AdjPercentile"] = df_v.apply(_adjust, axis=1)

        df_scores = (
            df_v.groupby("Ticker")["AdjPercentile"]
            .mean()
            .to_frame(name="Value")
        )

        self.scores = df_scores.round(2)
        return self.scores


# ==============================================================
# MOMENTUM FACTOR
# ==============================================================

class MomentumFactor:
    """
    Momentum = price performance over multiple horizons.
    Higher momentum = higher score.
    """

    def __init__(self, df_prices: pd.DataFrame):
        """
        df_prices: price history (Adj Close), columns = tickers
        """
        self.df_prices = df_prices.copy()
        self.scores = None

    def _trailing_return(self, prices: pd.Series, window: int) -> float:
        if len(prices) <= window:
            return np.nan
        return (prices.iloc[-1] / prices.iloc[-window]) - 1

    def compute(self):
        tickers = self.df_prices.columns
        rows = []

        for ticker in tickers:
            p = self.df_prices[ticker].dropna()
            if len(p) < 60:  # at least ~3 months
                continue

            # approximate windows: 252d, 126d, 63d
            r_12m = self._trailing_return(p, 252)
            r_6m = self._trailing_return(p, 126)
            r_3m = self._trailing_return(p, 63)

            # composite momentum (weights can be tuned)
            comp = (
                0.6 * (r_12m if pd.notna(r_12m) else 0) +
                0.3 * (r_6m if pd.notna(r_6m) else 0) +
                0.1 * (r_3m if pd.notna(r_3m) else 0)
            )

            rows.append({
                "Ticker": ticker,
                "MOM_raw": comp,
            })

        df_m = pd.DataFrame(rows)
        if df_m.empty:
            self.scores = pd.DataFrame(columns=["Momentum"])
            return self.scores

        df_m["Momentum"] = _percentile_rank(df_m["MOM_raw"])
        df_m.set_index("Ticker", inplace=True)
        self.scores = df_m[["Momentum"]].round(2)
        return self.scores


# ==============================================================
# RISK FACTOR
# ==============================================================

class RiskFactor:
    """
    Risk = market risk (volatility, beta, drawdown).
    Lower risk => higher score.
    """

    def __init__(
        self,
        df_prices: pd.DataFrame,
        df_returns: pd.DataFrame,
        bench_returns: pd.Series,
    ):
        self.df_prices = df_prices.copy()
        self.df_returns = df_returns.copy()
        self.bench_returns = bench_returns.dropna().copy()
        self.scores = None

    def _max_drawdown(self, prices: pd.Series) -> float:
        cum = prices / prices.iloc[0]
        roll_max = cum.cummax()
        drawdown = (cum - roll_max) / roll_max
        return drawdown.min()  # negative

    def _beta(self, r: pd.Series, rb: pd.Series) -> float:
        common_idx = r.index.intersection(rb.index)
        r_c = r.loc[common_idx]
        rb_c = rb.loc[common_idx]
        if len(r_c) < 30:
            return np.nan
        cov = np.cov(r_c, rb_c)[0, 1]
        var = np.var(rb_c)
        if var == 0:
            return np.nan
        return cov / var

    def compute(self):
        rows = []
        tickers = self.df_returns.columns

        for ticker in tickers:
            r = self.df_returns[ticker].dropna()
            p = self.df_prices[ticker].dropna()

            if len(r) < 60 or len(p) < 60:
                continue

            # use last 1Y if possible
            r_slice = r.tail(252)
            p_slice = p.tail(252)
            b_slice = self.bench_returns.tail(252)

            vol = r_slice.std() * np.sqrt(252)
            mdd = self._max_drawdown(p_slice)  # negative
            beta = self._beta(r_slice, b_slice)

            rows.append({
                "Ticker": ticker,
                "Volatility": vol,
                "MaxDrawdown": mdd,  # negative
                "Beta": beta,
            })

        df_r = pd.DataFrame(rows)
        if df_r.empty:
            self.scores = pd.DataFrame(columns=["Risk"])
            return self.scores

        # Convert MDD to positive magnitude for ranking
        df_r["MDD_mag"] = df_r["MaxDrawdown"].abs()

        # Percentiles (high value = high risk)
        for col in ["Volatility", "MDD_mag", "Beta"]:
            df_r[col + "_pct"] = _percentile_rank(df_r[col])

        # We want LOW risk => HIGH score => inverse percentiles
        df_r["Risk"] = (
            (100 - df_r["Volatility_pct"]) * 0.4 +
            (100 - df_r["MDD_mag_pct"]) * 0.4 +
            (100 - df_r["Beta_pct"]) * 0.2
        )

        df_r.set_index("Ticker", inplace=True)
        self.scores = df_r[["Risk"]].round(2)
        return self.scores


# ==============================================================
# MAIN QUANT MODEL ORCHESTRATOR
# ==============================================================

class QuantModel:
    """
    Industrial-grade quant factor model:
    - Growth
    - Quality
    - Value
    - Momentum
    - Risk
    - TotalFactorScore (0–100)
    """

    def __init__(
        self,
        api_key: str,
        tickers: list,
        start_price: str = "2018-01-01",
        end_price: str = None,
        benchmark_ticker: str = "SPY",
    ):
        self.api_key = api_key
        # Store original tickers for reference
        self.original_tickers = tickers 
        self.tickers = tickers
        self.start_price = start_price
        self.end_price = end_price
        self.benchmark_ticker = benchmark_ticker

        # Loaded later:
        self.df_financials = None
        self.df_priority = None
        self.df_direction = None
        self.df_extraction_report = None # Added report to track failures
        self.df_prices = None
        self.df_returns = None
        self.bench_returns = None
        
        # Track tickers that failed data loading
        self.failed_financials = []
        self.failed_prices = []
        
        # Factor score DataFrames
        self.growth_scores = None
        self.quality_scores = None
        self.value_scores = None
        self.momentum_scores = None
        self.risk_scores = None
        self.total_scores = None

    # ---------------- data loading ----------------
    def load_financials(self):
        # NOTE: fetch_financial_data must return an extraction report (df_extraction_report)
        # detailing which tickers failed or succeeded.
        (
            self.df_financials, 
            self.df_priority, 
            self.df_direction, 
            self.df_extraction_report
        ) = fetch_financial_data(
            self.tickers,
            self.api_key,
            years=5,
        )

        # 1. IDENTIFY FAILED FINANCIAL STOCKS
        if not self.df_financials.empty:
            # Get the list of tickers that have financial data
            successful_financial_tickers = self.df_financials['Ticker'].unique().tolist()
            
            # Find the missing ones
            self.failed_financials = [t for t in self.tickers if t not in successful_financial_tickers]
            
            # Update the working ticker list to only include successful ones
            self.tickers = successful_financial_tickers
            
            print(f"Financials loaded for {len(self.tickers)}/{len(self.original_tickers)} tickers.")
            if self.failed_financials:
                print(f"Skipped {len(self.failed_financials)} tickers due to failed financial data loading: {', '.join(self.failed_financials)}")
        else:
             print("❌ Failed to load financial data for all tickers.")
             self.tickers = []


    def load_prices(self):
        # Only attempt to load prices for tickers that successfully loaded financials (self.tickers)
        all_tickers = list(set(self.tickers + [self.benchmark_ticker]))
        
        # Check if there are any tickers left to process
        if not all_tickers or (len(all_tickers) == 1 and all_tickers[0] == self.benchmark_ticker):
             print("No tickers available for price loading.")
             return 

        df_prices_all, df_returns_all = fetch_price_history(
            all_tickers,
            start=self.start_price,
            end=self.end_price,
        )
        
        # Separate the benchmark
        if self.benchmark_ticker in df_returns_all.columns:
            self.bench_returns = df_returns_all[self.benchmark_ticker].copy()
        else:
            # Fallback: average of all returns as pseudo-benchmark
            other_returns = df_returns_all.drop(columns=[self.benchmark_ticker], errors='ignore')
            self.bench_returns = other_returns.mean(axis=1) if not other_returns.empty else pd.Series([], dtype=float)
            if self.bench_returns.empty:
                 print("❌ Benchmark data missing, and no other returns to use as fallback.")

        # Separate the stock data
        stock_columns = [t for t in self.tickers if t in df_prices_all.columns]
        
        self.df_prices = df_prices_all[stock_columns].copy()
        self.df_returns = df_returns_all[stock_columns].copy()
        
        # 2. IDENTIFY FAILED PRICE STOCKS
        self.failed_prices = [t for t in self.tickers if t not in stock_columns]
        
        # Final update to the working ticker list for scoring
        self.tickers = stock_columns
        
        print(f"Prices loaded for {len(self.tickers)}/{len(self.original_tickers)} tickers.")
        if self.failed_prices:
            print(f"Skipped {len(self.failed_prices)} tickers due to failed price data loading: {', '.join(self.failed_prices)}")
            
    # ---------------- factor computations ----------------
    def compute_factors(self):
        # We only compute factors for the self.tickers list which passed both loads.
        if not self.tickers:
            print("No tickers available for factor computation.")
            return

        # Growth
        gf = GrowthFactor(self.df_financials, self.df_priority)
        self.growth_scores = gf.compute().rename(columns={"Score": "Growth"})

        # Quality
        qf = QualityFactor(self.df_financials, self.df_direction)
        self.quality_scores = qf.compute().rename(columns={"Score": "Quality"})

        # Value
        vf = ValueFactor(self.df_financials, self.df_direction)
        self.value_scores = vf.compute().rename(columns={"Score": "Value"})

        # Momentum
        # df_prices must only contain the stocks we successfully loaded
        mf = MomentumFactor(self.df_prices) 
        self.momentum_scores = mf.compute().rename(columns={"Score": "Momentum"})

        # Risk
        # df_returns must only contain the stocks we successfully loaded
        rf = RiskFactor(self.df_prices, self.df_returns, self.bench_returns) 
        self.risk_scores = rf.compute().rename(columns={"Score": "Risk"})


    # ---------------- total factor score ----------------
    def aggregate_total_score(
        self,
        w_growth=0.20,
        w_quality=0.35,
        w_value=0.25,
        w_momentum=0.10,
        w_risk=0.10,
    ):
        # Use a list of factors that were successfully computed
        dfs = []
        if self.growth_scores is not None: dfs.append(self.growth_scores)
        if self.quality_scores is not None: dfs.append(self.quality_scores)
        if self.value_scores is not None: dfs.append(self.value_scores)
        if self.momentum_scores is not None: dfs.append(self.momentum_scores)
        if self.risk_scores is not None: dfs.append(self.risk_scores)
        
        if not dfs:
             return pd.DataFrame()

        # Merge all scores using an OUTER join on the index (Ticker)
        # This keeps any stock that successfully computed *at least one* factor score.
        # Missing scores will be NaN.
        df_all = pd.concat(dfs, axis=1, join="outer") 

        # Replace NaN scores with the average score for that factor, 
        # or 50 if the factor itself is all NaN. This is a robust way to handle
        # scoring where a stock might miss one factor (e.g., Value ratio unavailable)
        # without penalizing it with a zero.
        for col in df_all.columns:
            mean_score = df_all[col].mean()
            # If mean is NaN (all scores missing), use 50 (neutral score)
            fill_value = mean_score if not pd.isna(mean_score) else 50 
            df_all[col] = df_all[col].fillna(fill_value)

        df_all["TotalScore"] = (
            df_all["Growth"] * w_growth +
            df_all["Quality"] * w_quality +
            df_all["Value"] * w_value +
            df_all["Momentum"] * w_momentum +
            df_all["Risk"] * w_risk
        )
        
        # Add a column detailing which factors failed for each stock (optional but helpful)
        # This requires tracking which scores were NaN before filling (complex, so skipping for brevity)

        self.total_scores = df_all.round(2)
        return self.total_scores

    # ---------------- full pipeline ----------------
    def run(self):
        print(f"Starting QuantModel run for {len(self.original_tickers)} tickers.")
        
        # Load data – the self.tickers list is pruned here
        self.load_financials()
        self.load_prices()
        
        if not self.tickers:
            print("Analysis halted: No tickers passed both financial and price data checks.")
            return pd.DataFrame(columns=["TotalScore"])
            
        print(f"Proceeding with analysis for {len(self.tickers)} common tickers.")
        
        self.compute_factors()
        
        final_results = self.aggregate_total_score()
        
        # Report any failures (optional, but good practice)
        all_failed = list(set(self.failed_financials + self.failed_prices))
        if all_failed:
            print(f"\nSummary: {len(all_failed)} tickers were skipped entirely due to data failure: {', '.join(all_failed)}")
            
        return final_results

# ==============================================================
# USAGE
# ==============================================================

if __name__ == "__main__":
    API_KEY = "qKAL5tYjzDYub5SoUK98SrX8RbD4OJUV" # key 1
    API_KEY = "eStobmCXEMQ3XXAPdIhN7oycsELjuOta" # key 2
    
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "JNJ", "WMT", "CAT", "DG.PA", "BN.PA"]    

    tickers = [
        "MMM", "AXP", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "DIS", 
        "DOW", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "MCD", 
        "MRK", "MSFT", "NKE", "PG", "CRM", "TRV", "UNH", "VZ", "V", 
        "WBA", "WMT", "AMGN"
    ]

    qm = QuantModel(
        api_key=API_KEY,
        tickers=tickers,
        start_price="2018-01-01",
        benchmark_ticker="SPY",
    )

    df_scoring = qm.run()

    print("\n=== TOTAL QUANT FACTOR SCORES (0–100) ===")
    print(df_scoring.sort_values("TotalScore", ascending=False))

    print("\n=== Factor breakdown for AAPL ===")
    print(df_scoring.loc["AAPL"])

# %% 

