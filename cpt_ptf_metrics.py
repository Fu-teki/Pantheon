# %% COMPUTE PORTFOLIO METRICS

import pandas as pd
import numpy as np
import plotly.graph_objects as go

from input_etfs_dataset import fetch_etf_prices


class PortfolioMetrics:

    def __init__(
        self,
        df_prices,
        weights=None,
        risk_free_rate=0.0,
        portfolio="YES",
        benchmark_ticker=None,
        start_date=None,
        end_date=None,
    ):
        """
        Initialize the PortfolioMetrics object.
        """
        # Work on a copy to avoid side effects
        self.df_prices = df_prices.copy()

        # ====== MISSING DATA REPORT ======
        missing_counts = self.df_prices.isna().sum()
        print("🔍 Missing data per ticker (before cleaning):")
        print(missing_counts[missing_counts > 0])

        # Drop columns that are entirely NaN (no usable data)
        self.df_prices = self.df_prices.dropna(axis=1, how="all")

        # Forward-fill and backward-fill prices (standard for financial series)
        self.df_prices = self.df_prices.ffill().bfill()

        # If any NaNs are still present (edge cases), fill with column mean
        # (this prevents metrics from blowing up while limiting distortion)
        self.df_prices = self.df_prices.apply(
            lambda col: col.fillna(col.mean())
        )

        # Now compute returns on the cleaned price data
        self.df_returns = self.df_prices.pct_change().dropna()

        self.risk_free_rate = risk_free_rate
        self.portfolio = portfolio.upper()
        self.benchmark_ticker = benchmark_ticker

        # Equal weights if not provided
        if weights is None:
            self.weights = np.ones(len(self.df_prices.columns)) / len(self.df_prices.columns)
        else:
            self.weights = np.array(weights)

        self.df_metrics = None

        # ===== Fetch benchmark prices and returns if provided =====
        self.benchmark = None
        self.bench_returns = None

        if benchmark_ticker is not None:
            # Infer dates if not explicitly provided
            if start_date is None:
                start_date = self.df_prices.index.min().strftime("%Y-%m-%d")
            if end_date is None:
                end_date = self.df_prices.index.max().strftime("%Y-%m-%d")

            bench_df = fetch_etf_prices([benchmark_ticker], start_date, end_date)

            # Align benchmark with portfolio index & clean similarly
            bench_series = bench_df[benchmark_ticker].reindex(self.df_prices.index)

            # Basic fill for benchmark too
            bench_series = bench_series.ffill().bfill()
            bench_series = bench_series.fillna(bench_series.mean())

            self.benchmark = bench_series
            self.bench_returns = self.benchmark.pct_change().dropna()


    # ----------------------------------------------------------------------
    # Helper: maximum drawdown
    @staticmethod
    def max_drawdown(returns: pd.Series) -> float:
        """
        Compute maximum drawdown from a returns series.
        """
        cum_index = (1 + returns).cumprod()
        running_max = cum_index.cummax()
        drawdowns = cum_index / running_max - 1.0
        return drawdowns.min()

    # ----------------------------------------------------------------------
    def compute_metrics(self):
        """
        Compute key performance metrics.

        - If portfolio="YES": metrics for Portfolio and Benchmark (if provided),
          with metrics as rows and columns = ['Portfolio', 'Benchmark'].
        - If portfolio="NO": independent asset analysis (one row per ETF).
        """
        if self.portfolio != "YES":
            return self._compute_asset_metrics()

        # ===== Portfolio-level returns =====
        port_ret = self.df_returns.dot(self.weights)

        # Annualization factor (assuming daily data)
        periods_per_year = 252

        # Portfolio metrics
        port_cum = (1 + port_ret).prod() - 1
        port_ann = port_ret.mean() * periods_per_year
        port_vol = port_ret.std() * np.sqrt(periods_per_year)
        port_sharpe = (port_ann - self.risk_free_rate) / port_vol if port_vol != 0 else np.nan

        port_mdd = self.max_drawdown(port_ret)
        port_calmar = port_ann / abs(port_mdd) if port_mdd != 0 else np.nan

        port_downside = port_ret[port_ret < 0]
        port_downside_vol = port_downside.std() * np.sqrt(periods_per_year) if len(port_downside) > 0 else np.nan
        port_sortino = (
            (port_ann - self.risk_free_rate) / port_downside_vol
            if port_downside_vol not in (0, np.nan) else np.nan
        )

        # ===== Benchmark metrics (if available) =====
        if self.bench_returns is not None and len(self.bench_returns) > 0:
            # Align portfolio and benchmark by date for comparative stats
            port_ret_aligned, bench_ret_aligned = port_ret.align(self.bench_returns, join="inner")

            bench_cum = (1 + bench_ret_aligned).prod() - 1
            bench_ann = bench_ret_aligned.mean() * periods_per_year
            bench_vol = bench_ret_aligned.std() * np.sqrt(periods_per_year)
            bench_sharpe = (bench_ann - self.risk_free_rate) / bench_vol if bench_vol != 0 else np.nan

            bench_mdd = self.max_drawdown(bench_ret_aligned)
            bench_calmar = bench_ann / abs(bench_mdd) if bench_mdd != 0 else np.nan

            bench_downside = bench_ret_aligned[bench_ret_aligned < 0]
            bench_downside_vol = bench_downside.std() * np.sqrt(periods_per_year) if len(bench_downside) > 0 else np.nan
            bench_sortino = (
                (bench_ann - self.risk_free_rate) / bench_downside_vol
                if bench_downside_vol not in (0, np.nan) else np.nan
            )

            # Beta (portfolio vs benchmark) using daily returns
            daily_cov = port_ret_aligned.cov(bench_ret_aligned)
            daily_var_bench = bench_ret_aligned.var()
            beta = daily_cov / daily_var_bench if daily_var_bench != 0 else np.nan

            # Tracking error and information ratio
            excess_ret = port_ret_aligned - bench_ret_aligned
            tracking_error = excess_ret.std() * np.sqrt(periods_per_year)
            info_ratio = (
                (port_ann - bench_ann) / tracking_error
                if tracking_error not in (0, np.nan) else np.nan
            )

            # Correlation
            corr = port_ret_aligned.corr(bench_ret_aligned)

            # "by definition" values for benchmark vs itself
            beta_bench = 1.0
            tracking_error_bench = 0.0
            info_ratio_bench = np.nan
            corr_bench = 1.0

        else:
            # No benchmark: set benchmark metrics to NaN
            bench_cum = bench_ann = bench_vol = bench_sharpe = np.nan
            bench_mdd = bench_calmar = bench_sortino = np.nan
            beta = tracking_error = info_ratio = corr = np.nan
            beta_bench = tracking_error_bench = info_ratio_bench = corr_bench = np.nan

        # ===== Build final DataFrame (metrics as rows) =====
        index = [
            "Cumulative Return",
            "Annualized Return",
            "Annualized Volatility",
            "Sharpe Ratio",
            "Sortino Ratio",
            "Max Drawdown",
            "Calmar Ratio",
            "Beta vs Benchmark",
            "Tracking Error",
            "Information Ratio",
            "Correlation vs Benchmark",
        ]

        portfolio_values = [
            port_cum,
            port_ann,
            port_vol,
            port_sharpe,
            port_sortino,
            port_mdd,
            port_calmar,
            beta,
            tracking_error,
            info_ratio,
            corr,
        ]

        benchmark_values = [
            bench_cum,
            bench_ann,
            bench_vol,
            bench_sharpe,
            bench_sortino,
            bench_mdd,
            bench_calmar,
            beta_bench,
            tracking_error_bench,
            info_ratio_bench,
            corr_bench,
        ]

        self.df_metrics = pd.DataFrame(
            {
                "Portfolio": portfolio_values,
                "Benchmark": benchmark_values,
            },
            index=index,
        ).round(4)

        return self.df_metrics

    # ----------------------------------------------------------------------
    def _compute_asset_metrics(self):
        """
        Independent ETF analysis (no portfolio aggregation).
        Returns DataFrame with assets as rows and metrics as columns.
        """
        periods_per_year = 252
        metrics = {}

        for etf in self.df_returns.columns:
            r = self.df_returns[etf]
            cum = (1 + r).prod() - 1
            ann = r.mean() * periods_per_year
            vol = r.std() * np.sqrt(periods_per_year)
            sharpe = (ann - self.risk_free_rate) / vol if vol != 0 else np.nan
            mdd = self.max_drawdown(r)

            metrics[etf] = [cum, ann, vol, sharpe, mdd]

        self.df_metrics = pd.DataFrame(
            metrics,
            index=[
                "Cumulative Return",
                "Annualized Return",
                "Annualized Volatility",
                "Sharpe Ratio",
                "Max Drawdown",
            ],
        ).T.round(4)

        return self.df_metrics

    # === FIGURES (Plotly) ===
    def plot_price_chart(self):
        """
        Returns a Plotly figure of ETF prices (or portfolio price if portfolio="YES"),
        normalized to 100 at the start date.
        If a benchmark is provided and portfolio="YES", it is overlaid.
        """
        fig = go.Figure()

        if self.portfolio == "YES":
            # Portfolio index
            portfolio_returns = self.df_returns.dot(self.weights)
            portfolio_index = (1 + portfolio_returns).cumprod()
            portfolio_index = portfolio_index / portfolio_index.iloc[0] * 100  # Base 100

            fig.add_trace(go.Scatter(
                x=portfolio_index.index,
                y=portfolio_index,
                mode='lines',
                name='Portfolio',
                line=dict(width=2)
            ))

            # Optional benchmark overlay
            if self.benchmark is not None and len(self.benchmark) > 0:
                bench_index = self.benchmark / self.benchmark.iloc[0] * 100
                fig.add_trace(go.Scatter(
                    x=bench_index.index,
                    y=bench_index,
                    mode='lines',
                    name=f'Benchmark ({self.benchmark_ticker})',
                    line=dict(width=2, dash='dash')
                ))

            title = "Portfolio Price (Base 100)"

        else:
            # Individual ETF price evolution
            for col in self.df_prices.columns:
                normalized_price = self.df_prices[col] / self.df_prices[col].iloc[0] * 100
                fig.add_trace(go.Scatter(
                    x=self.df_prices.index,
                    y=normalized_price,
                    mode='lines',
                    name=col
                ))
            title = "ETF Price Evolution (Base 100)"

        fig.update_layout(
            template='plotly_dark',
            title=title,
            xaxis_title="Date",
            yaxis_title="Normalized Price (Base 100)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

    def plot_std_chart(self, window=20):
        """
        Returns a Plotly time-series figure of rolling standard deviation (volatility).
        If portfolio="YES" and a benchmark exists, shows both.
        """
        fig = go.Figure()

        if self.portfolio == "YES":
            # Portfolio rolling volatility
            portfolio_returns = self.df_returns.dot(self.weights)
            rolling_vol = portfolio_returns.rolling(window).std() * np.sqrt(252)
            fig.add_trace(go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol,
                mode='lines',
                name='Portfolio Rolling Volatility',
                line=dict(width=2, color='orange')
            ))

            # Benchmark rolling volatility
            if self.bench_returns is not None and len(self.bench_returns) > 0:
                bench_rolling_vol = self.bench_returns.rolling(window).std() * np.sqrt(252)
                fig.add_trace(go.Scatter(
                    x=bench_rolling_vol.index,
                    y=bench_rolling_vol,
                    mode='lines',
                    name=f'Benchmark ({self.benchmark_ticker}) Rolling Volatility',
                ))

            title = f"Portfolio {window}-Day Rolling Volatility"

        else:
            # Individual ETF rolling volatilities
            for col in self.df_returns.columns:
                rolling_vol = self.df_returns[col].rolling(window).std() * np.sqrt(252)
                fig.add_trace(go.Scatter(
                    x=rolling_vol.index,
                    y=rolling_vol,
                    mode='lines',
                    name=f"{col} Volatility"
                ))
            title = f"{window}-Day Rolling Volatility per ETF"

        fig.update_layout(
            template='plotly_dark',
            title=title,
            xaxis_title="Date",
            yaxis_title="Annualized Volatility",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        return fig

# %% RUN

if __name__ == "__main__":

    # === Step 1: Fetch ETF Prices ===
    etf_list = ['SPY', 'QQQ', 'VTI', "^STOXX"]
    etf_list=["IWDA.AS", "IWQU.L", "^DJI", "^IXIC", 
              "ZPRR.DE", "^STOXX", "EUMD.L", "AEEM.MI", 
              "ICGA.DE", "CMDY", "BTC-USD" ,"ETH-USD"]

    l_dates = ['2024-01-01', '2024-12-31']
    df_etf_price = fetch_etf_prices(etf_list, l_dates[0], l_dates[1])
    print(df_etf_price)

    # === Step 2: Portfolio analysis (with benchmark) ===
    # l_weights = [0.5, 0.3, 0.2]
    weights = 1/len(etf_list)
    l_weights = [weights] * len(etf_list)
    print("weights:", l_weights)

    benchmark_ticker = 'SPY'  # Example: S&P 500 ETF as benchmark

    analyzer_portfolio = PortfolioMetrics(
        df_etf_price,
        weights=l_weights,
        risk_free_rate=0.0,
        portfolio="YES",
        benchmark_ticker=benchmark_ticker,
        start_date=l_dates[0],
        end_date=l_dates[1],
    )

    df_portfolio_metrics = analyzer_portfolio.compute_metrics()
    print("\n📊 Portfolio vs Benchmark Metrics:\n", df_portfolio_metrics)

    fig_price_port = analyzer_portfolio.plot_price_chart()
    fig_std_port = analyzer_portfolio.plot_std_chart()
    fig_price_port.show()
    fig_std_port.show()

    # === Step 3: Independent asset analysis (unchanged mechanism) ===
    analyzer_assets = PortfolioMetrics(df_etf_price, portfolio="NO")
    df_asset_metrics = analyzer_assets.compute_metrics()
    print("\n📈 Individual ETF Metrics:\n", df_asset_metrics)

    fig_price_ind = analyzer_assets.plot_price_chart()
    fig_std_ind = analyzer_assets.plot_std_chart()
    fig_price_ind.show()
    fig_std_ind.show()

# %%
