### RISK ANALYSIS

# %% IMPORT - LIBS

# data
import pandas as pd, numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta

# finance
import yfinance as yf

# plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# stats
from scipy.stats.distributions import chi2
from scipy.stats import norm
from arch import arch_model

# %% CLASS

class RiskAnalysis():

    # INSTANCES VARIABLES

    def __init__(self):

        # df
        self.df_stdDev = pd.DataFrame()
        self.df_expectedLoss = pd.DataFrame()

        # dict
        self.dict_mc_results = None

        # plot
        self.fig_stdDev_ptfBench = None
        self.fig_el_ptfBench = None
        self.fig_mc_st = None

    # STANDARD DEVIATION

    def compute_stdDev(self, df_returns, freq="W", window=5):

        if freq == "D":
            df_stdDev = df_returns.rolling(window=window).std()
        else:
            df_stdDev = df_returns.resample(freq).std()
        
        df_stdDev.columns = ["Std. Dev."]

        # return df
        self.df_stdDev = df_stdDev

    def plot_std_dev_ptfBench(self, df_ptf_stdDev, df_bench_stdDev, title="Weekly Standard Deviation of Returns"):

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=df_ptf_stdDev.index, y=df_ptf_stdDev, mode='lines', name='PORTFOLIO - Std. Dev.', line=dict(color='gold')))
        fig.add_trace(go.Scatter(x=df_bench_stdDev.index, y=df_bench_stdDev, mode='lines', name='BENCHMARK - Std. Dev.', line=dict(color='white')))

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Standard Deviation (%)',
            template="plotly_dark"
        )

        self.fig_stdDev_ptfBench = fig

    # EXPECTED LOSS (EL)

    def setUp_garch(self, df_returns, mean="Zero", p=1,q=1):

        # set model
        model_garch_p_q = arch_model(df_returns, mean=mean, vol="GARCH", p=p,q=q)

        # fit data (store results)
        results_garch = model_garch_p_q.fit()
        # get results summary
        summary = results_garch.summary()

        return summary, results_garch

    def el_garch(self, df_returns):

        # Calculate VaR and CVaR
        confidence_level = 0.95

        # GARCH
        summary_garch_zero_1_1, results_garch_garch_zero_1_1 = self.setUp_garch(df_returns, mean="Zero", p=1,q=1)
        conditional_volatility = results_garch_garch_zero_1_1.conditional_volatility

        # Calculate VaR
        var = conditional_volatility * norm.ppf(1 - confidence_level)

        # Calculate CVaR
        cvar = conditional_volatility * (norm.pdf(norm.ppf(1 - confidence_level)) / (1 - confidence_level))

        # Create a DataFrame for plotting
        df_varCvarGarch = pd.DataFrame({
            'Date': df_returns.index,
            'VaR': var,  # Negative because VaR is a loss
            'CVaR': -cvar  # Negative because CVaR is a loss
        })

        return df_varCvarGarch

    def compute_el_garch(self, df_returns, freq="W", confidence_level=0.95):

        df_returns = df_returns.squeeze()
        df_resampled = df_returns.resample(freq).sum().dropna()

        var_values = []
        cvar_values = []
        dates = []

        for i in range(len(df_resampled)):
            try:
                model = arch_model(df_resampled.iloc[:i+1], mean="Zero", vol="GARCH", p=1, q=1)
                results = model.fit(disp="off")
                conditional_volatility = results.conditional_volatility.iloc[-1]

                var = conditional_volatility * norm.ppf(1 - confidence_level)
                cvar = conditional_volatility * (norm.pdf(norm.ppf(1 - confidence_level)) / (1 - confidence_level))

                var_values.append(var)
                cvar_values.append(-cvar)
                dates.append(df_resampled.index[i])

            except Exception as e:
                print(f"GARCH model failed at {df_resampled.index[i]}: {e}")
                var_values.append(np.nan)
                cvar_values.append(np.nan)
                dates.append(df_resampled.index[i])

        df_el = pd.DataFrame({
            "Date": dates,
            "VaR": var_values,
            "CVaR": cvar_values
        }).dropna()

        self.df_expectedLoss = df_el

    def plot_el_garch_ptfVsBench(self, df_ptf_varCvar, df_bench_varCvar, freq="W"):

        fig = go.Figure()

        datasets = [
            (df_ptf_varCvar, "PORTFOLIO"),
            (df_bench_varCvar, "BENCHMARK")
        ]

        colors = ["gold", "gold", "white", "white"]

        for idx, (risk_df, label) in enumerate(datasets):
            fig.add_trace(go.Scatter(
                x=risk_df["Date"],
                y=risk_df["VaR"],
                mode="lines",
                name=f"{label} VaR",
                line=dict(color=colors[2 * idx % len(colors)], dash="solid")
            ))
            fig.add_trace(go.Scatter(
                x=risk_df["Date"],
                y=risk_df["CVaR"],
                mode="lines",
                name=f"{label} CVaR",
                line=dict(color=colors[(2 * idx + 1) % len(colors)], dash="dot")
            ))

        fig.update_layout(
            title=f"VaR-95% & CVaR-95% (GARCH) (Frequency: {freq})",
            xaxis_title="Date",
            yaxis_title="VaR & CVaR (%)",
            template="plotly_dark"
        )

        self.fig_el_ptfBench = fig

    # MONTE CARLO STRESS TEST (MC ST)

    def setUp_garch_mc(self, df_returns, mean="Zero", p=1, q=1):
        # Rescale returns to improve numerical stability
        df_rescaled = df_returns * 100

        model = arch_model(df_rescaled, mean=mean, vol="GARCH", p=p, q=q)
        results = model.fit(disp="off")

        # Adjust volatility back to original scale
        results.conditional_volatility /= 100

        return results

    def compute_mc_st(self, df_returns, label="Portfolio", stressVolScenario=1.5, simulations=1000, days=40, confidence_level=0.95):
        df_returns = df_returns.squeeze()

        # === GARCH conditional volatility ===
        garch_results = self.setUp_garch_mc(df_returns)
        conditional_vol = garch_results.conditional_volatility
        latest_vol = conditional_vol.iloc[-1]

        mu = df_returns.mean()
        sigma = latest_vol

        simulated_returns = np.random.normal(mu, sigma, (simulations, days))
        stressed_returns = simulated_returns * stressVolScenario
        cumulative_returns = stressed_returns.sum(axis=1)

        # === VaR & CVaR ===
        var_1d = latest_vol * norm.ppf(1 - confidence_level)
        cvar_1d = latest_vol * norm.pdf(norm.ppf(1 - confidence_level)) / (1 - confidence_level)

        var_horizon = np.sqrt(days) * var_1d
        cvar_horizon = np.sqrt(days) * cvar_1d

        worst_5_percent = np.percentile(cumulative_returns, 5)
        print(f"[{label}] 5% worst-case return over {days} days: {worst_5_percent:.2%}")

        self.dict_mc_results = {
            "label": label,
            "cumulative_returns": cumulative_returns,
            "var_1d": var_1d,
            "cvar_1d": cvar_1d,
            "var_horizon": var_horizon,
            "cvar_horizon": cvar_horizon
        }

    def plot_mc_st(self, results_ptf, results_bench, days=40):

        fig = go.Figure()

        # === Benchmark histogram (background, cyan) ===
        fig.add_trace(go.Histogram(
            x=results_bench["cumulative_returns"],
            nbinsx=50,
            name=f"{results_bench['label']} {days}-Day Returns",
            opacity=0.5,
            marker_color="white"
        ))

        # === Portfolio histogram (foreground, gold) ===
        fig.add_trace(go.Histogram(
            x=results_ptf["cumulative_returns"],
            nbinsx=50,
            name=f"{results_ptf['label']} {days}-Day Returns",
            opacity=0.7,
            marker_color="gold"
        ))

        # === VaR & CVaR lines ===
        # Benchmark
        fig.add_vline(
            x=results_bench["var_horizon"],
            line=dict(color="white", dash="dash"),
            layer="below"
        )
        fig.add_vline(
            x=-results_bench["cvar_horizon"],
            line=dict(color="white", dash="dot"),
            layer="below"
        )

        # Portfolio
        fig.add_vline(
            x=results_ptf["var_horizon"],
            line=dict(color="gold", dash="dash"),
            layer="above"
        )
        fig.add_vline(
            x=-results_ptf["cvar_horizon"],
            line=dict(color="gold", dash="dot"),
            layer="above"
        )

        # Vertical line at 0
        fig.add_vline(
            x=0,
            line=dict(color="white", dash="solid", width=0.5)
        )

        # === Custom Legend Entries for VaR/CVaR ===
        # VaR (1d) - dash
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="white", dash="dash"),
            name="VaR (1d)"
        ))

        # CVaR (40d) - dot
        fig.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="white", dash="dot"),
            name=f"CVaR (1d)"
        ))

        fig.update_layout(
            title=f"Monte Carlo Stress Test - {days}-Day Horizon",
            xaxis_title="Cumulative Return",
            yaxis_title="Frequency",
            template="plotly_dark",
            bargap=0.02,
            barmode="overlay",
            # legend=dict(title="", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        self.fig_mc_st = fig

# %% EXECUTION

if __name__ == "__main__":

    # import stock data
    ticker_stock = "MSFT"
    ticker_bench = "^DJI"
    endDate = dt.datetime.today()
    startDate = endDate - relativedelta(years=5)

    df_stockPrice = yf.download(tickers=[ticker_stock, ticker_bench], start=startDate, end=endDate, auto_adjust=False)["Adj Close"]
    df_returns = df_stockPrice.pct_change().dropna()

    print(f"{ticker_stock} & {ticker_bench} - RETURNS DF:")
    print(df_returns)

    df_returns_stock = df_returns[f"{ticker_stock}"]
    df_returns_bench = df_returns[f"{ticker_bench}"]

    # initialize object
    risk_analysis = RiskAnalysis()

    ## STANDARD DEVIATION

        # compute standard dev for ptf & bench
    risk_analysis.compute_stdDev(df_returns_stock)
    df_stdDev_ptf = risk_analysis.df_stdDev
    risk_analysis.compute_stdDev(df_returns_bench)
    df_stdDev_bench = risk_analysis.df_stdDev

        # plot
    risk_analysis.plot_std_dev_ptfBench(df_stdDev_ptf, df_stdDev_bench)
    fig_stdDev_ptfBench = risk_analysis.fig_stdDev_ptfBench
    fig_stdDev_ptfBench.show()

    ## EXPECTED LOSS (GARCH)

        # compute expected loss for ptf & bench
    risk_analysis.compute_el_garch(df_returns_stock)
    df_el_stock = risk_analysis.df_expectedLoss
    risk_analysis.compute_el_garch(df_returns_bench)
    df_el_bench = risk_analysis.df_expectedLoss

        # plot
    risk_analysis.plot_el_garch_ptfVsBench(df_el_stock, df_el_bench)
    fig_el_ptfBench = risk_analysis.fig_el_ptfBench
    fig_el_ptfBench.show()

    ## MONTE CARLO

        # compute monte carlo results for ptf & bench
    risk_analysis.compute_mc_st(df_returns_stock, "Portfolio")
    dict_mc_results_stock = risk_analysis.dict_mc_results
    risk_analysis.compute_mc_st(df_returns_bench, "Benchmark")
    dict_mc_results_bench = risk_analysis.dict_mc_results

        # plot
    risk_analysis.plot_mc_st(dict_mc_results_stock, dict_mc_results_bench)
    fig_mc_st = risk_analysis.fig_mc_st
    fig_mc_st.show()

# %%
