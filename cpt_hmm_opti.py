# %% HIDDEN MARKOV CHAIN

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import datetime as dt


class HMMRegimeModel:

    def __init__(self, ticker="MSFT", start="2015-01-01", end=None, n_states=3, freq="M"):
        self.ticker = ticker
        self.start = start
        self.end = end or dt.datetime.today().strftime("%Y-%m-%d")
        self.n_states = n_states
        self.freq = freq.upper()   # "D", "W", "M"

        self.df = None
        self.model = None
        self.regime_stats = None
        self.regime_labels = None   # <- numeric state -> 'bull' / 'bear' / 'neutral'
        self.df_backtest = None
        self.df_summary = None

    # ---------------------------------------------------------
    # INTERNAL: Max Drawdown
    # ---------------------------------------------------------
    def _max_drawdown(self, equity: pd.Series) -> float:
        roll_max = equity.cummax()
        dd = equity / roll_max - 1.0
        return dd.min()

    # ---------------------------------------------------------
    # 1. Load Data (DAILY → MONTHLY)
    # ---------------------------------------------------------
    def load_data(self):
        data = yf.download(
            self.ticker,
            start=self.start,
            end=self.end,
            interval="1d",
            auto_adjust=False
        )

        if data.empty:
            raise ValueError("Failed to download price data.")

        # Use Adj Close if available, otherwise Close
        if "Adj Close" in data.columns:
            close = data["Adj Close"].astype(float)
        elif "Close" in data.columns:
            close = data["Close"].astype(float)
        else:
            raise ValueError("No 'Adj Close' or 'Close' column found in price data.")

        df_daily = pd.DataFrame(index=data.index)
        df_daily["Close"] = close

        # RESAMPLE BASED ON freq
        resample_map = {
            "D": "1D",
            "W": "W",
            "M": "M"
        }

        rule = resample_map[self.freq]

        df = pd.DataFrame()
        df["Close"] = df_daily["Close"].resample(rule).last()

        # log-returns
        df["Return"] = np.log(df["Close"] / df["Close"].shift(1))
        df = df.dropna()

        self.df = df
    
    # ---------------------------------------------------------
    # 2. Fit HMM and classify regimes
    #    Features: Return (monthly), Vol(3m), Mom(12m), DD
    # ---------------------------------------------------------
    def fit(self):
        if self.df is None:
            self.load_data()

        base = self.df.copy()

        if self.freq == "D":
            vol_window = 20      # 1 month
            mom_window = 252     # 12 months
            ann_factor = 252
        elif self.freq == "W":
            vol_window = 4       # 1 month
            mom_window = 52      # 12 months
            ann_factor = 52
        elif self.freq == "M":
            vol_window = 3       # 3 months
            mom_window = 12      # 12 months
            ann_factor = 12
        

        # Monthly return
        Return = base["Return"].astype(float)

        # 3-month rolling volatility of monthly returns
        Vol = Return.rolling(vol_window).std()

        # 12-month momentum
        Mom = base["Close"] / base["Close"].shift(mom_window) - 1

        # Drawdown from monthly peak
        DD = base["Close"] / base["Close"].cummax() - 1

        # Return reversal (detects large negative → positive inflection)
        ReturnReversal = (-Return.shift(1).clip(upper=0)) * (Return.clip(lower=0))

        # Rebuild clean feature dataframe
        df = pd.DataFrame(index=base.index)
        df["Return"] = Return
        df["Vol"] = Vol
        df["Mom"] = Mom
        df["DD"] = DD
        df["ReturnReversal"] = ReturnReversal

        # Drop NaN rows (from rolling & lagging)
        df = df.dropna()

        # Safety: ensure we have enough observations
        if df.shape[0] < self.n_states * 5:
            raise ValueError(
                f"Not enough monthly observations after feature construction "
                f"({df.shape[0]} rows) for {self.n_states} HMM states. "
                f"Use an earlier start date or reduce lookback windows."
            )

        # Fit HMM
        X = df[["Return", "Vol", "Mom", "DD", "ReturnReversal"]].values

        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=500,
            random_state=42
        )
        self.model.fit(X)

        states = self.model.predict(X)
        df["Regime"] = states

        # Predict most likely regime
        states = self.model.predict(X)
        df["Regime"] = states

        # ---------------------------------
        # Compute regime stats
        # ---------------------------------
        regime_stats = (
            df.groupby("Regime")[["Return", "Vol", "Mom", "DD", "ReturnReversal"]]
            .mean()
            .sort_values("Mom", ascending=False)
        )

        # ---------------------------------
        # Build labels_map (bull / bear / neutral)
        # ---------------------------------
        ordered_states = list(regime_stats.index)
        labels_map = {}

        if len(ordered_states) == 1:
            labels_map[ordered_states[0]] = "bull"
        elif len(ordered_states) == 2:
            labels_map[ordered_states[0]] = "bull"
            labels_map[ordered_states[1]] = "bear"
        else:
            labels_map[ordered_states[0]] = "bull"
            labels_map[ordered_states[-1]] = "bear"
            for s in ordered_states[1:-1]:
                labels_map[s] = "neutral"

        # Add regime semantic label
        df["RegimeLabel"] = df["Regime"].map(labels_map)
        regime_stats["Label"] = regime_stats.index.map(labels_map)

        # Save to object BEFORE probability step
        self.regime_stats = regime_stats
        self.regime_labels = labels_map

        # ---------------------------------
        # Now compute posterior probabilities
        # ---------------------------------
        probas = self.model.predict_proba(X)

        # Add raw probabilities (Prob_0, Prob_1, ...)
        for state in range(self.n_states):
            df[f"Prob_{state}"] = probas[:, state]

        # Add semantic probabilities (Prob_bull, Prob_bear, Prob_neutral)
        for state, label in labels_map.items():
            df[f"Prob_{label}"] = probas[:, state]

        # ---------------------------------------------------------
        #  Build regime-to-regime shift probability table
        # ---------------------------------------------------------

        # HMM transition matrix (n_states × n_states)
        T = self.model.transmat_

        # Store shift probabilities for each time step
        shift_records = []

        for i in range(len(df)):
            row = {"Date": df.index[i]}

            # current posterior probabilities (vector of size n_states)
            p_t = probas[i]  # e.g. [0.70, 0.20, 0.10]

            # predicted next probability vector = p_t ⋅ T
            p_next = p_t @ T  # matrix multiplication

            # store probability for each regime
            for state in range(self.n_states):
                row[f"NextProb_{state}"] = p_next[state]

            # also store semantic labels
            for state, label in labels_map.items():
                row[f"NextProb_{label}"] = p_next[state]

            shift_records.append(row)

        # Convert into DataFrame
        self.regime_shifts = pd.DataFrame(shift_records).set_index("Date")

        # Save final df
        self.df = df

    # ---------------------------------------------------------
    # 3. Backtest (long-only in bull regime)
    # ---------------------------------------------------------
    def backtest(self):
        if self.regime_stats is None:
            self.fit()

        df = self.df.copy()

        # Smooth dynamic exposure = probability of bull
        df["Position"] = df["Prob_bull"].shift(1).fillna(0)   # shift 1 to avoid look-ahead

        # --- Strategy / BH incremental returns ---
        df["BH_Incr"] = df["Return"]
        df["Strat_Incr"] = df["Position"] * df["Return"]

        # --- Proper compounded equity calculation (recursive) ---
        df["Equity_BuyHold"] = (1 + df["BH_Incr"]).cumprod()
        df["Equity_Strategy"] = (1 + df["Strat_Incr"]).cumprod()

        df["PnL_Strategy_%"] = (df["Equity_Strategy"] - 1.0) * 100
        df["PnL_BuyHold_%"]   = (df["Equity_BuyHold"] - 1.0) * 100

        # Annualization factor from freq
        ann_factor = {"D": 252, "W": 52, "M": 12}[self.freq]

        total_strat = df["PnL_Strategy_%"].iloc[-1]
        total_bh    = df["PnL_BuyHold_%"].iloc[-1]

        ann_strat = (df["Equity_Strategy"].iloc[-1] ** (ann_factor / len(df)) - 1) * 100
        ann_bh    = (df["Equity_BuyHold"].iloc[-1]   ** (ann_factor / len(df)) - 1) * 100

        vol_strat = df["Strat_Incr"].std() * np.sqrt(ann_factor)
        vol_bh    = df["BH_Incr"].std()    * np.sqrt(ann_factor)

        sharpe_strat = ann_strat / (vol_strat * 100) if vol_strat > 0 else np.nan
        sharpe_bh    = ann_bh / (vol_bh * 100)    if vol_bh > 0 else np.nan

        mdd_strat = self._max_drawdown(df["Equity_Strategy"]) * 100
        mdd_bh    = self._max_drawdown(df["Equity_BuyHold"]) * 100

        # Save
        self.df_backtest = df
        self.df_summary = pd.DataFrame({
            "Metric": [
                "Total Return (%)",
                "Annualized Return (%)",
                "Annualized Volatility",
                "Sharpe Ratio",
                "Max Drawdown (%)"
            ],
            "Strategy": [total_strat, ann_strat, vol_strat, sharpe_strat, mdd_strat],
            "Buy & Hold": [total_bh, ann_bh, vol_bh, sharpe_bh, mdd_bh]
        })

        return self.df_summary

    # ---------------------------------------------------------
    # 4. Plot regimes on price (base 100 from monthly returns)
    # ---------------------------------------------------------
    def plot_regimes(self):
        if self.df is None or "Regime" not in self.df:
            self.fit()

        df = self.df.copy().reset_index()
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

        # Base 100 rebasing
        df["PriceBase100"] = np.exp(df["Return"].cumsum()) * 100

        fig = go.Figure()

        # PRICE LINE (gold)
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["PriceBase100"],
            mode="lines",
            name=f"{self.ticker} (Base 100)",
            line=dict(color="white", width=2)
        ))

        # Colors by economic meaning
        label_colors = {
            "bull": "white",
            "bear": "black",
            "neutral": "darkgrey",
        }

        shapes = []
        y_min = df["PriceBase100"].min()
        y_max = df["PriceBase100"].max()

        prev_reg = df["Regime"].iloc[0]
        prev_label = df["RegimeLabel"].iloc[0]
        start_date = df["Date"].iloc[0]

        # Build regime rectangles
        for i in range(1, len(df)):
            cur_reg = df["Regime"].iloc[i]
            cur_label = df["RegimeLabel"].iloc[i]
            if cur_reg != prev_reg:
                color = label_colors.get(prev_label, "grey")
                shapes.append(dict(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=start_date,
                    x1=df["Date"].iloc[i],
                    y0=y_min,
                    y1=y_max,
                    fillcolor=color,
                    line_width=0,
                    layer="below",
                    opacity=0.18
                ))
                start_date = df["Date"].iloc[i]
                prev_reg = cur_reg
                prev_label = cur_label

        # Final segment
        color = label_colors.get(prev_label, "grey")
        shapes.append(dict(
            type="rect",
            xref="x",
            yref="y",
            x0=start_date,
            x1=df["Date"].iloc[-1],
            y0=y_min,
            y1=y_max,
            fillcolor=color,
            line_width=0,
            layer="below",
            opacity=0.18
        ))

        # LEGEND using semantic labels
        seen_labels = set()
        for reg in sorted(df["Regime"].unique()):
            label = self.regime_labels.get(reg, f"regime_{reg}")
            if label in seen_labels:
                continue
            seen_labels.add(label)
            col = label_colors.get(label, "grey")
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode="lines",
                line=dict(color=col, width=6),
                name=label.capitalize()
            ))

        # ---------------------------------------------------------
        # BUY SIGNALS (when probability of next bull regime is high)
        # ---------------------------------------------------------

        buy_threshold = 0.60  # 60% probability required to trigger a buy

        if hasattr(self, "regime_shifts") and self.regime_shifts is not None:

            shifts = self.regime_shifts.reset_index()
            shifts.rename(columns={"Date": "DateShift"}, inplace=True)

            # Merge with df to align dates
            df_merged = df.merge(shifts, left_on="Date", right_on="DateShift", how="left")

            # Buy signal = price point where Prob_nextBull >= threshold
            buys = df_merged[df_merged["NextProb_bull"] >= buy_threshold]

            fig.add_trace(go.Scatter(
                x=buys["Date"],
                y=buys["PriceBase100"],
                mode="markers",
                name=f"Buy Signal (Bull>{buy_threshold*100:.0f}%)",
                marker=dict(color="gold", size=5, symbol="circle"),
            ))

        # LAYOUT
        fig.update_layout(
            title=f"{self.ticker} Regimes (HMM {self.n_states} states) — Bull / Bear / Neutral",
            xaxis_title="Date",
            yaxis_title="Price (Base 100)",
            shapes=shapes,
            template="plotly_dark",
            plot_bgcolor="#111111",
            paper_bgcolor="#111111",
            font=dict(color="white"),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            )
        )

        fig.show()
        return fig

    # ---------------------------------------------------------
    # 5. Plot backtest performance (Strategy vs Buy&Hold)
    # ---------------------------------------------------------
    def plot_backtest(self):
        if self.df_backtest is None:
            self.backtest()

        df = self.df_backtest.copy().reset_index()
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

        fig = go.Figure()

        # --- Strategy curve ---
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["Equity_Strategy"],
            mode="lines",
            name="HMM Strategy",
            line=dict(color="gold", width=2)
        ))

        # --- Buy & Hold curve ---
        fig.add_trace(go.Scatter(
            x=df["Date"],
            y=df["Equity_BuyHold"],
            mode="lines",
            name="Buy & Hold",
            line=dict(color="white", width=2)
        ))

        fig.update_layout(
            title=f"{self.ticker} — Strategy vs Buy & Hold",
            xaxis_title="Date",
            yaxis_title="Equity Curve (Base = 1)",
            template="plotly_dark",
            font=dict(color="white"),
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        fig.show()
        return fig


# USAGE EXAMPLE
if __name__ == "__main__":
    # ⚠️ Use an EARLY start date so 12m momentum & 3m vol have enough history
    model = HMMRegimeModel("MSFT", "2015-01-01", n_states=2, freq="W")

    model.fit()
    print("Regime stats:\n", model.regime_stats, "\n")

    summary = model.backtest()
    print("Backtest summary:\n", summary, "\n")

    # Plot regimes
    model.plot_regimes()

    print("Regime Shifts:\n", model.regime_shifts, "\n")
    model.regime_shifts.to_excel("hmm_restults.xlsx")

    # Backtest
    model.backtest()
    model.plot_backtest()

# %%
