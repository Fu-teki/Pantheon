# %% TECHNICAL ANALYSIS (OOP VERSION) — NO TA-LIB

import numpy as np
import pandas as pd
import yfinance as yf

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import datetime as dt
from dateutil.relativedelta import relativedelta


# ============================================================
# INDICATORS (NO TA-LIB)
# ============================================================
def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder smoothing = EMA with alpha = 1/period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def macd(close: pd.Series, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9):
    macd_line = ema(close, fastperiod) - ema(close, slowperiod)
    signal_line = macd_line.ewm(span=signalperiod, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def bbands(close: pd.Series, period: int = 20, nbdev: float = 2.0):
    mid = close.rolling(window=period, min_periods=period).mean()
    # TA-Lib uses population std (ddof=0)
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = mid + nbdev * std
    lower = mid - nbdev * std
    return upper, mid, lower


def cdl_engulfing(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Approximation of TA-Lib CDLENGULFING output:
      +100 bullish engulfing
      -100 bearish engulfing
      0 otherwise
    """
    prev_open = open_.shift(1)
    prev_close = close.shift(1)

    prev_bull = prev_close > prev_open
    prev_bear = prev_close < prev_open
    curr_bull = close > open_
    curr_bear = close < open_

    # "Engulf" real body (open/close), allowing equality to be inclusive
    bullish = prev_bear & curr_bull & (open_ <= prev_close) & (close >= prev_open)
    bearish = prev_bull & curr_bear & (open_ >= prev_close) & (close <= prev_open)

    out = pd.Series(0, index=close.index, dtype=float)
    out[bullish] = 100.0
    out[bearish] = -100.0
    return out


# ============================================================
# TECHNICAL ANALYSIS CLASS
# ============================================================
class TechnicalAnalysis:

    PRESETS = {
        "short": {
            "ema_range": (10, 30),
            "macd_range": (6, 19, 5),
            "rsi_period": 7,
            "bb_period": 10,
            "bb_std": 2,
        },
        "medium": {
            "ema_range": (30, 100),
            "macd_range": (12, 26, 9),
            "rsi_period": 14,
            "bb_period": 20,
            "bb_std": 2,
        },
        "long": {
            "ema_range": (50, 200),
            "macd_range": (24, 52, 18),
            "rsi_period": 21,
            "bb_period": 50,
            "bb_std": 2,
        }
    }

    def __init__(
        self,
        ticker,
        start_date,
        end_date,
        mode="medium",
        ema_range=None,
        macd_range=None,
        rsi_period=None,
        bb_period=None,
        bb_std=None
    ):

        self.ticker = ticker
        self.start = start_date
        self.end = end_date

        mode = mode.lower()
        if mode not in self.PRESETS:
            raise ValueError(f"Mode must be one of {list(self.PRESETS.keys())}")

        preset = self.PRESETS[mode]

        self.ema_range = ema_range or preset["ema_range"]
        self.macd_range = macd_range or preset["macd_range"]
        self.rsi_period = rsi_period or preset["rsi_period"]
        self.bb_period = bb_period or preset["bb_period"]
        self.bb_std = bb_std or preset["bb_std"]

        self.df = None
        self.macd_hist = None
        self.fig = None
        self.backtest_trades = None

    # ---------------------------------------------------------
    # 1) FETCH PRICE DATA
    # ---------------------------------------------------------
    def load_data(self):
        df = yf.download(self.ticker, start=self.start, end=self.end, auto_adjust=False)
        df.dropna(inplace=True)
        df = df.groupby(level=0, axis=1).first()  # avoid multi-index columns
        self.df = df

    # ---------------------------------------------------------
    # 2) COMPUTE INDICATORS (NO TA-LIB)
    # ---------------------------------------------------------
    def compute_indicators(self):
        close = self.df["Close"].astype(float)
        open_ = self.df["Open"].astype(float)
        high = self.df["High"].astype(float)
        low = self.df["Low"].astype(float)

        # EMA
        fast, slow = self.ema_range
        self.df[f"EMA_{fast}"] = ema(close, fast)
        self.df[f"EMA_{slow}"] = ema(close, slow)

        # RSI
        self.df["RSI"] = rsi_wilder(close, period=self.rsi_period)

        # MACD
        fastp, slowp, signalp = self.macd_range
        macd_line, macd_signal, macd_hist = macd(close, fastperiod=fastp, slowperiod=slowp, signalperiod=signalp)

        self.df["MACD"] = macd_line
        self.df["MACD_Signal"] = macd_signal
        self.macd_hist = macd_hist

        # Engulfing pattern
        self.df["ENGULFING"] = cdl_engulfing(open_, high, low, close)

        # Bollinger bands
        upper, middle, lower = bbands(close, period=self.bb_period, nbdev=self.bb_std)
        self.df["BB_Middle"] = middle
        self.df["BB_Upper"] = upper
        self.df["BB_Lower"] = lower

    # ---------------------------------------------------------
    # 3) BUY/SELL SIGNALS — MACD CROSSOVER
    # ---------------------------------------------------------
    def compute_macd_signals(self):
        macd_s = self.df["MACD"].to_numpy()
        signal_s = self.df["MACD_Signal"].to_numpy()
        close_s = self.df["Close"].to_numpy()

        buy = []
        sell = []
        flag = 0

        for i in range(len(self.df)):
            if macd_s[i] > signal_s[i]:
                sell.append(np.nan)
                if flag != 1:
                    buy.append(close_s[i])
                    flag = 1
                else:
                    buy.append(np.nan)
            else:
                buy.append(np.nan)
                if flag != 0:
                    sell.append(close_s[i])
                    flag = 0
                else:
                    sell.append(np.nan)

        self.df["MACD_Buy"] = buy
        self.df["MACD_Sell"] = sell

    # ---------------------------------------------------------
    # 4) BUY/SELL SIGNALS — RSI OVERBOUGHT/OVERSOLD
    # ---------------------------------------------------------
    def compute_rsi_signals(self):
        rsi_s = self.df["RSI"].to_numpy()
        close_s = self.df["Close"].to_numpy()

        buy = []
        sell = []

        for i in range(len(self.df)):
            if rsi_s[i] < 30:
                buy.append(close_s[i])
                sell.append(np.nan)
            elif rsi_s[i] > 70:
                sell.append(close_s[i])
                buy.append(np.nan)
            else:
                buy.append(np.nan)
                sell.append(np.nan)

        self.df["RSI_Buy"] = buy
        self.df["RSI_Sell"] = sell

    # ---------------------------------------------------------
    # 5) PLOT DASHBOARD
    # ---------------------------------------------------------
    def plot_dashboard(self):

        fig = make_subplots(
            rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.02,
            row_heights=[0.45, 0.1, 0.10, 0.10, 0.07],
            subplot_titles=("Price + EMA", "Volume", "RSI", "MACD", "Engulfing Pattern")
        )

        # ---------------- PRICE & EMAs ----------------
        fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df["Close"],
            mode="lines", name="Close", line=dict(color="gold")
        ), row=1, col=1)

        fast, slow = self.ema_range
        fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df[f"EMA_{fast}"],
            mode="lines", name=f"EMA {fast}", line=dict(dash="dash")
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df[f"EMA_{slow}"],
            mode="lines", name=f"EMA {slow}", line=dict(dash="dash")
        ), row=1, col=1)

        # Buy/Sell markers (MACD)
        fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df["MACD_Buy"],
            mode="markers", name="Buy MACD",
            marker=dict(color="cyan", symbol="triangle-up", size=6)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df["MACD_Sell"],
            mode="markers", name="Sell MACD",
            marker=dict(color="violet", symbol="triangle-down", size=6)
        ), row=1, col=1)

        # Buy/Sell markers (RSI)
        fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df["RSI_Buy"],
            mode="markers", name="Buy RSI",
            marker=dict(color="cyan", symbol="circle", size=6)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df["RSI_Sell"],
            mode="markers", name="Sell RSI",
            marker=dict(color="violet", symbol="circle", size=6)
        ), row=1, col=1)

        # Bollinger bands
        fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df["BB_Upper"],
            name="BB Upper", line=dict(color="white", width=1, dash="dot")
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df["BB_Middle"],
            name="BB Middle", line=dict(color="lightgrey", width=1)
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df["BB_Lower"],
            name="BB Lower", line=dict(color="white", width=1, dash="dot")
        ), row=1, col=1)

        # ---------------- VOLUME ----------------
        colors = np.where(self.df["Close"] >= self.df["Close"].shift(1), "cyan", "pink")
        fig.add_trace(go.Bar(
            x=self.df.index, y=self.df["Volume"], marker=dict(color=colors),
            name="Volume"
        ), row=2, col=1)

        # ---------------- RSI ----------------
        fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df["RSI"],
            mode="lines", name="RSI", line=dict(color="white")
        ), row=3, col=1)

        fig.add_hline(y=70, line=dict(color="red", dash="dash"), row=3, col=1)
        fig.add_hline(y=30, line=dict(color="cyan", dash="dash"), row=3, col=1)

        # ---------------- MACD ----------------
        fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df["MACD"], mode="lines", name="MACD"
        ), row=4, col=1)

        fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df["MACD_Signal"], mode="lines", name="Signal"
        ), row=4, col=1)

        fig.add_trace(go.Bar(
            x=self.df.index, y=self.macd_hist,
            name="Histogram",
            marker=dict(color=["red" if v < 0 else "green" for v in self.macd_hist])
        ), row=4, col=1)

        # ---------------- ENGULFING ----------------
        fig.add_trace(go.Scatter(
            x=self.df.index, y=self.df["ENGULFING"],
            mode="lines", name="Engulfing"
        ), row=5, col=1)

        fig.update_layout(
            title=f"Technical Analysis Dashboard — {self.ticker}",
            template="plotly_dark",
            height=1000,
            showlegend=True
        )

        self.fig = fig
        return fig

    # ---------------------------------------------------------
    # 6) FULL PIPELINE
    # ---------------------------------------------------------
    def run(self):
        self.load_data()
        self.compute_indicators()
        self.compute_macd_signals()
        self.compute_rsi_signals()
        return self.plot_dashboard()

    # ---------------------------------------------------------
    # 7) BACKTESTING ENGINE – MULTI-STRATEGY
    # ---------------------------------------------------------
    def compute_backtests(self):
        df = self.df.copy()

        def process_strategy(buy_col, sell_col):
            trades = []
            buy_price = None

            for i in range(len(df)):
                if not np.isnan(df[buy_col].iloc[i]) and buy_price is None:
                    buy_price = df["Close"].iloc[i]
                elif not np.isnan(df[sell_col].iloc[i]) and buy_price is not None:
                    sell_price = df["Close"].iloc[i]
                    pnl = (sell_price - buy_price) / buy_price
                    trades.append(pnl)
                    buy_price = None

            if len(trades) == 0:
                return 0.0, []

            return sum(trades), trades

        macd_total, macd_trades = process_strategy("MACD_Buy", "MACD_Sell")
        rsi_total, rsi_trades = process_strategy("RSI_Buy", "RSI_Sell")

        df["Engulf_Buy"] = np.where(df["ENGULFING"] > 0, df["Close"], np.nan)
        df["Engulf_Sell"] = np.where(df["ENGULFING"] < 0, df["Close"], np.nan)
        engulf_total, engulf_trades = process_strategy("Engulf_Buy", "Engulf_Sell")

        df["BB_Buy"] = np.where(df["Close"] <= df["BB_Lower"], df["Close"], np.nan)
        df["BB_Sell"] = np.where(df["Close"] >= df["BB_Upper"], df["Close"], np.nan)
        bb_total, bb_trades = process_strategy("BB_Buy", "BB_Sell")

        df_bt = pd.DataFrame({
            "MACD_PnL": [macd_total * 100],
            "RSI_PnL": [rsi_total * 100],
            "ENGULFING_PnL": [engulf_total * 100],
            "BB_PnL": [bb_total * 100],
        }, index=["P&L"]).round(2)

        self.backtest_trades = {
            "MACD_Trades": macd_trades,
            "RSI_Trades": rsi_trades,
            "ENGULFING_Trades": engulf_trades,
            "BB_Trades": bb_trades,
            "Summary": df_bt
        }

        return df_bt


# ============================================================
# USAGE EXAMPLE
# ============================================================
if __name__ == "__main__":
    ticker = "MSFT"
    end = dt.datetime.today()
    start = end - relativedelta(years=5)

    ta = TechnicalAnalysis(
        ticker=ticker,
        start_date=start,
        end_date=end,
        mode="medium"
    )

    fig = ta.run()
    fig.show()

    df_backtest = ta.compute_backtests()
    print(ta.backtest_trades)
    print(df_backtest)

# %%
