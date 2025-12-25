### MARKET CONSENSUS (yfinance)

# %% IMPORT LIBS
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go

# %% CLASS - MARKET CONSENSUS

class MarketConsensus:

    def __init__(self, portfolio):
        """
        Initializes the MarketConsensus class.

        Args:
            portfolio (list): List of ticker symbols.
        """
        # data
        self.portfolio = portfolio
        self.df_consensus = None

        # plot
        self.fig_marketConsensus = None

    def get_stock_consensus(self, ticker):
        """
        Retrieves market consensus data for a single stock.

        Args:
            ticker (str): Stock ticker symbol.

        Returns:
            pd.DataFrame: DataFrame with consensus data for the stock.
        """
        try:
            stock = yf.Ticker(ticker)

            consensus_data = {
                "Reco Key": stock.info.get("recommendationKey"),
                "Reco Mean": stock.info.get("recommendationMean"),
                "Current Price": stock.info.get("currentPrice"),
                "Target Mean Price": stock.info.get("targetMeanPrice"),
                "Target Low Price": stock.info.get("targetLowPrice"),
                "Target High Price": stock.info.get("targetHighPrice"),
            }

            df_consensus = pd.DataFrame([consensus_data], index=[ticker])
            return df_consensus.round(1)

        except Exception as e:
            print(f"⚠ Error retrieving data for {ticker}: {e}")
            return None

    def build_portfolio_consensus(self):
        """
        Builds a DataFrame with consensus data for the entire portfolio.
        """
        df_all = pd.DataFrame()

        for ticker in self.portfolio:
            df_stock = self.get_stock_consensus(ticker)
            if df_stock is not None:
                df_all = pd.concat([df_all, df_stock], ignore_index=False)

        if df_all.empty:
            print("⚠ No data retrieved for any tickers.")
            return None

        # Sort by recommendation mean
        df_all.sort_values(by=["Reco Mean"], ascending=True, inplace=True)

        # Add ranking column
        df_all.insert(0, "Rank", range(1, len(df_all) + 1))

        self.df_consensus = df_all

    def create_plotly_table(self,title="Data Table"):
        """
        Creates a Plotly table from a Pandas DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to display.
            title (str): Title of the table.

        Returns:
            go.Figure: Plotly Figure object.
        """
        df = self.df_consensus.round(2)

        fig = go.Figure(data=[go.Table(
            header=dict(
                values=["Ticker"] + list(df.columns),
                align="center",
                font=dict(size=12, color="white"),
                fill_color="black",
                line_color="black"
            ),
            cells=dict(
                values=[df.index] + [df[col] for col in df.columns],
                align="center",
                fill_color=[["black"] * len(df.index)] +
                           [["#2E2E2E"] * len(df.index) for _ in df.columns],
                font=dict(size=11, color="white"),
                line_color="black"
            ),
        )])

        fig.update_layout(
            title=title,
            title_font=dict(size=16, color="white"),
            paper_bgcolor="#1C1C1C",
            plot_bgcolor="#1C1C1C"
        )
        self.fig_marketConsensus = fig

# %% EXECUTE

if __name__ == "__main__":
    
    l_portfolio = [
        "MMM", "AXP", "AMGN", "AAPL", "BA", "CAT", "CVX", "CSCO", "DOW", "GS",
        "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MRK", "MSFT",
        "NKE", "PG", "TRV", "UNH", "VZ", "V", "WBA", "WMT", "DIS"
    ]

    mc = MarketConsensus(l_portfolio)
    mc.build_portfolio_consensus()
    mc.df_consensus
    mc.create_plotly_table("Portfolio Market Consensus")
    mc.fig_marketConsensus

# recommandation mean interpretation
# 1 = Strong Buy
# 2 = Buy
# 3 = Hold
# 4 = Sell
# 5 = Strong Sell
