### MONITOR MY PERSONAL PORTFOLIO

# %% IMPORT LIBS

# data
import pandas as pd, numpy as np
import datetime as dt
from dateutil.relativedelta import relativedelta

# financial
import yfinance as yf

# plot
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# %% CLASS

class MyPortfolio():

    # Instance variable
    def __init__(self, path_excel_orders, excel_name_orders, benchIndex="^GSPC"):

        # inputs
        self.path_excel_orders = path_excel_orders
        self.excel_name_orders = excel_name_orders
        self.benchIndex = benchIndex

        # object variables
        self.df_book = pd.DataFrame(columns=["Date", "Ticker", "Name", "Price", "Quantity"])
        self.df_book.index.name = "Trade"

        # results
        self.df_portfolioInfo = pd.DataFrame()
        self.df_portfolioReturns = pd.DataFrame()
        self.df_benchReturns = pd.DataFrame()

    def importTradingBook(self):

        self.df_book = pd.read_excel(self.path_excel_orders + self.excel_name_orders, skiprows=3)
        self.df_book = self.df_book.iloc[:,[1,2,3,4,5,6,7]]
        self.df_book.set_index("Trade #", inplace=True)

    def get_orders(self, stock=None):
        if stock:
            return self.df_book[self.df_book["Ticker"] == stock]

        else:
            return self.df_book
    
    def get_currentPosition(self):

        if self.df_book.empty:
            raise ValueError("You must at least have 1 order set up -> using get_orders().")

        # Group by TICKER and NAME to sum QUANTITY and TOTAL POSITION
        df_portfolioInfo = self.df_book.groupby(['Ticker', 'Name']).agg({
            'Quantity': 'sum',
            'Total Position': 'sum'
        }).reset_index()

        # Fetch latest prices using Yahoo Finance
        current_prices = []

        for ticker in df_portfolioInfo['Ticker']:
            try:
                stock = yf.Ticker(ticker)
                history = stock.history(period='1d')

                # Get last available close price
                if not history.empty:
                    latest_price = history['Close'].iloc[-1]
                else:
                    latest_price = None

            except Exception:
                latest_price = None

            current_prices.append(latest_price)

        # Add to DataFrame
        df_portfolioInfo['Current Price'] = current_prices
        df_portfolioInfo['Current Value'] = df_portfolioInfo['Quantity'] * df_portfolioInfo['Current Price']

        # Calculate per-asset return
        df_portfolioInfo['RETURN (%)'] = (
            (df_portfolioInfo['Current Value'] - df_portfolioInfo['Total Position'])
            / df_portfolioInfo['Total Position']
        ) * 100

        # Portfolio totals
        total_initial = df_portfolioInfo['Total Position'].sum()
        total_current = df_portfolioInfo['Current Value'].sum()
        total_return_pct = ((total_current - total_initial) / total_initial) * 100

        # Round numeric columns
        df_portfolioInfo[['Current Price', 'Current Value', 'RETURN (%)']] = (
            df_portfolioInfo[['Current Price', 'Current Value', 'RETURN (%)']].round(2)
        )

        totalPositionValue = df_portfolioInfo["Current Value"].sum()

        # Weights
        if totalPositionValue > 0:
            df_portfolioInfo["Current Weights"] = (
                df_portfolioInfo["Current Value"] / totalPositionValue
            ).round(2)
        else:
            df_portfolioInfo["Current Weights"] = 0
            print("⚠️ Warning: all current prices are missing — weights set to 0.")

        # Display portfolio summary
        print(f"📊 Total Initial Value: ${total_initial:.2f}")
        print(f"📈 Total Current Value: ${total_current:.2f}")
        print(f"💹 Portfolio Return: {total_return_pct:.2f}%")

        # Save results
        self.df_portfolioInfo = df_portfolioInfo

    def get_portfolio_returns(self, base_value=100):

        if self.df_book.empty:
            raise ValueError("You must at least have 1 order set up -> using get_orders().")

        self.df_book['Date'] = pd.to_datetime(self.df_book['Date'], dayfirst=True)
        
        # Get only the trades on the first date (initial portfolio)
        first_trade_date = self.df_book['Date'].min()
        initial_trades = self.df_book[self.df_book['Date'] == first_trade_date].copy()

        tickers = initial_trades['Ticker'].unique().tolist()
        end_date = pd.Timestamp.today().normalize()

        # Download historical price data
        price_data = yf.download(tickers, start=first_trade_date, end=end_date + pd.Timedelta(days=1), auto_adjust=False)['Adj Close']
        price_data = price_data.fillna(method='ffill')

        # Ensure it's a DataFrame
        if isinstance(price_data, pd.Series):
            price_data = price_data.to_frame()

        # Create weight based on initial capital allocation
        initial_trades['value'] = initial_trades['Quantity'] * initial_trades['Price']
        total_initial_value = initial_trades['value'].sum()
        initial_trades['weight'] = initial_trades['value'] / total_initial_value

        # Calculate daily returns of each ticker
        daily_returns = price_data.pct_change().fillna(0)

        # Multiply each return by its weight to get weighted daily portfolio return
        weighted_returns = daily_returns.multiply(initial_trades.set_index('Ticker')['weight'], axis=1)
        portfolio_returns = weighted_returns.sum(axis=1)

        # Calculate normalized portfolio value starting from base_value
        normalized_value = (1 + portfolio_returns).cumprod() * base_value

        # Return as DataFrame
        self.df_portfolioReturns = pd.DataFrame({
            'Daily Return': portfolio_returns,
            'My Portfolio Normalized': normalized_value
        })

    def get_benchmark_returns(self):

        if self.df_book.empty:
            raise ValueError("You must at least have 1 order set up -> using get_orders().")

        startDate = self.df_book["Date"].iloc[0]
        endDate = dt.datetime.today()
        
        dow_data = yf.download(self.benchIndex, start=startDate, end=endDate, auto_adjust=False)['Adj Close']
        dow_data = dow_data.dropna()
        dow_normalized = dow_data / dow_data.iloc[0] * 100
        
        self.df_benchReturns = pd.DataFrame(dow_normalized).round(2)
        self.df_benchReturns.rename(columns={self.benchIndex:f"{self.benchIndex} Normalized"}, inplace=True)

    def plot_portfolio_vs_bench(self):

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=self.df_portfolioReturns.index,
            y=self.df_portfolioReturns['My Portfolio Normalized'],
            mode='lines',
            name='My Portfolio',
            line=dict(color='yellow', width=2)
        ))

        fig.add_trace(go.Scatter(
            x=self.df_benchReturns.index,
            y=self.df_benchReturns[f'{self.benchIndex} Normalized'],
            mode='lines',
            name=self.benchIndex,
            line=dict(color='white', width=2, dash='dot')
        ))

        fig.update_layout(
            title=f"Your Portfolio vs {self.benchIndex}",
            xaxis_title="Date",
            yaxis_title="Normalized Performance (Start = 100)",
            template="plotly_dark",
            legend=dict(x=0.05, y=0.95),
            height=500
        )

        fig.show()

# %% EXECUTION

if __name__ == "__main__":

    # INPUTS
    path_excel_orders = r"C:/Users/amint/Desktop/PANTHEON RESEARCH/USER/"
    excel_name_orders = "TRADE_BOOK.xlsx"
    benchIndex="^GSPC"

    # Initialize
    mp = MyPortfolio(path_excel_orders, excel_name_orders, benchIndex)

    # show trading book
    mp.importTradingBook()

    # get portfolio info
    mp.get_currentPosition()
    mp.df_portfolioInfo

    # get returns
    mp.get_portfolio_returns(base_value=100)
    mp.get_benchmark_returns()
    mp.df_portfolioReturns
    mp.df_benchReturns

    # plot
    mp.plot_portfolio_vs_bench()

# %%
