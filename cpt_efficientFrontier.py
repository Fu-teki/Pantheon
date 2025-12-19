### MARKOWITZ - EFFICIENT FRONTIER

# %% IMPORT

# data
import numpy as np
import pandas as pd

# date
import datetime as dt
from dateutil.relativedelta import relativedelta

# finance
import yfinance as yf

# optimization
import scipy.optimize as sc
import plotly.graph_objects as go

# %% EF - CLASS

class EfficientFrontier:

    def __init__(self, stocks, start_date=None, end_date=None, risk_free_rate=0):

        # Inputs
        self.stocks = stocks
        self.end_date = end_date or dt.datetime.now()
        self.start_date = start_date or (self.end_date - relativedelta(years=3))
        self.risk_free_rate = risk_free_rate
        
        # data
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None

        # results
        self.fig_corrMatrix = None
        self.fig_eff = None
        self.fig_perfTable = None
        self.fig_maxSR = None
        self.fig_minVol = None

    ## DATA LOADING

    def get_data(self):
        stock_data = yf.download(self.stocks, start=self.start_date, end=self.end_date, auto_adjust=False)['Adj Close']
        self.returns = stock_data.pct_change().dropna()
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        return self.returns, self.mean_returns, self.cov_matrix

    # PORTFOLIO PERF.

    def portfolio_performance(self, weights):
        returns = np.sum(self.mean_returns * weights) * 252
        std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))) * np.sqrt(252)
        return returns, std

    def negative_sharpe_ratio(self, weights):
        p_returns, p_std = self.portfolio_performance(weights)
        return - (p_returns - self.risk_free_rate) / p_std

    def max_sharpe_ratio(self, bounds=(0, 1)):
        num_assets = len(self.mean_returns)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple(bounds for _ in range(num_assets))
        result = sc.minimize(self.negative_sharpe_ratio,
                             num_assets * [1. / num_assets],
                             method='SLSQP',
                             bounds=bounds,
                             constraints=constraints)
        return result

    def portfolio_variance(self, weights):
        return self.portfolio_performance(weights)[1]

    def min_variance(self, bounds=(0, 1)):
        num_assets = len(self.mean_returns)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple(bounds for _ in range(num_assets))
        result = sc.minimize(self.portfolio_variance,
                             num_assets * [1. / num_assets],
                             method='SLSQP',
                             bounds=bounds,
                             constraints=constraints)
        return result

    def portfolio_return(self, weights):
        return self.portfolio_performance(weights)[0]

    def efficient_opt(self, return_target, bounds=(0, 1)):
        num_assets = len(self.mean_returns)
        constraints = ({'type': 'eq', 'fun': lambda x: self.portfolio_return(x) - return_target},
                       {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple(bounds for _ in range(num_assets))
        result = sc.minimize(self.portfolio_variance,
                             num_assets * [1. / num_assets],
                             args=(),
                             method='SLSQP',
                             bounds=bounds,
                             constraints=constraints)
        return result

    # COMPUTE RESULTS

    def calculated_results(self):
        max_sr = self.max_sharpe_ratio()
        min_vol = self.min_variance()

        max_sr_ret, max_sr_std = self.portfolio_performance(max_sr['x'])
        min_vol_ret, min_vol_std = self.portfolio_performance(min_vol['x'])

        max_sr_alloc = pd.DataFrame(max_sr['x'], index=self.mean_returns.index, columns=['weightings'])
        min_vol_alloc = pd.DataFrame(min_vol['x'], index=self.mean_returns.index, columns=['weightings'])

        target_returns = np.linspace(min_vol_ret, max_sr_ret, 20)
        efficient_list = [self.efficient_opt(target)['fun'] for target in target_returns]

        return (round(max_sr_ret * 100, 2), round(max_sr_std * 100, 2), max_sr_alloc,
                round(min_vol_ret * 100, 2), round(min_vol_std * 100, 2), min_vol_alloc,
                efficient_list, target_returns)

    # PLOTTING

    def plot_correlation_matrix(self):

        corr_matrix = self.returns.corr()

        self.fig_corrMatrix = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation")
        ))
        self.fig_corrMatrix.update_layout(title="Correlation Matrix", template="plotly_dark", yaxis_autorange='reversed')

    def plot_efficient_frontier(self):

        max_sr_ret, max_sr_std, _, min_vol_ret, min_vol_std, _, efficient_list, target_returns = self.calculated_results()

        fig = go.Figure()

        fig.add_trace(go.Scatter(x=[max_sr_std], y=[max_sr_ret], mode='markers',
                                 marker=dict(color='red', size=14, line=dict(width=3, color='black')),
                                 name='Max Sharpe Ratio'))
        fig.add_trace(go.Scatter(x=[min_vol_std], y=[min_vol_ret], mode='markers',
                                 marker=dict(color='green', size=14, line=dict(width=3, color='black')),
                                 name='Min Volatility'))
        fig.add_trace(go.Scatter(x=[round(std * 100, 2) for std in efficient_list],
                                 y=[round(ret * 100, 2) for ret in target_returns],
                                 mode='lines', name='Efficient Frontier',
                                 line=dict(color='cyan', width=4, dash='dashdot')))
        fig.update_layout(title="Efficient Frontier",
                          xaxis=dict(title='Annualised Volatility (%)'),
                          yaxis=dict(title='Annualised Return (%)'),
                          template="plotly_dark")

        self.fig_eff = fig

    def plot_table_results(self):

        (max_sr_returns, max_sr_std, max_sr_alloc,
         min_vol_returns, min_vol_std, min_vol_alloc,
         _, _) = self.calculated_results()

        df_ef = pd.DataFrame({
            'Max Sharpe Return (%)': [round(max_sr_returns, 2)],
            'Max Sharpe Std (%)': [round(max_sr_std, 2)],
            'Min Vol Return (%)': [round(min_vol_returns, 2)],
            'Min Vol Std (%)': [round(min_vol_std, 2)]
        })

        self.fig_perfTable = go.Figure(data=[go.Table(
            header=dict(values=list(df_ef.columns), fill_color='dodgerblue',
                        font=dict(color='white', size=14), align='center'),
            cells=dict(values=[df_ef[col] for col in df_ef.columns], fill_color='black',
                       font=dict(color='white'), align='center')
        )])

        self.fig_perfTable.update_layout(title="Efficient Frontier - Results", template="plotly_dark")

    def plot_alloc_pie(self):

        (_, _, max_sr_alloc,
         _, _, min_vol_alloc,
         _, _) = self.calculated_results()

        # Max SR

        max_sr_alloc = max_sr_alloc.T
        max_sr_alloc = max_sr_alloc.iloc[0][max_sr_alloc.iloc[0] > 0]
        self.fig_maxSR = go.Figure(data=[go.Pie(labels=max_sr_alloc.index, values=max_sr_alloc, hole=0.3,
                                         textinfo="label+percent")])
        self.fig_maxSR.update_layout(title="Max Sharpe Ratio - Portfolio", template="plotly_dark")


        # Min Vol

        min_vol_alloc = min_vol_alloc.T
        min_vol_alloc = min_vol_alloc.iloc[0][min_vol_alloc.iloc[0] > 0]
        self.fig_minVol = go.Figure(data=[go.Pie(labels=min_vol_alloc.index, values=min_vol_alloc, hole=0.3,
                                         textinfo="label+percent")])
        self.fig_minVol.update_layout(title="Min Volatility - Portfolio", template="plotly_dark")


# %% EXECUTE

if __name__ == "__main__":

    # l_stocks = ["AAPL", "MSFT", "NVDA"]
    l_stocks=["IWDA.AS", "IWQU.L", "^DJI", "^IXIC", 
            "ZPRR.DE", "^STOXX", "EUMD.L", "AEEM.MI", 
            "ICGA.DE", "CMDY", "BTC-USD" ,"ETH-USD"]

    ef = EfficientFrontier(l_stocks)
    ef.get_data()
    ef.plot_correlation_matrix()
    ef.plot_efficient_frontier()
    ef.plot_table_results()
    ef.plot_alloc_pie()

    # ef.fig_maxSR.show()
    # ef.fig_minVol.show()

# %%
