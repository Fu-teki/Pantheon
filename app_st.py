
# -*- coding: utf-8 -*-

# %% MAIN STREAMLIT DASHBOARD

# Basics
import streamlit as st
from datetime import date
import pandas as pd, numpy as np
import yfinance as yf
import datetime as dt
from dateutil.relativedelta import relativedelta

# Import custom modules
from cpt_ptf_metrics import PortfolioMetrics
from input_etfs_dataset import fetch_etf_prices
from cpt_efficientFrontier import EfficientFrontier
from cpt_myPtf import MyPortfolio
from cpt_quantScoring import QuantModel
from cpt_stock_ta import TechnicalAnalysis

# 🚀 PAGE CONFIG (must be first Streamlit command)
st.set_page_config(page_title="Equity Research Platform", layout="wide", page_icon="💼")

# === PAGE HEADER ===
st.title("🏛️ EQUITY RESEARCH")
st.markdown("An integrated analytics workspace for **My Portfolio**, **ETFs**, **US Equities**, and **Machine Learning**.")

# === SIDEBAR NAVIGATION ===
st.sidebar.title("🏛️ PANTHEON")
page = st.sidebar.radio(
    "Choose analysis module:",
    ["My Portfolio", "General Portfolio Analysis", "Equity Analysis", "Stock Technical Analysis", "Machine Learning"]
)

# =========================================================
# 0️⃣ MY PORTFOLIO
# =========================================================
if page == "My Portfolio":
    st.header("📘 My Personal Portfolio Monitor")

    st.sidebar.subheader("Portfolio Settings")

    # DEFAULT VALUES (your inputs pre-filled)
    path_excel = st.sidebar.text_input(
        "Excel Folder Path:",
        "C:/Users/amint/Desktop/PANTHEON RESEARCH/USER/"
    )

    excel_name = st.sidebar.text_input(
        "Excel File Name:",
        "TRADE_BOOK.xlsx"
    )

    benchIndex = st.sidebar.text_input(
        "Benchmark Index:",
        "^GSPC"
    )

    run_portfolio = st.sidebar.button("Run Portfolio Monitor")

    if run_portfolio:

        st.success("🔄 Loading your portfolio...")

        # Initialize class (API REMOVED)
        mp = MyPortfolio(
            path_excel_orders=path_excel,
            excel_name_orders=excel_name,
            benchIndex=benchIndex
        )

        # === STEP 1: TRADING BOOK ===
        try:
            mp.importTradingBook()
        except Exception as e:
            st.error(f"❌ Error loading Excel file: {e}")
            st.stop()

        # === STEP 2: CURRENT POSITIONS ===
        try:
            mp.get_currentPosition()
        except Exception as e:
            st.error(f"❌ Error calculating positions: {e}")
            st.stop()

        # === STEP 3: PORTFOLIO RETURNS ===
        try:
            mp.get_portfolio_returns()
        except Exception as e:
            st.error(f"❌ Error calculating portfolio returns: {e}")
            st.stop()

        # === STEP 4: BENCHMARK RETURNS ===
        try:
            mp.get_benchmark_returns()
        except Exception as e:
            st.error(f"❌ Benchmark error: {e}")
            st.stop()

        # === FINAL PLOT ===
        st.subheader("⚖️ Portfolio vs Benchmark Comparison")

        import plotly.graph_objects as go

        fig = go.Figure()

        # Portfolio curve
        fig.add_trace(go.Scatter(
            x=mp.df_portfolioReturns.index,
            y=mp.df_portfolioReturns['My Portfolio Normalized'],
            mode='lines',
            name='My Portfolio',
            line=dict(color='yellow', width=3)
        ))

        # Benchmark curve
        bench_col = mp.df_benchReturns.columns[0]
        fig.add_trace(go.Scatter(
            x=mp.df_benchReturns.index,
            y=mp.df_benchReturns[bench_col],
            mode='lines',
            name=bench_col,
            line=dict(color='white', width=2, dash='dot')
        ))

        fig.update_layout(
            title="My Portfolio vs Benchmark",
            xaxis_title="Date",
            yaxis_title="Normalized Performance (100 = Starting Value)",
            template="plotly_dark",
            legend=dict(x=0.02, y=0.98),
            height=550
        )

        st.plotly_chart(fig, use_container_width=True)

        # === ADD THE PORTFOLIO BREAKDOWN TABLE ===
        st.subheader("📊 Current Portfolio Breakdown")
        st.dataframe(mp.df_portfolioInfo, use_container_width=True)

        # === RISK ANALYSIS SECTION ===
        st.subheader("📉 Risk Analysis (Portfolio vs Benchmark)")

        from cpt_risk import RiskAnalysis

        # Prepare return series
        ptf_returns = mp.df_portfolioReturns["Daily Return"].dropna()

        # Benchmark returns: compute % change on normalized benchmark prices
        bench_series = mp.df_benchReturns.iloc[:, 0]
        bench_returns = bench_series.pct_change().dropna()

        # Initialize risk object
        risk = RiskAnalysis()

        # --- STD DEV ---
        risk.compute_stdDev(ptf_returns)
        df_std_ptf = risk.df_stdDev.copy()

        risk.compute_stdDev(bench_returns)
        df_std_bench = risk.df_stdDev.copy()

        risk.plot_std_dev_ptfBench(df_std_ptf, df_std_bench)
        st.plotly_chart(risk.fig_stdDev_ptfBench, use_container_width=True)

        # --- EXPECTED LOSS (GARCH) ---
        risk.compute_el_garch(ptf_returns)
        df_el_ptf = risk.df_expectedLoss.copy()

        risk.compute_el_garch(bench_returns)
        df_el_bench = risk.df_expectedLoss.copy()

        risk.plot_el_garch_ptfVsBench(df_el_ptf, df_el_bench)
        st.plotly_chart(risk.fig_el_ptfBench, use_container_width=True)

        # --- MONTE CARLO ---
        risk.compute_mc_st(ptf_returns, "Portfolio")
        mc_ptf = risk.dict_mc_results.copy()

        risk.compute_mc_st(bench_returns, "Benchmark")
        mc_bench = risk.dict_mc_results.copy()

        risk.plot_mc_st(mc_ptf, mc_bench)
        st.plotly_chart(risk.fig_mc_st, use_container_width=True)

    else:
        st.info("👈 Click **Run Portfolio Monitor** to load your tracker.")

# =========================================================
# 1️⃣ GENERAL PORTFOLIO ANALYSIS PAGE
# =========================================================
elif page == "General Portfolio Analysis":
    st.header("📊 General Portfolio Analysis")

    # --- SIDEBAR INPUTS ---
    st.sidebar.subheader("Ticker Settings")

    PANTHEON_TICKERS = ["IWDA.AS", "IWQU.L", "^DJI", "^IXIC",
                    "ZPRR.DE", "^STOXX", "EUMD.L", "AEEM.MI",
                    "ICGA.DE", "CMDY", "BTC-USD", "ETH-USD"]

    ticker_mode = st.sidebar.radio(
        "Ticker selection:",
        options=["Pantheon's tickers", "Your own tickers"],
        index=0
    )

    if ticker_mode == "Pantheon's tickers":
        ticker_list = st.sidebar.multiselect(
            "Select Tickers:",
            options=PANTHEON_TICKERS,
            default=PANTHEON_TICKERS
        )
    else:
        tickers_text = st.sidebar.text_input(
            "Enter tickers (comma-separated):",
            value="AAPL, MSFT, NVDA"
        )
        ticker_list = [t.strip() for t in tickers_text.split(",") if t.strip()]

        if not ticker_list:
            st.sidebar.warning("Please enter at least one ticker (comma-separated).")

    # === DEFAULT DATES (Last 5 Years) ===
    end_default = dt.datetime.today().date()
    start_default = (dt.datetime.today() - relativedelta(years=5)).date()

    start_date = st.sidebar.date_input("Start Date", start_default)
    end_date = st.sidebar.date_input("End Date", end_default)

    portfolio_mode = st.sidebar.radio(
        "Analysis Mode:",
        options=["YES (Portfolio)", "NO (Individual Assets)"]
    )

    risk_free_rate = st.sidebar.number_input("Risk-Free Rate (annual %)", value=0.0) / 100

    benchmark_ticker = st.sidebar.text_input(
        "Benchmark Ticker (optional):",
        value="SPY"
    )

    weights = None
    if portfolio_mode.startswith("YES"):
        st.sidebar.subheader("Portfolio Weights")
        weights = []
        for ticker in ticker_list:
            w = st.sidebar.number_input(f"Weight for {ticker}", value=1.0 / len(ticker_list), step=0.05)
            weights.append(w)
        total = sum(weights)
        if total != 1:
            weights = [w / total for w in weights]
            st.sidebar.info("Weights normalized to sum = 1")

    # --- RUN ANALYSIS ---
    if st.sidebar.button("Run General Portfolio Analysis"):

        # -----------------------------
        # 1. FETCH PRICE DATA
        # -----------------------------
        with st.spinner("📥 Fetching market data..."):
            df_prices = fetch_etf_prices(
                ticker_list,
                str(start_date),
                str(end_date)
            )

        st.success("Market data loaded ✔")

        # -----------------------------
        # 2. RUN UPGRADED METRICS CLASS
        # -----------------------------
        analyzer = PortfolioMetrics(
            df_prices=df_prices,
            weights=weights,
            risk_free_rate=risk_free_rate,
            portfolio="YES" if portfolio_mode.startswith("YES") else "NO",
            benchmark_ticker=benchmark_ticker if benchmark_ticker.strip() != "" else None,
            start_date=str(start_date),
            end_date=str(end_date),
        )

        df_metrics = analyzer.compute_metrics()

        # -----------------------------
        # 3. DISPLAY RESULTS
        # -----------------------------
        st.subheader("📈 Performance Metrics (Portfolio vs Benchmark)")
        st.dataframe(df_metrics, use_container_width=True)

        # -----------------------------
        # 4. PLOTS
        # -----------------------------
        st.subheader("📈 Normalized Price (Base 100)")
        st.plotly_chart(analyzer.plot_price_chart(), use_container_width=True)

        # -----------------------------
        # 5. EFFICIENT FRONTIER
        # -----------------------------
        st.subheader("🚀 Efficient Frontier")
        ef = EfficientFrontier(ticker_list)
        ef.get_data()
        ef.plot_correlation_matrix()
        ef.plot_efficient_frontier()
        ef.plot_table_results()
        ef.plot_alloc_pie()

        if ef.fig_corrMatrix:
            st.plotly_chart(ef.fig_corrMatrix, use_container_width=True)
        if ef.fig_eff:
            st.plotly_chart(ef.fig_eff, use_container_width=True)
        if ef.fig_perfTable:
            st.plotly_chart(ef.fig_perfTable, use_container_width=True)
        if ef.fig_maxSR:
            st.plotly_chart(ef.fig_maxSR, use_container_width=True)
        if ef.fig_minVol:
            st.plotly_chart(ef.fig_minVol, use_container_width=True)

        # -----------------------------
        # 6. RISK ANALYSIS (using portfolio & benchmark returns automatically)
        # -----------------------------
        st.subheader("📉 Risk Analysis (Portfolio vs Benchmark)")

        from cpt_risk import RiskAnalysis

        # Use PortfolioMetrics' aligned return series
        ptf_returns = analyzer.df_returns.dot(np.array(weights)) if portfolio_mode.startswith("YES") else analyzer.df_returns.mean(axis=1)

        if analyzer.bench_returns is not None:
            bench_returns = analyzer.bench_returns
        else:
            st.warning("⚠ No benchmark selected — defaulting to SPY.")
            df_bench = yf.download("SPY", start=start_date, end=end_date, auto_adjust=False)["Adj Close"]
            bench_returns = df_bench.pct_change().dropna()

        risk = RiskAnalysis()

        # --- STD DEV ---
        risk.compute_stdDev(ptf_returns)
        df_std_ptf = risk.df_stdDev.copy()

        risk.compute_stdDev(bench_returns)
        df_std_bench = risk.df_stdDev.copy()

        risk.plot_std_dev_ptfBench(df_std_ptf, df_std_bench)
        st.plotly_chart(risk.fig_stdDev_ptfBench, use_container_width=True)

        # --- EXPECTED LOSS (GARCH) ---
        risk.compute_el_garch(ptf_returns)
        df_el_ptf = risk.df_expectedLoss.copy()

        risk.compute_el_garch(bench_returns)
        df_el_bench = risk.df_expectedLoss.copy()

        risk.plot_el_garch_ptfVsBench(df_el_ptf, df_el_bench)
        st.plotly_chart(risk.fig_el_ptfBench, use_container_width=True)

        # --- MONTE CARLO ---
        risk.compute_mc_st(ptf_returns, "Portfolio")
        mc_ptf = risk.dict_mc_results.copy()

        risk.compute_mc_st(bench_returns, "Benchmark")
        mc_bench = risk.dict_mc_results.copy()

        risk.plot_mc_st(mc_ptf, mc_bench)
        st.plotly_chart(risk.fig_mc_st, use_container_width=True)

    else:
        st.info("👈 Configure your analysis then click **Run General Portfolio Analysis**.")

# =========================================================
# 2️⃣ EQUITY ANALYSIS PAGE
# =========================================================
elif page == "Equity Analysis":
    st.header("Equity Analysis")

    # --- Static list of DJIA components for demonstration ---
    DOW_JONES_TICKERS = [
        "MMM", "AXP", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "DIS", 
        "DOW", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "MCD", 
        "MRK", "MSFT", "NKE", "PG", "CRM", "TRV", "UNH", "VZ", "V", 
        "WBA", "WMT", "AMGN"
    ]
    # --------------------------------------------------------

    # ============================
    # Sidebar – Stock Input
    # ============================
    st.sidebar.subheader("Stock Selection")
    
    # 1. Radio button to select input type
    selection_mode = st.sidebar.radio(
        "Choose stocks to analyze:",
        ("Custom Tickers", "Entire Dow Jones Index (DJIA)"),
        key="selection_mode"
    )

    l_stocks = [] # Initialize list of stocks
    
    if selection_mode == "Custom Tickers":
        ticker_input = st.sidebar.text_input(
            "Enter tickers separated by commas:",
            "AAPL,MSFT,CAT"
        )

        l_stocks = [x.strip().upper() for x in ticker_input.split(",") if x.strip()]

        # Nice tag-style display
        if l_stocks:
            st.sidebar.markdown(
                "### 🏷️ Selected Tickers:"
            )
            tags = " ".join([f"<span style='background:#262730;padding:4px 10px;border-radius:8px;margin:2px;display:inline-block;'>{t}</span>"
                            for t in l_stocks])
            st.sidebar.markdown(tags, unsafe_allow_html=True)
        
    elif selection_mode == "Entire Dow Jones Index (DJIA)":
        # 3. Use the static DJIA list
        l_stocks = DOW_JONES_TICKERS
        st.sidebar.info(f"Analyzing {len(l_stocks)} DJIA stocks.")

    # Show the final list of stocks being analyzed
    if l_stocks:
         st.sidebar.caption(f"**Current Tickers:** {', '.join(l_stocks[:5])}... ({len(l_stocks)} total)")


    # ============================
    # Sidebar – Dates
    # ============================
    # === DEFAULT DATES (Last 5 Years) ===
    end_default = dt.datetime.today().date()
    start_default = (dt.datetime.today() - relativedelta(years=5)).date()

    start_date = st.sidebar.date_input("Start Date", start_default)
    end_date = st.sidebar.date_input("End Date", end_default)

    # ============================
    # Button: Run Equity Analysis
    # ============================
    if st.sidebar.button("Run Equity Analysis"):
        
        if not l_stocks:
            st.warning("Please enter or select at least one ticker before running the analysis.")
        else:
            # ----------------------------
            # 🔎 RUN QUANT SCORING ONLY HERE
            # ----------------------------
            st.subheader("📊 Quant Factor Scoring")

            API_KEY = "eStobmCXEMQ3XXAPdIhN7oycsELjuOta"

            try:
                # Assuming QuantModel is defined elsewhere and handles the list of stocks
                # The l_stocks list is now dynamically set based on user's choice
                qm = QuantModel(
                    api_key=API_KEY,
                    tickers=l_stocks, 
                    start_price="2018-01-01",
                    benchmark_ticker="SPY",
                )

                df_scoring = qm.run()

                st.success("Quant scoring completed successfully ✔")

                st.subheader("🏆 Total Quant Scores (0–100)")
                st.dataframe(
                    df_scoring.sort_values("TotalScore", ascending=False),
                    use_container_width=True
                )

            except Exception as e:
                st.error(f"❌ Error running QuantModel: {e}")


            # ----------------------------
            # PRICE DATA
            # ----------------------------
            # Assuming yfinance is imported (from previous context/notebook)
            with st.spinner("📥 Downloading price data..."):
                # Use the filtered/selected l_stocks list for yf.download
                df = yf.download(l_stocks, start=start_date, end=end_date, auto_adjust=False)['Adj Close']

            if df.empty:
                st.error("No price data found for the selected dates/tickers.")
            else:
                st.success("Price data loaded.")

                st.subheader("📈 Normalized Price (Base 100)")
                st.line_chart(df / df.iloc[0] * 100)

                # Efficient Frontier... (Assuming EfficientFrontier is defined elsewhere)
                ef = EfficientFrontier(l_stocks)
                ef.get_data()
                ef.plot_correlation_matrix()
                ef.plot_efficient_frontier()
                ef.plot_table_results()
                ef.plot_alloc_pie()

                # Display Plotly charts
                if ef.fig_corrMatrix:
                    st.plotly_chart(ef.fig_corrMatrix, use_container_width=True)
                if ef.fig_eff:
                    st.plotly_chart(ef.fig_eff, use_container_width=True)
                if ef.fig_perfTable:
                    st.plotly_chart(ef.fig_perfTable, use_container_width=True)
                if ef.fig_maxSR:
                    st.plotly_chart(ef.fig_maxSR, use_container_width=True)
                if ef.fig_minVol:
                    st.plotly_chart(ef.fig_minVol, use_container_width=True)

    else:
        st.info("👈 Select your stocks (Custom or DJIA) then click **Run Equity Analysis**.")# =========================================================

# =========================================================
# 📈 3️⃣ STOCK TECHNICAL ANALYSIS PAGE
# =========================================================
elif page == "Stock Technical Analysis":
    st.header("📈 Stock Technical Analysis")

    st.sidebar.subheader("Technical Analysis Settings")

    # --- USER INPUTS ---
    ticker = st.sidebar.text_input(
        "Stock Ticker:",
        value="MSFT"
    ).upper()

    # === DEFAULT DATES (Last 5 Years) ===
    end_default = dt.datetime.today().date()
    start_default = (dt.datetime.today() - relativedelta(years=5)).date()

    # === USER DATE INPUTS (editable) ===
    start_date = st.sidebar.date_input("Start Date:", value=start_default)
    end_date = st.sidebar.date_input("End Date:", value=end_default)

    # === MODE SELECTION (short / medium / long) ===
    mode = st.sidebar.selectbox(
        "Technical Analysis Mode:",
        ["short", "medium", "long"],
        index=1  # medium as default
    )

    # Run the analysis
    run_ta = st.sidebar.button("Run Technical Analysis")

    # --- RUN TA ---
    if run_ta:

        st.subheader(f"📊 Technical Indicators for **{ticker}** ({mode} term)")

        try:
            ta = TechnicalAnalysis(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                mode=mode
            )

            # --- CHART ---
            fig = ta.run()
            st.plotly_chart(fig, use_container_width=True)

            # --- BACKTEST (P&L SUMMARY ONLY) ---
            st.subheader("💰 Backtest P&L Summary")

            try:
                df_backtest = ta.compute_backtests()

                # Display only the P&L dataframe (NO TRADES)
                st.dataframe(df_backtest, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Error during backtest: {e}")

        except Exception as e:
            st.error(f"❌ Error computing technical analysis: {e}")

    else:
        st.info("👈 Configure the settings and click **Run Technical Analysis**.")

# 3️⃣ MACHINE LEARNING PAGE
# =========================================================
elif page == "Machine Learning":
    st.header("🤖 Machine Learning Lab")
    st.info("🚧 Work in progress.")

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("Powered by Pantheon")

# %%
