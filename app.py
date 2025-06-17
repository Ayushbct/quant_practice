import streamlit as st
import yfinance as yf
import backtrader as bt
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

class SmaRsiStrategy(bt.Strategy):
    params = (
        ("rsi_low", 30),
        ("rsi_high", 70),
        ("sma_period", 20),
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.sma_period)
        self.trade_log = []
        self.buy_dates = []
        self.buy_prices = []
        self.sell_dates = []
        self.sell_prices = []

    def next(self):
        if not self.position:
            if self.rsi < self.p.rsi_low and self.data.close[0] > self.sma[0]:
                self.buy()
                self.trade_log.append(f"BUY at {self.data.close[0]:.2f} on {self.data.datetime.date(0)}")
                self.buy_dates.append(self.data.datetime.date(0))
                self.buy_prices.append(self.data.close[0])
        else:
            if self.rsi > self.p.rsi_high:
                self.sell()
                self.trade_log.append(f"SELL at {self.data.close[0]:.2f} on {self.data.datetime.date(0)}")
                self.sell_dates.append(self.data.datetime.date(0))
                self.sell_prices.append(self.data.close[0])


# Streamlit UI
st.set_page_config(page_title="Backtrader Strategy Backtester", layout="wide")

# Sidebar Inputs
symbol = st.sidebar.text_input("Ticker", "AAPL")
start = st.sidebar.date_input("Start Date", pd.to_datetime("2022-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))
rsi_low = st.sidebar.slider("RSI Buy Threshold", 10, 90)
rsi_high = st.sidebar.slider("RSI Sell Threshold", 50, 90, 70)
sma_period = st.sidebar.slider("SMA Window", 5, 200)

# Load Data
data_df = yf.download(symbol, start=start, end=end)

if isinstance(data_df.columns, pd.MultiIndex):
    data_df.columns = [col[0] for col in data_df.columns]

data_df.columns = [col.lower() for col in data_df.columns]

if data_df.empty:
    st.error("No data found for your selection.")
else:
    # Run Backtest
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(10000)
    data_feed = bt.feeds.PandasData(dataname=data_df)
    cerebro.adddata(data_feed)
    cerebro.addstrategy(SmaRsiStrategy, rsi_low=rsi_low, rsi_high=rsi_high, sma_period=sma_period)

    results = cerebro.run()
    strat = results[0]

    final_portfolio = cerebro.broker.getvalue()
    total_return = (final_portfolio - 10000) / 10000 * 100

    st.metric("Final Portfolio Value", f"${final_portfolio:,.2f}")
    st.metric("Total Return", f"{total_return:.2f}%")

    # Plot with Plotly
    fig = go.Figure()

    # Plot price and SMA
    fig.add_trace(go.Scatter(
    x=data_df.index,
    y=data_df['close'],
    mode='lines',
    name='Close Price',
    line=dict(color='cyan', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=data_df.index,
        y=list(strat.sma.array),
        mode='lines',
        name=f'SMA ({sma_period})',
        line=dict(color='orange', width=2, dash='dash')
    ))



    # Plot buy/sell markers
    fig.add_trace(go.Scatter(
        x=strat.buy_dates,
        y=strat.buy_prices,
        mode='markers',
        marker=dict(symbol='triangle-up', color='green', size=12),
        name='Buy Signal'
    ))

    fig.add_trace(go.Scatter(
        x=strat.sell_dates,
        y=strat.sell_prices,
        mode='markers',
        marker=dict(symbol='triangle-down', color='red', size=12),
        name='Sell Signal'
    ))

    fig.update_layout(
    title=f"{symbol} Price with SMA and Trade Signals",
    xaxis_title="Date",
    yaxis_title="Price",

    # template="plotly_white",  # ✅ this makes it light-themed
    # plot_bgcolor='white',         # ✅ plot area background
    # paper_bgcolor='white',        # ✅ entire figure background
    # font=dict(color='black'),
    hovermode="x unified",
    height=600,
    legend=dict(x=0.01, y=0.99),
)


    st.plotly_chart(fig, use_container_width=True)

    # Optional: Show trade log
    with st.expander("📄 Trade Log"):
        for entry in strat.trade_log:
            st.write(entry)
