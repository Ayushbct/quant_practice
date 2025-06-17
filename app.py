import streamlit as st
import yfinance as yf
import backtrader as bt
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Base Strategy Classes
class RsiSmaStrategy(bt.Strategy):
    params = (
        ("rsi_low", 30),
        ("rsi_high", 70),
        ("sma_period", 20),
    )

    def __init__(self):
        self.rsi = bt.indicators.RSI(self.data.close, period=14)
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.sma_period)
        self.buy_dates, self.buy_prices = [], []
        self.sell_dates, self.sell_prices = [], []
        self.trade_log = []

    def next(self):
        if not self.position:
            if self.rsi < self.p.rsi_low and self.data.close[0] > self.sma[0]:
                self.buy()
                self.buy_dates.append(self.data.datetime.date(0))
                self.buy_prices.append(self.data.close[0])
                self.trade_log.append(f"BUY at {self.data.close[0]:.2f} on {self.data.datetime.date(0)}")
        else:
            if self.rsi > self.p.rsi_high:
                self.sell()
                self.sell_dates.append(self.data.datetime.date(0))
                self.sell_prices.append(self.data.close[0])
                self.trade_log.append(f"SELL at {self.data.close[0]:.2f} on {self.data.datetime.date(0)}")


class SmaCrossStrategy(bt.Strategy):
    params = (
    ("fast", 10),
    ("slow", 50),
)


    def __init__(self):
        sma_fast = bt.ind.SMA(period=self.p.fast)
        sma_slow = bt.ind.SMA(period=self.p.slow)
        self.crossover = bt.ind.CrossOver(sma_fast, sma_slow)
        self.sma_fast = sma_fast
        self.sma_slow = sma_slow
        self.buy_dates, self.buy_prices = [], []
        self.sell_dates, self.sell_prices = [], []
        self.trade_log = []

    def next(self):
        if not self.position and self.crossover > 0:
            self.buy()
            self.buy_dates.append(self.data.datetime.date(0))
            self.buy_prices.append(self.data.close[0])
            self.trade_log.append(f"BUY at {self.data.close[0]:.2f} on {self.data.datetime.date(0)}")
        elif self.position and self.crossover < 0:
            self.sell()
            self.sell_dates.append(self.data.datetime.date(0))
            self.sell_prices.append(self.data.close[0])
            self.trade_log.append(f"SELL at {self.data.close[0]:.2f} on {self.data.datetime.date(0)}")


class BollingerBandStrategy(bt.Strategy):
    params = (
    ("period", 20),
    ("devfactor", 2.0),
    )


    def __init__(self):
        self.bb = bt.ind.BollingerBands(period=self.p.period, devfactor=self.p.devfactor)
        self.buy_dates, self.buy_prices = [], []
        self.sell_dates, self.sell_prices = [], []
        self.trade_log = []

    def next(self):
        if not self.position and self.data.close[0] < self.bb.lines.bot[0]:
            self.buy()
            self.buy_dates.append(self.data.datetime.date(0))
            self.buy_prices.append(self.data.close[0])
            self.trade_log.append(f"BUY at {self.data.close[0]:.2f} on {self.data.datetime.date(0)}")
        elif self.position and self.data.close[0] > self.bb.lines.top[0]:
            self.sell()
            self.sell_dates.append(self.data.datetime.date(0))
            self.sell_prices.append(self.data.close[0])
            self.trade_log.append(f"SELL at {self.data.close[0]:.2f} on {self.data.datetime.date(0)}")


class MacdStrategy(bt.Strategy):
    def __init__(self):
        self.macd = bt.ind.MACD()
        self.crossover = bt.ind.CrossOver(self.macd.macd, self.macd.signal)
        self.buy_dates, self.buy_prices = [], []
        self.sell_dates, self.sell_prices = [], []
        self.trade_log = []

    def next(self):
        if not self.position and self.crossover > 0:
            self.buy()
            self.buy_dates.append(self.data.datetime.date(0))
            self.buy_prices.append(self.data.close[0])
            self.trade_log.append(f"BUY at {self.data.close[0]:.2f} on {self.data.datetime.date(0)}")
        elif self.position and self.crossover < 0:
            self.sell()
            self.sell_dates.append(self.data.datetime.date(0))
            self.sell_prices.append(self.data.close[0])
            self.trade_log.append(f"SELL at {self.data.close[0]:.2f} on {self.data.datetime.date(0)}")


# Strategy dictionary
STRATEGY_MAP = {
    "RSI + SMA": RsiSmaStrategy,
    "SMA Crossover": SmaCrossStrategy,
    "Bollinger Bands": BollingerBandStrategy,
    "MACD Crossover": MacdStrategy,
}

# Streamlit UI
st.set_page_config(page_title="Backtrader Strategy Backtester", layout="wide")

symbol = st.sidebar.text_input("Ticker", "AAPL")
start = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))
strategy_name = st.sidebar.selectbox("Select Strategy", list(STRATEGY_MAP.keys()))

params = {}
if strategy_name == "RSI + SMA":
    params = {
        "rsi_low": st.sidebar.slider("RSI Buy Threshold", 10, 90, 50),
        "rsi_high": st.sidebar.slider("RSI Sell Threshold", 10, 90, 50),
        "sma_period": st.sidebar.slider("SMA Period", 5, 50, 20)
    }
elif strategy_name == "SMA Crossover":
    params = {
        "fast": st.sidebar.slider("Fast SMA", 5, 50, 10),
        "slow": st.sidebar.slider("Slow SMA", 20, 200, 50)
    }
elif strategy_name == "Bollinger Bands":
    params = {
        "period": st.sidebar.slider("BB Period", 5, 50, 20),
        "devfactor": st.sidebar.slider("Deviation Factor", 1.0, 3.0, 2.0)
    }

# Load Data
data_df = yf.download(symbol, start=start, end=end)

if isinstance(data_df.columns, pd.MultiIndex):
    data_df.columns = [col[0] for col in data_df.columns]
data_df.columns = [col.lower() for col in data_df.columns]

if data_df.empty:
    st.error("No data found for your selection.")
else:
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(10000)
    data_feed = bt.feeds.PandasData(dataname=data_df)
    cerebro.adddata(data_feed)
    cerebro.addstrategy(STRATEGY_MAP[strategy_name], **params)

    results = cerebro.run()
    strat = results[0]

    final_portfolio = cerebro.broker.getvalue()
    total_return = (final_portfolio - 10000) / 10000 * 100

    st.metric("Final Portfolio Value", f"${final_portfolio:,.2f}")
    st.metric("Total Return", f"{total_return:.2f}%")

    # Plot with Plotly
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data_df.index, y=data_df['close'], mode='lines', name='Close Price', line=dict(color='cyan', width=2)))

    if hasattr(strat, 'sma'):
        fig.add_trace(go.Scatter(x=data_df.index, y=list(strat.sma.array), mode='lines', name='SMA', line=dict(color='orange', width=2, dash='dash')))
    if hasattr(strat, 'sma_fast') and hasattr(strat, 'sma_slow'):
        fig.add_trace(go.Scatter(x=data_df.index, y=list(strat.sma_fast.array), mode='lines', name='Fast SMA', line=dict(color='lime', width=2)))
        fig.add_trace(go.Scatter(x=data_df.index, y=list(strat.sma_slow.array), mode='lines', name='Slow SMA', line=dict(color='magenta', width=2)))
    if hasattr(strat, 'bb'):
        fig.add_trace(go.Scatter(x=data_df.index, y=list(strat.bb.lines.top), mode='lines', name='BB Upper', line=dict(color='yellow', dash='dot')))
        fig.add_trace(go.Scatter(x=data_df.index, y=list(strat.bb.lines.bot), mode='lines', name='BB Lower', line=dict(color='yellow', dash='dot')))

    # Buy/sell markers
    fig.add_trace(go.Scatter(x=strat.buy_dates, y=strat.buy_prices, mode='markers', marker=dict(symbol='triangle-up', color='green', size=12), name='Buy'))
    fig.add_trace(go.Scatter(x=strat.sell_dates, y=strat.sell_prices, mode='markers', marker=dict(symbol='triangle-down', color='red', size=12), name='Sell'))

    fig.update_layout(
        title=f"{symbol} Price with Strategy Signals",
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,
        hovermode="x unified",
        legend=dict(x=0.01, y=0.99)
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📄 Trade Log"):
        for entry in strat.trade_log:
            st.write(entry)
