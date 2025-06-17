import streamlit as st
import yfinance as yf
import backtrader as bt
import pandas as pd
import plotly.graph_objects as go
import numpy as np
# --- Strategies ---

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

class SmaCrossStrategy(bt.Strategy):
    params = dict(short_period=10, long_period=30)
    def __init__(self):
        self.sma_short = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.short_period)
        self.sma_long = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.long_period)
        self.trade_log = []
        self.buy_dates = []
        self.buy_prices = []
        self.sell_dates = []
        self.sell_prices = []
    def next(self):
        if not self.position:
            if self.sma_short[0] > self.sma_long[0] and self.sma_short[-1] <= self.sma_long[-1]:
                self.buy()
                self.trade_log.append(f"BUY at {self.data.close[0]:.2f} on {self.data.datetime.date(0)}")
                self.buy_dates.append(self.data.datetime.date(0))
                self.buy_prices.append(self.data.close[0])
        else:
            if self.sma_short[0] < self.sma_long[0] and self.sma_short[-1] >= self.sma_long[-1]:
                self.sell()
                self.trade_log.append(f"SELL at {self.data.close[0]:.2f} on {self.data.datetime.date(0)}")
                self.sell_dates.append(self.data.datetime.date(0))
                self.sell_prices.append(self.data.close[0])

class MacdStrategy(bt.Strategy):
    def __init__(self):
        macd = bt.indicators.MACD(self.data)
        self.macd_line = macd.macd
        self.signal_line = macd.signal
        self.trade_log = []
        self.buy_dates = []
        self.buy_prices = []
        self.sell_dates = []
        self.sell_prices = []
    def next(self):
        if not self.position:
            if self.macd_line[0] > self.signal_line[0] and self.macd_line[-1] <= self.signal_line[-1]:
                self.buy()
                self.trade_log.append(f"BUY at {self.data.close[0]:.2f} on {self.data.datetime.date(0)}")
                self.buy_dates.append(self.data.datetime.date(0))
                self.buy_prices.append(self.data.close[0])
        else:
            if self.macd_line[0] < self.signal_line[0] and self.macd_line[-1] >= self.signal_line[-1]:
                self.sell()
                self.trade_log.append(f"SELL at {self.data.close[0]:.2f} on {self.data.datetime.date(0)}")
                self.sell_dates.append(self.data.datetime.date(0))
                self.sell_prices.append(self.data.close[0])

class BollingerBandsStrategy(bt.Strategy):
    def __init__(self):
        self.bbands = bt.indicators.BollingerBands(self.data.close, period=20)
        self.trade_log = []
        self.buy_dates = []
        self.buy_prices = []
        self.sell_dates = []
        self.sell_prices = []
    def next(self):
        if not self.position:
            if self.data.close[0] < self.bbands.lines.bot[0]:
                self.buy()
                self.trade_log.append(f"BUY at {self.data.close[0]:.2f} on {self.data.datetime.date(0)}")
                self.buy_dates.append(self.data.datetime.date(0))
                self.buy_prices.append(self.data.close[0])
        else:
            if self.data.close[0] > self.bbands.lines.top[0]:
                self.sell()
                self.trade_log.append(f"SELL at {self.data.close[0]:.2f} on {self.data.datetime.date(0)}")
                self.sell_dates.append(self.data.datetime.date(0))
                self.sell_prices.append(self.data.close[0])

class MomentumStrategy(bt.Strategy):
    params = dict(momentum_period=10)
    def __init__(self):
        self.momentum = bt.indicators.Momentum(self.data.close, period=self.p.momentum_period)
        self.trade_log = []
        self.buy_dates = []
        self.buy_prices = []
        self.sell_dates = []
        self.sell_prices = []
    def next(self):
        if not self.position:
            if self.momentum[0] > 0:
                self.buy()
                self.trade_log.append(f"BUY at {self.data.close[0]:.2f} on {self.data.datetime.date(0)}")
                self.buy_dates.append(self.data.datetime.date(0))
                self.buy_prices.append(self.data.close[0])
        else:
            if self.momentum[0] < 0:
                self.sell()
                self.trade_log.append(f"SELL at {self.data.close[0]:.2f} on {self.data.datetime.date(0)}")
                self.sell_dates.append(self.data.datetime.date(0))
                self.sell_prices.append(self.data.close[0])

# --- Helper Functions ---

def get_strategy_by_name(name):
    mapping = {
        "SMA + RSI": SmaRsiStrategy,
        "SMA Crossover": SmaCrossStrategy,
        "MACD": MacdStrategy,
        "Bollinger Bands": BollingerBandsStrategy,
        "Momentum": MomentumStrategy,
    }
    return mapping.get(name)

def run_backtest(strategy_class, data_df, **kwargs):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(10000)
    data_feed = bt.feeds.PandasData(dataname=data_df)
    cerebro.adddata(data_feed)
    cerebro.addstrategy(strategy_class, **kwargs)
    results = cerebro.run()
    strat = results[0]
    final_portfolio = cerebro.broker.getvalue()
    total_return = (final_portfolio - 10000) / 10000 * 100
    return strat, final_portfolio, total_return

def plot_results(data_df, strat, strategy_name, sma_period=None):
    

    fig = go.Figure()

    # Always plot close price
    fig.add_trace(go.Scatter(
        x=data_df.index,
        y=data_df['close'],
        mode='lines',
        name='Close Price',
        line=dict(color='cyan', width=2)
    ))

    # Plot SMA if relevant (for SMA+RSI and SMA Crossover)
    if strategy_name in ['SMA + RSI', 'SMA Crossover'] and sma_period is not None:
        if strategy_name == 'SMA + RSI':
            sma_vals = list(strat.sma.array)
        else:  # SMA Crossover has two SMAs
            sma_short = list(strat.sma_short.array)
            sma_long = list(strat.sma_long.array)
            fig.add_trace(go.Scatter(
                x=data_df.index,
                y=sma_short,
                mode='lines',
                name='SMA Short',
                line=dict(color='orange', width=2, dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=data_df.index,
                y=sma_long,
                mode='lines',
                name='SMA Long',
                line=dict(color='yellow', width=2, dash='dot')
            ))
        if strategy_name == 'SMA + RSI':
            fig.add_trace(go.Scatter(
                x=data_df.index,
                y=sma_vals,
                mode='lines',
                name=f'SMA ({sma_period})',
                line=dict(color='orange', width=2, dash='dash')
            ))

    # Plot Bollinger Bands if relevant
    if strategy_name == 'Bollinger Bands':
        # Convert arrays to numpy arrays and replace nan with None
        bband_top = np.array(strat.bbands.lines.top.array)
        bband_mid = np.array(strat.bbands.lines.mid.array)
        bband_bot = np.array(strat.bbands.lines.bot.array)
    
        bband_top = [None if np.isnan(x) else x for x in bband_top]
        bband_mid = [None if np.isnan(x) else x for x in bband_mid]
        bband_bot = [None if np.isnan(x) else x for x in bband_bot]
    
        fig.add_trace(go.Scatter(
            x=data_df.index,
            y=bband_top,
            mode='lines',
            name='Bollinger Band Top',
            line=dict(color='lightblue', width=1, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=data_df.index,
            y=bband_mid,
            mode='lines',
            name='Bollinger Band Mid',
            line=dict(color='lightgray', width=1, dash='dot')
        ))
        fig.add_trace(go.Scatter(
            x=data_df.index,
            y=bband_bot,
            mode='lines',
            name='Bollinger Band Bottom',
            line=dict(color='lightblue', width=1, dash='dash')
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
        title=f"{strategy_name} on {symbol}",
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        height=600,
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        legend=dict(x=0.01, y=0.99),
    )

    return fig

# --- Streamlit UI ---

st.set_page_config(page_title="Backtrader Strategy Backtester", layout="wide")

symbol = st.sidebar.text_input("Ticker", "AAPL")
start = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))

strategy_name = st.sidebar.selectbox("Select Strategy", [
    "SMA + RSI",
    "SMA Crossover",
    "MACD",
    "Bollinger Bands",
    "Momentum"
])

# Extra params for selected strategy
rsi_low = 30
rsi_high = 70
sma_period = 20

if strategy_name == "SMA + RSI":
    rsi_low = st.sidebar.slider("RSI Buy Threshold", 10, 90, 70)
    rsi_high = st.sidebar.slider("RSI Sell Threshold", 50, 90, 70)
    sma_period = st.sidebar.slider("SMA Window", 5, 50, 20)
elif strategy_name == "SMA Crossover":
    short_period = st.sidebar.slider("Short SMA Period", 5, 50, 10)
    long_period = st.sidebar.slider("Long SMA Period", 10, 200, 30)
elif strategy_name == "Momentum":
    momentum_period = st.sidebar.slider("Momentum Period", 5, 50, 10)

# Load data
data_df = yf.download(symbol, start=start, end=end)
if isinstance(data_df.columns, pd.MultiIndex):
    data_df.columns = [col[0] for col in data_df.columns]
data_df.columns = [col.lower() for col in data_df.columns]

if data_df.empty:
    st.error("No data found for your selection.")
else:
    # Prepare strategy kwargs
    strat_kwargs = {}
    if strategy_name == "SMA + RSI":
        strat_kwargs = {"rsi_low": rsi_low, "rsi_high": rsi_high, "sma_period": sma_period}
    elif strategy_name == "SMA Crossover":
        strat_kwargs = {"short_period": short_period, "long_period": long_period}
    elif strategy_name == "Momentum":
        strat_kwargs = {"momentum_period": momentum_period}

    strategy_class = get_strategy_by_name(strategy_name)

    strat, final_portfolio, total_return = run_backtest(strategy_class, data_df, **strat_kwargs)

    st.metric("Final Portfolio Value", f"${final_portfolio:,.2f}")
    st.metric("Total Return", f"{total_return:.2f}%")

    fig = plot_results(data_df, strat, strategy_name, sma_period=sma_period if strategy_name=="SMA + RSI" else None)

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("📄 Trade Log"):
        for entry in strat.trade_log:
            st.write(entry)
