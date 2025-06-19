# --- Imports ---
import streamlit as st
import yfinance as yf
import backtrader as bt
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from data_loader import load_data
# --- Strategy Definitions ---
def get_all_strategies():

    class RelativeStrengthStrategy(bt.Strategy):
        params = dict(benchmark_symbol="SPY", rs_ma_period=10)

        def __init__(self):
            # Benchmark price will be passed as "data1"
            self.relative_strength = self.data.close / self.datas[1].close
            self.rs_ma = bt.indicators.SimpleMovingAverage(self.relative_strength, period=self.p.rs_ma_period)
            self.init_trade_tracking()

        def init_trade_tracking(self):
            self.trade_log, self.buy_dates, self.buy_prices = [], [], []
            self.sell_dates, self.sell_prices = [], []

        def next(self):
            if not self.position and self.relative_strength[0] > self.rs_ma[0] and self.relative_strength[-1] <= self.rs_ma[-1]:
                self.buy(); self.log_trade("BUY")
            elif self.position and self.relative_strength[0] < self.rs_ma[0] and self.relative_strength[-1] >= self.rs_ma[-1]:
                self.sell(); self.log_trade("SELL")

        def log_trade(self, action):
            price = self.data.close[0]
            date = self.data.datetime.date(0)
            self.trade_log.append(f"{action} at {price:.2f} on {date}")
            if action == "BUY":
                self.buy_dates.append(date); self.buy_prices.append(price)
            else:
                self.sell_dates.append(date); self.sell_prices.append(price)


    class SmaRsiStrategy(bt.Strategy):
        params = ("rsi_low", 70), ("rsi_high", 60), ("sma_period", 20)
        def __init__(self):
            self.rsi = bt.indicators.RSI(self.data.close, period=14)
            self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.p.sma_period)
            self.init_trade_tracking()
        def init_trade_tracking(self):
            self.trade_log, self.buy_dates, self.buy_prices = [], [], []
            self.sell_dates, self.sell_prices = [], []
        def next(self):
            if not self.position and self.rsi < self.p.rsi_low and self.data.close[0] > self.sma[0]:
                self.buy(); self.log_trade("BUY")
            elif self.position and self.rsi > self.p.rsi_high:
                self.sell(); self.log_trade("SELL")
        def log_trade(self, action):
            price = self.data.close[0]
            date = self.data.datetime.date(0)
            self.trade_log.append(f"{action} at {price:.2f} on {date}")
            if action == "BUY":
                self.buy_dates.append(date); self.buy_prices.append(price)
            else:
                self.sell_dates.append(date); self.sell_prices.append(price)

    class SmaCrossStrategy(bt.Strategy):
        params = dict(short_period=10, long_period=30)
        def __init__(self):
            self.sma_short = bt.indicators.SMA(self.data.close, period=self.p.short_period)
            self.sma_long = bt.indicators.SMA(self.data.close, period=self.p.long_period)
            self.init_trade_tracking()
        def init_trade_tracking(self):
            self.trade_log, self.buy_dates, self.buy_prices = [], [], []
            self.sell_dates, self.sell_prices = [], []
        def next(self):
            if not self.position and self.sma_short[0] > self.sma_long[0] and self.sma_short[-1] <= self.sma_long[-1]:
                self.buy(); self.log_trade("BUY")
            elif self.position and self.sma_short[0] < self.sma_long[0] and self.sma_short[-1] >= self.sma_long[-1]:
                self.sell(); self.log_trade("SELL")
        def log_trade(self, action):
            price = self.data.close[0]
            date = self.data.datetime.date(0)
            self.trade_log.append(f"{action} at {price:.2f} on {date}")
            if action == "BUY":
                self.buy_dates.append(date); self.buy_prices.append(price)
            else:
                self.sell_dates.append(date); self.sell_prices.append(price)

    class MacdStrategy(bt.Strategy):
        def __init__(self):
            macd = bt.indicators.MACD(self.data)
            self.macd_line = macd.macd
            self.signal_line = macd.signal
            self.init_trade_tracking()
        def init_trade_tracking(self):
            self.trade_log, self.buy_dates, self.buy_prices = [], [], []
            self.sell_dates, self.sell_prices = [], []
        def next(self):
            if not self.position and self.macd_line[0] > self.signal_line[0] and self.macd_line[-1] <= self.signal_line[-1]:
                self.buy(); self.log_trade("BUY")
            elif self.position and self.macd_line[0] < self.signal_line[0] and self.macd_line[-1] >= self.signal_line[-1]:
                self.sell(); self.log_trade("SELL")
        def log_trade(self, action):
            price = self.data.close[0]
            date = self.data.datetime.date(0)
            self.trade_log.append(f"{action} at {price:.2f} on {date}")
            if action == "BUY":
                self.buy_dates.append(date); self.buy_prices.append(price)
            else:
                self.sell_dates.append(date); self.sell_prices.append(price)

    class BollingerBandsStrategy(bt.Strategy):
        def __init__(self):
            self.bbands = bt.indicators.BollingerBands(self.data.close, period=20)
            self.init_trade_tracking()
        def init_trade_tracking(self):
            self.trade_log, self.buy_dates, self.buy_prices = [], [], []
            self.sell_dates, self.sell_prices = [], []
        def next(self):
            if not self.position and self.data.close[0] < self.bbands.lines.bot[0]:
                self.buy(); self.log_trade("BUY")
            elif self.position and self.data.close[0] > self.bbands.lines.top[0]:
                self.sell(); self.log_trade("SELL")
        def log_trade(self, action):
            price = self.data.close[0]
            date = self.data.datetime.date(0)
            self.trade_log.append(f"{action} at {price:.2f} on {date}")
            if action == "BUY":
                self.buy_dates.append(date); self.buy_prices.append(price)
            else:
                self.sell_dates.append(date); self.sell_prices.append(price)

    class MomentumStrategy(bt.Strategy):
        params = dict(momentum_period=10)
        def __init__(self):
            self.momentum = bt.indicators.Momentum(self.data.close, period=self.p.momentum_period)
            self.init_trade_tracking()
        def init_trade_tracking(self):
            self.trade_log, self.buy_dates, self.buy_prices = [], [], []
            self.sell_dates, self.sell_prices = [], []
        def next(self):
            if not self.position and self.momentum[0] > 0:
                self.buy(); self.log_trade("BUY")
            elif self.position and self.momentum[0] < 0:
                self.sell(); self.log_trade("SELL")
        def log_trade(self, action):
            price = self.data.close[0]
            date = self.data.datetime.date(0)
            self.trade_log.append(f"{action} at {price:.2f} on {date}")
            if action == "BUY":
                self.buy_dates.append(date); self.buy_prices.append(price)
            else:
                self.sell_dates.append(date); self.sell_prices.append(price)

    class StochRsiStrategy(bt.Strategy):
        params = dict(period=14, lower=0.2, upper=0.8, smooth_k=3)
        def __init__(self):
            rsi = bt.indicators.RSI(self.data.close, period=self.p.period)
            rsi_min = bt.indicators.Lowest(rsi, period=self.p.period)
            rsi_max = bt.indicators.Highest(rsi, period=self.p.period)
            raw_stochrsi = (rsi - rsi_min) / (rsi_max - rsi_min)
            self.stochrsi = bt.indicators.MovingAverageSimple(raw_stochrsi, period=self.p.smooth_k)
            self.init_trade_tracking()
        def init_trade_tracking(self):
            self.trade_log, self.buy_dates, self.buy_prices = [], [], []
            self.sell_dates, self.sell_prices = [], []
        def next(self):
            value = self.stochrsi[0]
            if not self.position and value < self.p.lower:
                self.buy(); self.log_trade("BUY")
            elif self.position and value > self.p.upper:
                self.sell(); self.log_trade("SELL")
        def log_trade(self, action):
            price = self.data.close[0]
            date = self.data.datetime.date(0)
            self.trade_log.append(f"{action} at {price:.2f} on {date}")
            if action == "BUY":
                self.buy_dates.append(date); self.buy_prices.append(price)
            else:
                self.sell_dates.append(date); self.sell_prices.append(price)

    return {
        "SMA + RSI": SmaRsiStrategy,
        "SMA Crossover": SmaCrossStrategy,
        "MACD": MacdStrategy,
        "Bollinger Bands": BollingerBandsStrategy,
        "Momentum": MomentumStrategy,
        "StochRSI": StochRsiStrategy,
        "Relative Strength": RelativeStrengthStrategy,

    }

# --- Backtesting Function ---
def run_backtest(strategy_class, data_df, benchmark_df=None, **kwargs):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(10000)

    data_feed = bt.feeds.PandasData(dataname=data_df)
    cerebro.adddata(data_feed)

    if benchmark_df is not None:
        benchmark_feed = bt.feeds.PandasData(dataname=benchmark_df)
        cerebro.adddata(benchmark_feed)

    cerebro.addstrategy(strategy_class, **kwargs)
    results = cerebro.run()
    strat = results[0]
    final_value = cerebro.broker.getvalue()
    return strat, final_value, (final_value - 10000) / 10000 * 100


# --- Plotting ---
def plot_results(data_df, strat, strategy_name, sma_period=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_df.index, y=data_df['close'], mode='lines', name='Close', line=dict(color='cyan')))
    
    if strategy_name == "SMA + RSI":
        fig.add_trace(go.Scatter(x=data_df.index, y=list(strat.sma.array), name=f"SMA({sma_period})", line=dict(color='orange')))
    elif strategy_name == "SMA Crossover":
        fig.add_trace(go.Scatter(x=data_df.index, y=list(strat.sma_short.array), name="Short SMA", line=dict(color='orange')))
        fig.add_trace(go.Scatter(x=data_df.index, y=list(strat.sma_long.array), name="Long SMA", line=dict(color='yellow')))
    elif strategy_name == "Bollinger Bands":
        bband_top = np.array(strat.bbands.lines.top.array)
        bband_mid = np.array(strat.bbands.lines.mid.array)
        bband_bot = np.array(strat.bbands.lines.bot.array)
        fig.add_trace(go.Scatter(x=data_df.index, y=[None if np.isnan(x) else x for x in bband_top], mode='lines', name='Bollinger Band Top', line=dict(color='lightblue', width=1, dash='dash')))
        fig.add_trace(go.Scatter(x=data_df.index, y=[None if np.isnan(x) else x for x in bband_mid], mode='lines', name='Bollinger Band Mid', line=dict(color='lightgray', width=1, dash='dot')))
        fig.add_trace(go.Scatter(x=data_df.index, y=[None if np.isnan(x) else x for x in bband_bot], mode='lines', name='Bollinger Band Bottom', line=dict(color='lightblue', width=1, dash='dash')))

    fig.add_trace(go.Scatter(x=strat.buy_dates, y=strat.buy_prices, mode='markers', name='Buy', marker=dict(color='green', symbol='triangle-up', size=12)))
    fig.add_trace(go.Scatter(x=strat.sell_dates, y=strat.sell_prices, mode='markers', name='Sell', marker=dict(color='red', symbol='triangle-down', size=12)))

    fig.update_layout(title=f"{strategy_name}", template="plotly_dark", height=600)
    return fig

# --- Streamlit UI ---
st.set_page_config(page_title="Strategy Backtester", layout="wide")

symbol = st.sidebar.text_input("Symbol", "AAPL")
start = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("2025-01-01"))

strategy_name = st.sidebar.selectbox("Select Strategy", list(get_all_strategies().keys()))
params = {}
if strategy_name == "SMA + RSI":
    params = {
        "rsi_low": st.sidebar.slider("RSI Buy Threshold", 10, 90, 70),
        "rsi_high": st.sidebar.slider("RSI Sell Threshold", 10, 90, 50),
        "sma_period": st.sidebar.slider("SMA Period", 5, 50, 20)
    }
elif strategy_name == "SMA Crossover":
    params = {
        "short_period": st.sidebar.slider("Short SMA", 5, 50, 10),
        "long_period": st.sidebar.slider("Long SMA", 10, 200, 30)
    }
elif strategy_name == "Momentum":
    params = {"momentum_period": st.sidebar.slider("Momentum Period", 5, 50, 10)}
elif strategy_name == "StochRSI":
    params = {
        "period": st.sidebar.slider("RSI Period", 5, 50, 14),
        "smooth_k": st.sidebar.slider("Smooth K", 1, 10, 3),
        "lower": st.sidebar.slider("Buy Threshold", 0.0, 1.0, 0.2),
        "upper": st.sidebar.slider("Sell Threshold", 0.0, 1.0, 0.8)
    }
elif strategy_name == "Relative Strength":
    params = {
        "benchmark_symbol": st.sidebar.text_input("Benchmark Symbol", "SPY"),
        "rs_ma_period": st.sidebar.slider("RS MA Period", 2, 50, 10)
    }



# from data_loader import load_data

# Add a selector for source
source = st.sidebar.selectbox("Data Source", ["yahoo", "nepse"])

data_df = load_data(symbol, start=start, end=end, source=source)

benchmark_df = None
if strategy_name == "Relative Strength":
    benchmark_df = load_data(params["benchmark_symbol"], start=start, end=end, source=source)


# data_df = yf.download(symbol, start=start, end=end)
# if isinstance(data_df.columns, pd.MultiIndex):
#     data_df.columns = [col[0] for col in data_df.columns]
# data_df.columns = [col.lower() for col in data_df.columns]

# benchmark_df = None
# if strategy_name == "Relative Strength":
#     benchmark_df = yf.download(params["benchmark_symbol"], start=start, end=end)
#     if isinstance(benchmark_df.columns, pd.MultiIndex):
#         benchmark_df.columns = [col[0] for col in benchmark_df.columns]
#     benchmark_df.columns = [col.lower() for col in benchmark_df.columns]



if data_df.empty:
    st.error("No data found.")
else:
    strategy_class = get_all_strategies()[strategy_name]
    strat, final_val, total_return = run_backtest(strategy_class, data_df, benchmark_df, **params)

    st.metric("Final Portfolio Value", f"${final_val:,.2f}")
    st.metric("Total Return", f"{total_return:.2f}%")
    st.plotly_chart(plot_results(data_df, strat, strategy_name, params.get("sma_period")), use_container_width=True)
    with st.expander("📘 Trade Log"):
        for log in strat.trade_log:
            st.write(log)
