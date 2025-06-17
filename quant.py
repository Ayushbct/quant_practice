import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Parameters
ticker = "AAPL"
start_date = "2018-01-01"
end_date = "2025-01-01"
momentum_window = 20

# Download historical data

data = yf.download(ticker, start=start_date, end=end_date)
# Calculate daily returns
data['Return'] = data['Close'].pct_change()
data['Momentum'] = data['Close'].pct_change(momentum_window)

# Generate trading signals: buy if momentum > 0, else sell
data['Signal'] = np.where(data['Momentum'] > 0, 1, -1)

# Calculate strategy returns
data['Strategy_Return'] = data['Signal'].shift(1) * data['Return']

# Calculate cumulative returns
data['Cumulative_Market'] = (1 + data['Return']).cumprod()
data['Cumulative_Strategy'] = (1 + data['Strategy_Return']).cumprod()

# Performance metrics
total_return = data['Cumulative_Strategy'].iloc[-1] - 1
annualized_return = (data['Cumulative_Strategy'].iloc[-1])**(252/len(data)) - 1

annualized_vol = data['Strategy_Return'].std() * np.sqrt(252)
sharpe_ratio = annualized_return / annualized_vol

print(f"Total Strategy Return: {total_return:.2%}")
print(f"Annualized Return: {annualized_return:.2%}")
print(f"Annualized Volatility: {annualized_vol:.2%}")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Plot cumulative returns
plt.figure(figsize=(12,6))
plt.plot(data['Cumulative_Market'], label='Market Return (Buy & Hold)')
plt.plot(data['Cumulative_Strategy'], label='Momentum Strategy')
plt.legend()
plt.title(f'Momentum Strategy vs Market on {ticker}')
plt.savefig("momentum_strategy_plot.png", dpi=300)
print("Plot saved as momentum_strategy_plot.png")

