import yfinance as yf
import pandas as pd

def download_yahoo_data(symbol: str, start, end):
    """Download data from Yahoo Finance."""
    data_df = yf.download(symbol, start=start, end=end)
    if isinstance(data_df.columns, pd.MultiIndex):
        data_df.columns = [col[0] for col in data_df.columns]
    data_df.columns = [col.lower() for col in data_df.columns]
    return data_df

def download_nepse_data(symbol: str, start, end):
    """
    Placeholder for NEPSE data loading. Replace this with actual logic.
    This could load from a local CSV, API, or database.
    """
    # Example: load from CSV
    path = f"data/nepse/{symbol}.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.set_index("date")
    df = df.loc[start:end]
    df.columns = [col.lower() for col in df.columns]
    return df

def load_data(symbol: str, start, end, source="yahoo"):
    """Dispatcher for multiple data sources."""
    if source == "nepse":
        return download_nepse_data(symbol, start, end)
    else:
        return download_yahoo_data(symbol, start, end)
