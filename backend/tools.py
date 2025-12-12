# tools.py
import pandas as pd
import yfinance as yf
from .price_scraper import downloadStockPrice
import os

def get_current_price(symbol: str):
    """Returns current asset price using yfinance."""
    data = yf.Ticker(symbol)
    price = data.info.get("regularMarketPrice")
    return price

def get_historical_data(symbol: str):
    """Runs your MarketWatch scraper and returns the last month."""
    downloadStockPrice(symbol)
    csv_file = f"{symbol}_data.csv"

    if not os.path.exists(csv_file):
        return None

    df = pd.read_csv(csv_file)
    # sort by most recent
    df = df.sort_values(by="Date", ascending=False)
    return df.head(30).to_dict(orient="records")
