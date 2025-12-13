# tools.py
import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta, timezone

def get_current_price(symbol: str):
    """Returns current asset price using yfinance with robust fallbacks."""
    try:
        t = yf.Ticker(symbol)
        # Prefer fast_info if available
        price = getattr(t, "fast_info", {}).get("last_price")
        if price is None:
            info = getattr(t, "info", {})
            price = info.get("regularMarketPrice")
        # As a last resort, fetch recent history and use the last close
        if price is None:
            hist = t.history(period="5d")
            if isinstance(hist, pd.DataFrame) and not hist.empty:
                price = float(hist["Close"].iloc[-1])
        return None if price is None else float(price)
    except Exception:
        return None

def get_historical_data(symbol: str):
    """Return recent historical OHLCV using yfinance (last 30 calendar days).

    Provides a clean list of dicts with ISO dates and close prices. Avoids
    MarketWatch scraping issues (stale or repeated data).
    """
    try:
        # Pull ~30 days including today; yfinance will handle market days
        t = yf.Ticker(symbol)
        df = t.history(period="1mo", interval="1d")

        if df is None or df.empty:
            # Fallback: explicit date range last 45 days to increase chance of data
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=45)
            df = yf.download(symbol, start=start.date().isoformat(), end=end.date().isoformat(), interval="1d")

        if df is None or df.empty:
            return []

        # Ensure we have Date column in ISO format
        df = df.reset_index()
        # yfinance may return DatetimeIndex under column name 'Date' or 'Datetime'
        date_col = "Date" if "Date" in df.columns else ("Datetime" if "Datetime" in df.columns else df.columns[0])
        df["Date"] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
        df = df.dropna(subset=["Date"])  # drop invalid rows
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

        # Select relevant fields and sort recent first
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        out = df[["Date", *cols]].sort_values(by="Date", ascending=False).head(30)

        return out.to_dict(orient="records")
    except Exception:
        return []
