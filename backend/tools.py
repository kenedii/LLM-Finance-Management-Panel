# tools.py
import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta, timezone

# ----------------------------------------------
# Asset resolution: map common names to Yahoo symbols
# ----------------------------------------------
def _resolve_asset(symbol: str) -> dict:
    """
    Normalize user-provided asset identifiers to appropriate Yahoo Finance symbols,
    and classify the asset type. Also return display name and currency when available.

    Supported examples:
    - Crypto: BTC -> BTC-USD, ETH -> ETH-USD
    - Gold: GOLD/XAU -> XAUUSD=X (Gold Spot USD per troy ounce)
    - Silver: SILVER/XAG -> XAGUSD=X (Silver Spot USD per troy ounce)
    - Otherwise: use provided symbol (e.g., AAPL, TSLA)
    """
    s = (symbol or "").strip()
    su = s.upper()

    asset_type = "equity"
    mapped = s

    # Crypto mapping
    if su in {"BTC", "BTCUSD", "BTC-USD"}:
        mapped = "BTC-USD"
        asset_type = "crypto"
    elif su in {"ETH", "ETHUSD", "ETH-USD"}:
        mapped = "ETH-USD"
        asset_type = "crypto"

    # Precious metals (spot per ounce)
    elif su in {"GOLD", "XAU", "XAUUSD", "XAUUSD=X"}:
        mapped = "XAUUSD=X"  # Gold Spot USD per troy ounce
        asset_type = "metal"
    elif su in {"SILVER", "XAG", "XAGUSD", "XAGUSD=X"}:
        mapped = "XAGUSD=X"  # Silver Spot USD per troy ounce
        asset_type = "metal"

    # Common futures alternative (if user explicitly passes GC=F, SI=F, keep)
    elif su in {"GC=F", "SI=F"}:
        mapped = su
        asset_type = "future"

    # Currency pairs can pass-through (e.g., EURUSD=X)
    elif su.endswith("=X"):
        mapped = su
        asset_type = "fx"

    # Otherwise assume equity or ETF with given ticker
    # mapped stays as provided

    # Try to fetch basic metadata
    name = None
    currency = None
    try:
        t = yf.Ticker(mapped)
        info = getattr(t, "info", {}) or {}
        name = info.get("shortName") or info.get("longName")
        currency = info.get("currency")
    except Exception:
        pass

    return {"input": s, "symbol": mapped, "asset_type": asset_type, "name": name, "currency": currency}


def get_current_price(symbol: str):
    """Return current price as a float for the resolved asset symbol.

    This ensures tool outputs remain simple scalars for price.
    """
    meta = _resolve_asset(symbol)
    mapped = meta["symbol"]
    try:
        t = yf.Ticker(mapped)
        price = None
        fi = getattr(t, "fast_info", None)
        if isinstance(fi, dict):
            price = fi.get("last_price")
        if price is None:
            info = getattr(t, "info", {}) or {}
            price = info.get("regularMarketPrice")
        if price is None:
            hist = t.history(period="5d")
            if isinstance(hist, pd.DataFrame) and not hist.empty:
                price = float(hist["Close"].iloc[-1])
        return None if price is None else float(price)
    except Exception:
        return None


def get_historical_data(symbol: str):
    """Return recent historical OHLCV using yfinance (last 30 calendar days), with metadata.

    Output: list of { Date (ISO), Close, Open?, High?, Low?, Volume?, symbol, name, currency, asset_type }.
    """
    meta = _resolve_asset(symbol)
    mapped = meta["symbol"]
    try:
        # Pull ~30 days including today; yfinance will handle market days
        t = yf.Ticker(mapped)
        df = t.history(period="1mo", interval="1d")

        if df is None or df.empty:
            # Fallback: explicit date range last 45 days to increase chance of data
            end = datetime.now(timezone.utc)
            start = end - timedelta(days=45)
            df = yf.download(mapped, start=start.date().isoformat(), end=end.date().isoformat(), interval="1d")

        if df is None or df.empty:
            return []

        # Ensure we have Date column in ISO format
        df = df.reset_index()
        date_col = "Date" if "Date" in df.columns else ("Datetime" if "Datetime" in df.columns else df.columns[0])
        df["Date"] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
        df = df.dropna(subset=["Date"])  # drop invalid rows
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        out = df[["Date", *cols]].sort_values(by="Date", ascending=False).head(30).copy()
        out["symbol"] = meta["symbol"]
        out["name"] = meta.get("name")
        out["currency"] = meta.get("currency")
        out["asset_type"] = meta.get("asset_type")

        return out.to_dict(orient="records")
    except Exception:
        return []

def get_asset_info(symbol: str):
    """Return basic metadata for the resolved asset to help verify correctness."""
    meta = _resolve_asset(symbol)
    return meta
