# streamlit_app.py
import streamlit as st
from utils import chat_with_llm, get_portfolio, add_to_portfolio, delete_from_portfolio
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="LLM Stock Platform", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Chat", "Portfolio"])

provider = st.sidebar.selectbox("LLM Provider", ["openai", "deepseek", "grok", "gemini", "local"])

# --- CHAT PAGE ---
if page == "Chat":
    st.title("💬 LLM Stock Advisor")

    message = st.text_area("Your message")
    symbol = st.text_input("Optional: Stock symbol for focused analysis")

    if st.button("Send"):
        response = chat_with_llm(provider, message, symbol)
        st.write("### Response:")
        st.write(response)


# --- PORTFOLIO PAGE ---
elif page == "Portfolio":
    st.title("📊 Your Portfolio")

    pf = get_portfolio()
    df = pd.DataFrame(pf).T

    if not df.empty:
        # Ensure columns exist
        for col in ["symbol", "avg_buy", "avg_sell", "quantity", "realized_profit"]:
            if col not in df.columns:
                df[col] = 0.0 if col != "symbol" else df.index

        # Realized profit only
        total_realized = float(df.get("realized_profit", pd.Series(dtype=float)).fillna(0.0).sum())
        st.metric("Total Realized Profit", f"${total_realized:,.2f}")

        # Optional: show unrealized P/L separately for info (not part of total realized)
        # current_price = df.index.map(lambda s: yf.Ticker(s).info.get("regularMarketPrice"))
        # df["unrealized_pl"] = (current_price - df["avg_buy"]) * df["quantity"]

        st.dataframe(
            df[["symbol", "quantity", "avg_buy", "avg_sell", "realized_profit"]]
              .fillna(0.0)
              .rename(columns={"avg_sell": "last_sell"})
        )

    st.subheader("Add Asset")

    symbol = st.text_input("Symbol")
    avg_buy = st.number_input("Avg Buy", value=0.0, min_value=0.0)
    avg_sell = st.number_input("Avg Sell (Optional)", value=0.0, min_value=0.0)
    qty = st.number_input("Quantity", value=0.0)

    if st.button("Add to Portfolio"):
        # Treat 0.0 as None for avg_sell so backend sees it as a BUY unless explicitly set
        payload_avg_sell = None if avg_sell == 0.0 else avg_sell
        add_to_portfolio(symbol, avg_buy, payload_avg_sell, qty)
        st.success(f"Added {symbol}")
        st.rerun()

    st.subheader("Remove Asset")
    del_symbol = st.text_input("Symbol to remove")

    if st.button("Delete Asset"):
        delete_from_portfolio(del_symbol)
        st.success(f"Deleted {del_symbol}")
        st.rerun()