# streamlit_app.py
import streamlit as st
from utils import chat_with_llm, get_portfolio, add_to_portfolio, delete_from_portfolio, create_chat, list_chats, rename_chat, append_message
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="LLM Stock Platform", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Chat", "Portfolio"])

provider = st.sidebar.selectbox("LLM Provider", ["openai", "deepseek", "grok", "gemini", "local"])

# --- CHAT PAGE ---
if page == "Chat":
    st.title("💬 LLM Stock Advisor")

    # Initialize a fresh chat session on page load
    if "session_id" not in st.session_state or st.session_state.get("reset_chat", True):
        new = create_chat()
        st.session_state["session_id"] = new.get("id")
        st.session_state["reset_chat"] = False

    session_id = st.session_state.get("session_id")

    # Sidebar: list chats ordered by updated_at desc
    st.sidebar.subheader("Chat History")
    sessions = list_chats()
    titles = [f"{s.get('title','')}" for s in sessions]
    ids = [s.get("id") for s in sessions]
    if sessions:
        idx = st.sidebar.selectbox("Select a chat", options=list(range(len(sessions))), format_func=lambda i: titles[i])
        selected_id = ids[idx]
        if selected_id != session_id:
            st.session_state["session_id"] = selected_id
            session_id = selected_id
        # Rename current chat
        new_title = st.sidebar.text_input("Rename chat", value=sessions[idx].get("title", ""))
        if st.sidebar.button("Save Title"):
            rename_chat(session_id, new_title)

    message = st.text_area("Your message")
    symbol = st.text_input("Optional: Stock symbol for focused analysis")
    include_portfolio = st.checkbox("Let LLM see your portfolio")
    use_crew = st.checkbox("Use Crew (multi-agent)", value=True)
    use_history = st.checkbox("Use chat history as context", value=False)

    if st.button("Send"):
        msg = message
        if include_portfolio:
            # Compact portfolio context
            pf = get_portfolio()
            if isinstance(pf, dict) and pf:
                lines = []
                for sym, rec in pf.items():
                    if isinstance(rec, dict):
                        qty = rec.get("quantity", 0)
                        avg_buy = rec.get("avg_buy", 0)
                        last_sell = rec.get("avg_sell", None)
                        realized = rec.get("realized_profit", 0)
                        lines.append(f"{sym}: qty={qty}, avg_buy={avg_buy}, last_sell={last_sell}, realized_profit={realized}")
                if lines:
                    summary = "\n".join(lines)
                    msg = f"{message}\n\n[Portfolio Summary]\n{summary}"

        # Persist the user's message regardless
        if session_id:
            append_message(session_id, "user", msg)

        # Pass use_crew flag and history settings via utils
        response = chat_with_llm(provider, msg, symbol, use_crew=use_crew, use_history=use_history, session_id=session_id)
        st.write("### Response:")
        st.write(response)

        # Save assistant message to history
        if session_id:
            append_message(session_id, "assistant", response)


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

        # Realized profit only (avoid FutureWarning by coercing to numeric first)
        if "realized_profit" in df.columns:
            rp_series = pd.to_numeric(df["realized_profit"], errors="coerce").fillna(0.0)
        else:
            rp_series = pd.Series([], dtype="float64")
        total_realized = float(rp_series.sum())
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