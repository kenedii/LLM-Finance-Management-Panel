# streamlit_app.py
import streamlit as st
from utils import chat_with_llm, get_portfolio, add_to_portfolio, delete_from_portfolio, create_chat, list_chats, rename_chat, append_message, delete_chat
import pandas as pd
import yfinance as yf

st.set_page_config(page_title="LLM Stock Platform", layout="wide")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Chat", "Portfolio", "Settings"])

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
    if st.sidebar.button("New Chat"):
        # Do not persist a chat until the first message is sent; just clear selection
        st.session_state["session_id"] = None
        st.session_state["reset_chat"] = False
        st.rerun()
    if st.sidebar.button("Delete Chat"):
        if st.session_state.get("session_id"):
            delete_chat(st.session_state["session_id"])
        # Do not auto-create a new chat; leave empty until first message is sent
        st.session_state["session_id"] = None
        st.rerun()
    sessions = list_chats()
    titles = [f"{s.get('title','')}" for s in sessions]
    ids = [s.get("id") for s in sessions]
    # Selection: default to unsaved new chat when no session selected
    select_options = ["— New chat (unsaved) —"] + titles
    if session_id:
        # Try to set index to current session title
        try:
            current_index = titles.index(next(t for i,t in enumerate(titles) if ids[i] == session_id)) + 1
        except Exception:
            current_index = 0
    else:
        current_index = 0

    chosen = st.sidebar.selectbox("Select a chat", options=select_options, index=current_index)
    if chosen != "— New chat (unsaved) —":
        chosen_idx = titles.index(chosen)
        selected_id = ids[chosen_idx]
        if selected_id != session_id:
            st.session_state["session_id"] = selected_id
            session_id = selected_id

    # Rename only when a saved chat is selected
    if session_id:
        current_session_title = next((s.get("title","") for s in sessions if s.get("id") == session_id), "")
        new_title = st.sidebar.text_input("Rename chat", value=current_session_title)
        if st.sidebar.button("Save Title"):
            rename_chat(session_id, new_title)
            st.rerun()

    # Fetch current session details to render transcript
    current_session = None
    if sessions and session_id:
        for s in sessions:
            if s.get("id") == session_id:
                current_session = s
                break

    # Controls for selected chat
    st.subheader("Conversation")

    # Styles for chat bubbles and scrollable area
    st.markdown(
        """
        <style>
        .chat-container { max-height: 480px; overflow-y: auto; padding: 8px; border: 1px solid #ddd; border-radius: 8px; background: #fafafa; }
        .msg { margin: 8px 0; display: flex; }
        .bubble { padding: 10px 12px; border-radius: 14px; max-width: 80%; box-shadow: 0 1px 2px rgba(0,0,0,0.08); color: #000; }
        .user { justify-content: flex-end; }
        .user .bubble { background: #DCF8C6; }
        .assistant { justify-content: flex-start; }
        .assistant .bubble { background: #ffffff; }
        .meta { font-size: 11px; color: #000; margin-top: 4px; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    if current_session and current_session.get("messages"):
        for m in current_session["messages"]:
            role = (m.get("role") or "user").lower()
            content = m.get("content") or ""
            ts = m.get("ts") or ""
            icon = "👤" if role == "user" else "🤖"
            st.markdown(
                f'<div class="msg {role}"><div class="bubble">{icon} {content}<div class="meta">{ts}</div></div></div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("No messages yet. Start the conversation below.")
    st.markdown('</div>', unsafe_allow_html=True)

    message = st.text_area("Your message")
    symbol = st.text_input("Optional: Stock symbol for focused analysis")
    include_portfolio = st.checkbox("Let LLM see your portfolio")
    use_crew = st.checkbox("Use Crew (multi-agent)", value=True)
    use_history = st.checkbox("Use chat history as context", value=False)
    use_tools = st.checkbox("Let LLM use tools (price/history)", value=False)

    # Make tool availability explicit in the chat UI
    if use_tools:
        st.info(
            "Tools enabled: The assistant will autonomously fetch data using these tools — "
            "get_current_price(symbol) and get_historical_data(symbol). "
            "Ask for prices or history (e.g., 'Price of AAPL and TSLA' or 'Show recent history for my portfolio'). "
            "It won't ask for confirmation unless your instructions are unclear."
        )

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

        # Create session only when the first message is sent
        if not session_id:
            new = create_chat()
            st.session_state["session_id"] = new.get("id")
            session_id = st.session_state["session_id"]
        
        # Persist the user's original message (without portfolio context block)
        append_message(session_id, "user", message)

        # Pass use_crew flag and history settings via utils
        # Pull optional per-provider model overrides from session state
        openai_model = st.session_state.get("openai_model")
        deepseek_model = st.session_state.get("deepseek_model")
        anthropic_model = st.session_state.get("anthropic_model")
        gemini_model = st.session_state.get("gemini_model")
        xai_model = st.session_state.get("xai_model")
        local_model_path = st.session_state.get("local_model_path")
        response = chat_with_llm(
            provider,
            msg,
            symbol,
            use_crew=use_crew,
            use_history=use_history,
            session_id=session_id,
            use_tools=use_tools,
            openai_model=openai_model,
            deepseek_model=deepseek_model,
            anthropic_model=anthropic_model,
            gemini_model=gemini_model,
            xai_model=xai_model,
            local_model_path=local_model_path,
        )
        st.write("### Response:")
        st.write(response)

        # Save assistant message to history
        append_message(session_id, "assistant", response)
        # Refresh sidebar ordering (most-recent first)
        st.rerun()


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

# --- SETTINGS PAGE ---
elif page == "Settings":
    st.title("⚙️ Settings")
    st.subheader("Model configuration per provider")

    st.markdown("Configure the model used for each provider. These settings override defaults during chat requests.")

    # OpenAI model
    st.session_state.setdefault("openai_model", "gpt-4o-mini")
    openai_model = st.text_input("OpenAI model", value=st.session_state["openai_model"], help="Example: gpt-4o, gpt-4o-mini, o3-mini, etc.")
    # DeepSeek model
    st.session_state.setdefault("deepseek_model", "deepseek-chat")
    deepseek_model = st.text_input("DeepSeek model", value=st.session_state["deepseek_model"], help="Example: deepseek-chat, deepseek-reasoner")

    # Anthropic (Claude)
    st.session_state.setdefault("anthropic_model", "claude-3-5-sonnet-202410")
    anthropic_model = st.text_input("Anthropic (Claude) model", value=st.session_state["anthropic_model"], help="Example: claude-3-5-sonnet-202410")

    # Gemini (Google)
    st.session_state.setdefault("gemini_model", "gemini-1.5-pro")
    gemini_model = st.text_input("Gemini model", value=st.session_state["gemini_model"], help="Example: gemini-1.5-pro")

    # xAI (Grok)
    st.session_state.setdefault("xai_model", "grok-beta")
    xai_model = st.text_input("xAI (Grok) model", value=st.session_state["xai_model"], help="Example: grok-beta")

    # Local transformers model path
    st.session_state.setdefault("local_model_path", "")
    local_model_path = st.text_input("Local model path (HF)", value=st.session_state["local_model_path"], help="Path or model id for local transformers model.")

    if st.button("Save Model Settings"):
        st.session_state["openai_model"] = openai_model
        st.session_state["deepseek_model"] = deepseek_model
        st.session_state["anthropic_model"] = anthropic_model
        st.session_state["gemini_model"] = gemini_model
        st.session_state["xai_model"] = xai_model
        st.session_state["local_model_path"] = local_model_path
        st.success("Model settings saved. They will be used for future chats.")

    st.subheader("Environment configuration (.env)")
    st.markdown("Open your backend .env file in Notepad to edit API keys and defaults.")
    from utils import open_env_in_notepad
    if st.button("Open .env in Notepad"):
        open_env_in_notepad()
        st.info(".env should open in Notepad. If it does not, ensure the backend server is running and you are on Windows.")