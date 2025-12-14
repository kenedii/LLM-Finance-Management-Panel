from fastapi import FastAPI
from pydantic import BaseModel
import json
from pathlib import Path
import os
from dotenv import load_dotenv
from datetime import datetime
import subprocess
from typing import List, Dict, Any

# --- CRITICAL FIX: Centralize Imports for Autogen/LLM/Tools ---
# CrewAI imports (like build_agents) and logic have been removed.
try:
    from backend.llm_provider import LLMProvider
    from backend.tools import get_current_price, get_historical_data
    from backend.autogen_agents import run_autogen_conversation
except Exception:
    # Fallback for local execution
    from llm_provider import LLMProvider
    from tools import get_current_price, get_historical_data
    from autogen_agents import run_autogen_conversation 

app = FastAPI()

# Load environment variables from backend/.env so providers get API keys
_ENV_PATH = Path(__file__).with_name(".env")
try:
    load_dotenv(dotenv_path=_ENV_PATH)
except Exception:
    # Fallback: try project root .env
    load_dotenv()

# Store the DB next to this file so it works regardless of CWD
PORTFOLIO_FILE = Path(__file__).with_name("portfolio_db.json")
CHAT_DB_FILE = Path(__file__).with_name("chat_db.json")

def load_portfolio():
    try:
        with PORTFOLIO_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
            # Ensure realized_profit exists for each symbol
            for sym, rec in list(data.items()):
                if isinstance(rec, dict) and "symbol" in rec:
                    rec.setdefault("realized_profit", 0.0)
            return data
    except Exception:
        return {}

def save_portfolio(data):
    with PORTFOLIO_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def load_chats():
    try:
        with CHAT_DB_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"sessions": []}


def save_chats(data):
    with CHAT_DB_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


class ChatRequest(BaseModel):
    provider: str
    message: str
    symbol: str | None = None
    # CrewAI has been removed, this is kept for compatibility but defaults to False
    use_crew: bool = False 
    use_autogen: bool = False
    use_history: bool = False
    session_id: str | None = None
    use_tools: bool = False
    # When asking for multiple symbols explicitly (optional override)
    symbols: list[str] | None = None
    # Optional model overrides per provider
    openai_model: str | None = None
    deepseek_model: str | None = None
    anthropic_model: str | None = None
    gemini_model: str | None = None
    xai_model: str | None = None
    local_model_path: str | None = None


class PortfolioEntry(BaseModel):
    symbol: str
    avg_buy: float
    avg_sell: float | None = None
    quantity: float = 0.0


@app.post("/chat")
def chat(req: ChatRequest):
    # Always return a JSON dictionary with a "response" key, even on failure.
    try:
        # Build context from history if requested
        history_messages: List[Dict[str, str]] = []
        if req.use_history and req.session_id:
            chats = load_chats()
            session = next((s for s in chats.get("sessions", []) if s.get("id") == req.session_id), None)
            if session:
                for m in session.get("messages", []):
                    # Map to generic chat format
                    history_messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})

        # --- 1. Autogen Agent Mode ---
        if req.use_autogen:
            history = history_messages if req.use_history else []
            # This calls your Autogen implementation in autogen_agents.py
            result = run_autogen_conversation(provider=req.provider, user_message=req.message, history=history)
            return {"response": result}

        # --- 2. Direct LLM Mode (Default/Fallback) ---
        # This mode is used when use_autogen is False. It makes a direct call to the LLM.
        llm = LLMProvider.get(req.provider)

        # Helper: detect symbols from request, message text, and portfolio
        def _detect_symbols_from_text(text: str) -> list[str]:
            syms = set()
            words = [w.strip().upper().strip('.,:;!()[]{}') for w in (text or '').split()]
            # Simple heuristic: uppercase tokens of 1-6 chars (AAPL, TSLA, BTC, ETH)
            for w in words:
                if 1 <= len(w) <= 6 and w.isalnum() and w.upper() == w and not w.isdigit():
                    syms.add(w)
            return list(syms)

        # Build list of candidate symbols
        candidate_symbols: list[str] = []
        if req.symbol:
            candidate_symbols.append(req.symbol.upper())
        if req.symbols:
            candidate_symbols.extend([s.upper() for s in req.symbols])
        candidate_symbols.extend(_detect_symbols_from_text(req.message))
        
        portfolio_syms = []
        if "portfolio" in (req.message.lower()):
            try:
                portfolio_syms = list(load_portfolio().keys())
            except Exception:
                portfolio_syms = []
        candidate_symbols.extend([s.upper() for s in portfolio_syms])
        
        # Deduplicate and cap
        candidate_symbols = list(dict.fromkeys(candidate_symbols))[:10]

        tool_results = []
        tool_context = ""
        
        # --- PROACTIVE Tool Run (Simpler non-agentic mode) ---
        if req.use_tools and candidate_symbols:
            tool_results_list = []
            # Use the imported tool functions
            for sym in candidate_symbols:
                try:
                    # Use the imported tool functions
                    price = get_current_price(sym)
                except Exception as e:
                    price = {"error": str(e)}
                try:
                    hist = get_historical_data(sym) or []
                    # Only include a short preview for context
                    hist_preview = hist[:10] if isinstance(hist, list) else [] 
                except Exception as e:
                    hist_preview = {"error": str(e)}
                
                tool_results_list.append({"symbol": sym, "current_price": price, "recent_history_preview": hist_preview})
            
            tool_context = f"PRE-FETCHED DATA FOR ANALYSIS:\n{json.dumps(tool_results_list, ensure_ascii=False)}"
            tool_results = tool_results_list # Keep for the final prompt

        # --- LLM Chat Invocation ---
        messages: List[Dict[str, str]] = [
            *history_messages,
            {"role": "system", "content": "You are a helpful Financial Analyst. Your goal is to provide a concise, data-backed answer."},
        ]
        
        # If tool data was pre-fetched, inject it as a context message before the user message
        if tool_context:
            messages.append({"role": "system", "content": tool_context})

        messages.append({"role": "user", "content": req.message})

        p = (req.provider or "").strip().lower()
        
        # Determine per-provider model override
        model_override = None
        if p == "deepseek":
            model_override = req.deepseek_model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        elif p == "openai":
            model_override = req.openai_model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        elif p in {"anthropic", "claude"}:
            model_override = req.anthropic_model or os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-202410")
        elif p in {"gemini", "google"}:
            model_override = req.gemini_model or os.getenv("GEMINI_MODEL", "gemini-1.5-pro")
        elif p in {"grok", "xai"}:
            model_override = req.xai_model or os.getenv("XAI_MODEL", "grok-beta")

        # Call LLM
        # Local models ignore 'model' override
        if p in {"local", "pytorch", "hf", "transformers"}:
            final_text = llm.chat(messages, max_tokens=640)
        else:
            final_text = llm.chat(messages, max_tokens=640, model=model_override)
        
        return {"response": final_text}


    except Exception as e:
        # CRITICAL FIX: Ensure a valid JSON response even if an exception occurs
        return {"response": f"An unhandled backend error occurred: {e}"}


# -----------------------------
# Chat sessions management
# -----------------------------
class CreateChatRequest(BaseModel):
    title: str | None = None


@app.post("/chats")
def create_chat(req: CreateChatRequest):
    chats = load_chats()
    now = datetime.now()
    chat_id = f"chat-{int(now.timestamp()*1000)}"
    title = req.title or now.strftime("Chat %Y-%m-%d %H:%M")
    session = {
        "id": chat_id,
        "title": title,
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
        "messages": [],
    }
    chats.setdefault("sessions", []).append(session)
    save_chats(chats)
    return {"id": chat_id, "title": title}


@app.get("/chats")
def list_chats():
    chats = load_chats()
    sessions = chats.get("sessions", [])
    # Order by updated_at desc
    sessions = sorted(sessions, key=lambda s: s.get("updated_at", s.get("created_at", "")), reverse=True)
    return {"sessions": sessions}


class RenameChatRequest(BaseModel):
    title: str


@app.post("/chats/{chat_id}/rename")
def rename_chat(chat_id: str, req: RenameChatRequest):
    chats = load_chats()
    for s in chats.get("sessions", []):
        if s.get("id") == chat_id:
            s["title"] = req.title
            s["updated_at"] = datetime.now().isoformat()
            break
    save_chats(chats)
    return {"status": "renamed"}


class AppendMessageRequest(BaseModel):
    role: str
    content: str


def _append_message(chat_id: str, role: str, content: str):
    chats = load_chats()
    sessions = chats.setdefault("sessions", [])
    session = next((s for s in sessions if s.get("id") == chat_id), None)
    now_iso = datetime.now().isoformat()
    if not session:
        # Create session on-the-fly if missing
        session = {"id": chat_id, "title": datetime.now().strftime("Chat %Y-%m-%d %H:%M"), "created_at": now_iso, "updated_at": now_iso, "messages": []}
        sessions.append(session)

    session.setdefault("messages", []).append({"role": role, "content": content, "ts": now_iso})
    session["updated_at"] = now_iso

    # Auto-title on first user message if the title is generic
    if role.lower() == "user":
        _auto_title_if_needed(session, content)
    save_chats(chats)


def _auto_title_if_needed(session: dict, first_user_content: str):
    """Set a friendly title based on the first user message if still default."""
    current_title = session.get("title") or ""
    # Consider default titles that start with "Chat " as placeholders
    if not current_title or current_title.startswith("Chat "):
        preview = (first_user_content or "").strip()
        if preview:
            # first ~6 words
            parts = preview.split()
            short = " ".join(parts[:6])
            session["title"] = f"{short} — {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        else:
            session["title"] = datetime.now().strftime("Chat %Y-%m-%d %H:%M")


@app.post("/chats/{chat_id}/message")
def append_message(chat_id: str, req: AppendMessageRequest):
    _append_message(chat_id, req.role, req.content)
    return {"status": "appended"}


@app.post("/chats/{chat_id}/delete")
def delete_chat(chat_id: str):
    chats = load_chats()
    sessions = chats.get("sessions", [])
    chats["sessions"] = [s for s in sessions if s.get("id") != chat_id]
    save_chats(chats)
    return {"status": "deleted"}


@app.get("/portfolio")
def get_portfolio():
    return load_portfolio()


@app.post("/portfolio/add")
def add_stock(entry: PortfolioEntry):
    """
    Behavior:
    - If avg_sell is None -> BUY: weighted-average the buy price and increase quantity.
    - If avg_sell is set  -> SELL: decrease quantity and add realized profit = (sell - avg_buy) * qty_sold.
    """
    pf = load_portfolio()
    sym = entry.symbol.upper() # Ensure symbol is uppercase for consistency

    current = pf.get(sym)
    if not current:
        # Initialize record for a new symbol
        current = {
            "symbol": sym,
            "avg_buy": 0.0, # Initialize buy price to 0.0
            "avg_sell": None,
            "quantity": 0.0,
            "realized_profit": 0.0,
        }
    else:
        # Ensure data types are floats for calculations
        current["quantity"] = float(current.get("quantity", 0.0))
        current["avg_buy"] = float(current.get("avg_buy", 0.0))
        current["realized_profit"] = float(current.get("realized_profit", 0.0))

    qty_transacted = float(entry.quantity or 0.0)

    if entry.avg_sell is None:
        # --- BUY LOGIC ---
        buy_price = float(entry.avg_buy)
        prev_qty = current["quantity"]
        prev_avg_buy = current["avg_buy"]
        new_qty = prev_qty + qty_transacted

        if new_qty > 0:
            # Weighted Average Cost (WAC) calculation
            # New Avg Buy = [(Old Avg Buy * Old Qty) + (New Buy Price * New Qty)] / New Total Qty
            new_avg_buy = ((prev_avg_buy * prev_qty) + (buy_price * qty_transacted)) / new_qty
            current["avg_buy"] = round(new_avg_buy, 6)
        # If new_qty is 0 (e.g., initial qty was -2 and you bought 2), avg_buy is preserved.

        current["quantity"] = round(new_qty, 6)
        # We do not touch avg_sell on a buy, though we might clear it if we want to track
        # *only* the last sale price when current holdings exist. We'll leave it as None/last sell.

    else:
        # --- SELL LOGIC ---
        sell_price = float(entry.avg_sell)
        prev_qty = current["quantity"]
        prev_avg_buy = current["avg_buy"]

        # Determine the quantity sold, limited by the current holdings
        sell_qty = min(qty_transacted, prev_qty)

        if sell_qty > 0:
            # Realized Profit calculation: (Sell Price - Average Cost) * Quantity Sold
            realized = (sell_price - prev_avg_buy) * sell_qty
            current["realized_profit"] = round(current["realized_profit"] + realized, 6)
            
            # Update quantity held
            current["quantity"] = round(prev_qty - sell_qty, 6)
            
        # The key change: Update the last sell price without changing avg_buy
        current["avg_sell"] = round(sell_price, 6)
        
        # Scenario where selling more than held (e.g., shorting or error):
        # We process the sell for the held amount, and leave the remaining quantity as a potential
        # error or short position, but we don't change `avg_buy`.

    # Only save the symbol if it has an actual transaction or remaining quantity
    if current["quantity"] != 0.0 or current["realized_profit"] != 0.0 or current["avg_sell"] is not None:
        pf[sym] = current
    elif sym in pf:
        # If all values zeroed out and it exists, delete it (optional cleanup)
        del pf[sym]

    save_portfolio(pf)
    return {"status": "updated", "portfolio": pf}


@app.post("/portfolio/delete")
def delete_stock(symbol: str):
    pf = load_portfolio()
    pf.pop(symbol, None)
    save_portfolio(pf)
    return {"status": "deleted", "portfolio": pf}


# -----------------------------
# Settings utilities
# -----------------------------
@app.post("/settings/open_env")
def open_env_in_notepad():
    """Open the backend .env file in Windows Notepad for editing."""
    try:
        env_path = _ENV_PATH
        # Use Notepad on Windows; fallback to system default opener otherwise
        if os.name == "nt":
            subprocess.Popen(["notepad", str(env_path)])
        else:
            # macOS/Linux fallback: try xdg-open or open
            try:
                subprocess.Popen(["xdg-open", str(env_path)])
            except Exception:
                subprocess.Popen(["open", str(env_path)])
        return {"status": "opened"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


# -----------------------------
# Agent configuration (for Autogen)
# -----------------------------
class AgentConfig(BaseModel):
    agents: list[dict]

AGENTS_FILE = Path(__file__).with_name("agents_config.json")

def load_agents_config():
    try:
        with AGENTS_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # Default roles based on previous CrewAI purposes (now for Autogen)
        return {
            "agents": [
                {"name": "Prediction Analyst", "role": "Prediction Analyst", "goal": "Predict future performance of the asset.", "backstory": "Expert in quantitative forecasting."},
                {"name": "Short Term Market Analyst", "role": "Short Term Market Analyst", "goal": "Analyze short-term price action.", "backstory": "Expert in technical analysis."},
                {"name": "Long Term Market Analyst", "role": "Long Term Market Analyst", "goal": "Evaluate long-term viability.", "backstory": "Fundamental analyst."},
                {"name": "Risk Analyst", "role": "Risk Analyst", "goal": "Assess risk factors.", "backstory": "Risk modeling specialist."},
            ]
        }

def save_agents_config(cfg: dict):
    with AGENTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=4)


@app.get("/agents/config")
def get_agents_config():
    return load_agents_config()


@app.post("/agents/config")
def set_agents_config(req: AgentConfig):
    cfg = {"agents": req.agents}
    save_agents_config(cfg)
    return {"status": "saved"}


if __name__ == "__main__":
    import uvicorn
    # If running from repo root, use 'backend.main:app'. If running from backend, use 'main:app'.
    # Defaulting to the path Uvicorn expects when run from the root:
    uvicorn.run("backend.main:app", host="127.0.0.1", port=int(os.getenv("PORT", "8000")), reload=True)