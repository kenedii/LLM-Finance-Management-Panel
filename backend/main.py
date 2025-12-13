# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import json
from pathlib import Path
import os
from dotenv import load_dotenv
from datetime import datetime

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
    use_crew: bool = True
    use_history: bool = False
    session_id: str | None = None
    use_tools: bool = False
    # When asking for multiple symbols explicitly (optional override)
    symbols: list[str] | None = None


class PortfolioEntry(BaseModel):
    symbol: str
    avg_buy: float
    avg_sell: float | None = None
    quantity: float = 0.0


@app.post("/chat")
def chat(req: ChatRequest):
    # Build context from history if requested
    history_messages = []
    if req.use_history and req.session_id:
        chats = load_chats()
        session = next((s for s in chats.get("sessions", []) if s.get("id") == req.session_id), None)
        if session:
            for m in session.get("messages", []):
                # Map to generic chat format
                history_messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})

    # Optional: direct LLM mode
    if not req.use_crew:
        try:
            # Use provider factory directly (supports OpenAI-compatible for deepseek)
            try:
                from backend.llm_provider import LLMProvider
            except Exception:
                from llm_provider import LLMProvider

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
            # If portfolio is loaded, add its symbols when user asks generically about "my portfolio"
            portfolio_syms = []
            if "portfolio" in (req.message.lower()):
                try:
                    portfolio_syms = list(load_portfolio().keys())
                except Exception:
                    portfolio_syms = []
            candidate_symbols.extend([s.upper() for s in portfolio_syms])
            # Deduplicate and cap to a small batch to avoid long calls
            candidate_symbols = list(dict.fromkeys(candidate_symbols))[:10]

            # Optional tools: proactively run for multiple symbols, even if model doesn't explicitly ask
            tool_results = []
            tool_context_header = (
                "You have access to two tools and should use them autonomously when the user asks for prices or history: "
                "get_current_price(symbol: str) and get_historical_data(symbol: str). "
                "Do not ask the user to confirm tool usage. If the request is clear, fetch the data and provide the final answer. "
                "Only ask follow-up questions if the user's instructions are strictly unclear or contradictory."
            )
            tool_context = tool_context_header
            if req.use_tools and candidate_symbols:
                try:
                    try:
                        from backend.tools import get_current_price, get_historical_data
                    except Exception:
                        from tools import get_current_price, get_historical_data
                    for sym in candidate_symbols:
                        # Current price
                        try:
                            price = get_current_price(sym)
                        except Exception as e:
                            price = {"error": str(e)}
                        # Short history preview
                        try:
                            hist = get_historical_data(sym) or []
                            hist_preview = hist[:10] if isinstance(hist, list) else []
                        except Exception as e:
                            hist_preview = {"error": str(e)}
                        tool_results.append({"symbol": sym, "current_price": price, "recent_history": hist_preview})
                    tool_context = (
                        tool_context_header + "\n[Tools]\nBatchResults: " + json.dumps(tool_results, ensure_ascii=False)
                    )
                except Exception as e:
                    tool_context = f"[ToolsError] {e}"

            # OpenAI-style client first (llm.chat.completions.create)
            if hasattr(llm, "chat") and hasattr(llm.chat, "completions") and callable(getattr(llm.chat.completions, "create", None)):
                # Choose sensible default model based on requested provider, not base URL
                p = (req.provider or "openai").strip().lower()
                if p == "deepseek":
                    default_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
                else:
                    default_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                # Always include a system message enumerating available tools
                messages = [
                    *history_messages,
                    {"role": "system", "content": "You can autonomously use tools to answer questions requiring current price or historical data. "
                                                "Available tools: get_current_price(symbol: str), get_historical_data(symbol: str). "
                                                "Avoid asking for confirmation; provide a final, data-backed answer unless the query is unclear."},
                    {"role": "user", "content": f"{tool_context}\n\n{req.message}"},
                ]
                comp = llm.chat.completions.create(
                    model=default_model,
                    messages=messages,
                    temperature=0.7,
                )
                first_text = comp.choices[0].message.content if comp and getattr(comp, "choices", None) else ""

                # Tool-calling loop: support single or multiple tool requests via JSON
                try:
                    import json as _json
                    tc = _json.loads(first_text)
                    tool_batch = []
                    if isinstance(tc, dict):
                        tool_batch = [tc]
                    elif isinstance(tc, list):
                        tool_batch = tc

                    if req.use_tools and tool_batch:
                        # Execute all requested tools
                        results = []
                        try:
                            try:
                                from backend.tools import get_current_price, get_historical_data
                            except Exception:
                                from tools import get_current_price, get_historical_data
                            for item in tool_batch:
                                tname = item.get("tool")
                                sym = (item.get("symbol") or req.symbol or "").upper()
                                if not sym:
                                    continue
                                if tname == "get_current_price":
                                    try:
                                        results.append({"symbol": sym, "current_price": get_current_price(sym)})
                                    except Exception as te:
                                        results.append({"symbol": sym, "error": str(te)})
                                elif tname == "get_historical_data":
                                    try:
                                        hist = get_historical_data(sym) or []
                                        results.append({"symbol": sym, "recent_history": hist[:50]})
                                    except Exception as te:
                                        results.append({"symbol": sym, "error": str(te)})
                        except Exception as te:
                            results = [{"error": str(te)}]

                        follow_messages = messages + [
                            {"role": "system", "content": f"Tool results: {json.dumps(results, ensure_ascii=False)}"},
                            {"role": "user", "content": "Using the tool results above, provide the final answer with clear, current figures and cite the date range for history."},
                        ]
                        comp2 = llm.chat.completions.create(
                            model=default_model,
                            messages=follow_messages,
                            temperature=0.7,
                        )
                        final_text = comp2.choices[0].message.content if comp2 and getattr(comp2, "choices", None) else ""
                        return {"response": final_text}
                except Exception:
                    pass

                # If tools were enabled but the model didn't ask, we already provided tool_context with batch results.
                # Ask the model to incorporate them explicitly.
                if req.use_tools and tool_results:
                    follow_messages = messages + [
                        {"role": "system", "content": f"Tool results: {json.dumps(tool_results, ensure_ascii=False)}"},
                        {"role": "user", "content": "Using the tool results above, provide the final answer with clear, current figures and cite the date range for history."},
                    ]
                    comp2 = llm.chat.completions.create(
                        model=default_model,
                        messages=follow_messages,
                        temperature=0.7,
                    )
                    final_text = comp2.choices[0].message.content if comp2 and getattr(comp2, "choices", None) else first_text
                    return {"response": final_text}

                return {"response": first_text}

            # LocalPyTorchLLM or any object exposing a callable .chat(messages, ...)
            if callable(getattr(llm, "chat", None)):
                messages = [
                    *history_messages,
                    {"role": "system", "content": "You can autonomously use tools to answer questions requiring current price or historical data. "
                                                "Available tools: get_current_price(symbol: str), get_historical_data(symbol: str). "
                                                "Avoid asking for confirmation; provide a final, data-backed answer unless the query is unclear."},
                    {"role": "user", "content": f"{tool_context}\n\n{req.message}"},
                ]
                text = llm.chat(messages, max_tokens=256)
                # Attempt the same tool-calling pattern for local models
                try:
                    import json as _json
                    tc = _json.loads(text)
                    tool_batch = []
                    if isinstance(tc, dict):
                        tool_batch = [tc]
                    elif isinstance(tc, list):
                        tool_batch = tc
                    if req.use_tools and tool_batch:
                        results = []
                        try:
                            try:
                                from backend.tools import get_current_price, get_historical_data
                            except Exception:
                                from tools import get_current_price, get_historical_data
                            for item in tool_batch:
                                tname = item.get("tool")
                                sym = (item.get("symbol") or req.symbol or "").upper()
                                if not sym:
                                    continue
                                if tname == "get_current_price":
                                    try:
                                        results.append({"symbol": sym, "current_price": get_current_price(sym)})
                                    except Exception as te:
                                        results.append({"symbol": sym, "error": str(te)})
                                elif tname == "get_historical_data":
                                    try:
                                        hist = get_historical_data(sym) or []
                                        results.append({"symbol": sym, "recent_history": hist[:50]})
                                    except Exception as te:
                                        results.append({"symbol": sym, "error": str(te)})
                        except Exception as te:
                            results = [{"error": str(te)}]

                        final = llm.chat(messages + [
                            {"role": "system", "content": f"Tool results: {json.dumps(results, ensure_ascii=False)}"},
                            {"role": "user", "content": "Using the tool results above, provide the final answer with clear, current figures and cite the date range for history."},
                        ], max_tokens=256)
                        return {"response": final}
                except Exception:
                    pass
                if req.use_tools and tool_results:
                    final = llm.chat(messages + [
                        {"role": "system", "content": f"Tool results: {json.dumps(tool_results, ensure_ascii=False)}"},
                        {"role": "user", "content": "Using the tool results above, provide the final answer with clear, current figures and cite the date range for history."},
                    ], max_tokens=256)
                    return {"response": final}
                return {"response": text}

            return {"response": "Direct chat not supported for this provider."}
        except Exception as e:
            return {"response": f"Direct LLM error: {e}"}

    # Crew mode (default): lazy import avoids Windows SIGHUP error at startup.
    # Try package import first, then local module import depending on CWD.
    try:
        from backend.agents import build_agents  # when running from repo root
    except Exception:
        try:
            from agents import build_agents  # when running inside backend directory
        except Exception as e:
            return {"response": f"Agents unavailable on this platform: {e}"}

    crew = build_agents(req.provider)
    # Crew mode: we only send current user message; agents can be made to use history if desired
    task = {"role": "User", "content": req.message}
    result = crew.run(task)
    return {"response": getattr(result, "raw", str(result))}


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
    - If avg_sell is set   -> SELL: decrease quantity and add realized profit = (sell - avg_buy) * qty_sold.
    """
    pf = load_portfolio()
    sym = entry.symbol

    current = pf.get(sym)
    if not current:
        # Initialize record for a new symbol
        current = {
            "symbol": sym,
            "avg_buy": float(entry.avg_buy) if entry.avg_sell is None else 0.0,
            "avg_sell": None,
            "quantity": 0.0,
            "realized_profit": 0.0,
        }

    qty = float(entry.quantity or 0.0)

    if entry.avg_sell is None:
        # BUY: merge with weighted-average
        prev_qty = float(current.get("quantity", 0.0))
        prev_avg_buy = float(current.get("avg_buy", entry.avg_buy))
        new_qty = prev_qty + qty

        if new_qty > 0:
            new_avg_buy = ((prev_avg_buy * prev_qty) + (float(entry.avg_buy) * qty)) / new_qty
        else:
            new_avg_buy = prev_avg_buy  # safe fallback

        current["avg_buy"] = round(new_avg_buy, 6)
        current["quantity"] = round(new_qty, 6)
        current["avg_sell"] = None
    else:
        # SELL: realize profit and reduce quantity
        sell_price = float(entry.avg_sell)
        prev_qty = float(current.get("quantity", 0.0))
        prev_avg_buy = float(current.get("avg_buy", entry.avg_buy))

        # If user provided both buy and sell in one submission and no holdings exist,
        # treat it as a buy-then-sell for the given quantity to realize profit.
        if prev_qty <= 0 and entry.avg_buy and qty > 0:
            # bootstrap a buy
            current["avg_buy"] = float(entry.avg_buy)
            current["quantity"] = float(qty)
            prev_qty = current["quantity"]
            prev_avg_buy = current["avg_buy"]

        # After optional bootstrap, if still no holdings, just record last sell price and exit
        if prev_qty <= 0:
            current["avg_sell"] = round(sell_price, 6)
            pf[sym] = current
            save_portfolio(pf)
            return {"status": "no-holdings", "portfolio": pf}

        sell_qty = min(qty, prev_qty)
        if sell_qty > 0:
            realized = (sell_price - prev_avg_buy) * sell_qty
            current["realized_profit"] = round(float(current.get("realized_profit", 0.0)) + realized, 6)
            current["quantity"] = round(prev_qty - sell_qty, 6)
            current["avg_sell"] = round(sell_price, 6)

    pf[sym] = current
    save_portfolio(pf)
    return {"status": "updated", "portfolio": pf}


@app.post("/portfolio/delete")
def delete_stock(symbol: str):
    pf = load_portfolio()
    pf.pop(symbol, None)
    save_portfolio(pf)
    return {"status": "deleted", "portfolio": pf}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="127.0.0.1", port=int(os.getenv("PORT", "8000")), reload=True)