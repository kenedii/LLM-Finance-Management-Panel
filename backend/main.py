# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import json
from pathlib import Path
import os
from dotenv import load_dotenv

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


class ChatRequest(BaseModel):
    provider: str
    message: str
    symbol: str | None = None
    use_crew: bool = True


class PortfolioEntry(BaseModel):
    symbol: str
    avg_buy: float
    avg_sell: float | None = None
    quantity: float = 0.0


@app.post("/chat")
def chat(req: ChatRequest):
    # Optional: direct LLM mode
    if not req.use_crew:
        try:
            # Use provider factory directly (supports OpenAI-compatible for deepseek)
            try:
                from backend.llm_provider import LLMProvider
            except Exception:
                from llm_provider import LLMProvider

            llm = LLMProvider.get(req.provider)

            # Handle OpenAI-compatible clients
            if hasattr(llm, "chat"):
                # LocalPyTorchLLM.chat interface
                text = llm.chat([{"role": "user", "content": req.message}], max_tokens=256)
                return {"response": text}

            # OpenAI Python SDK
            if hasattr(llm, "chat") and hasattr(llm.chat, "completions"):
                model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
                comp = llm.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": req.message}],
                    temperature=0.7,
                )
                text = comp.choices[0].message.content if comp and comp.choices else ""
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
    task = {"role": "User", "content": req.message}
    result = crew.run(task)
    return {"response": getattr(result, "raw", str(result))}


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