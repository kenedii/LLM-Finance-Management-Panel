# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import json
from agents import build_agents

app = FastAPI()

PORTFOLIO_FILE = "portfolio_db.json"

def load_portfolio():
    try:
        return json.load(open(PORTFOLIO_FILE))
    except:
        return {}

def save_portfolio(data):
    json.dump(data, open(PORTFOLIO_FILE, "w"), indent=4)


class ChatRequest(BaseModel):
    provider: str
    message: str
    symbol: str | None = None


class PortfolioEntry(BaseModel):
    symbol: str
    avg_buy: float
    avg_sell: float | None = None
    quantity: float = 0.0


@app.post("/chat")
def chat(req: ChatRequest):
    crew = build_agents(req.provider)
    task = {
        "role": "User",
        "content": req.message
    }

    result = crew.run(task)
    return {"response": result.raw}


@app.get("/portfolio")
def get_portfolio():
    return load_portfolio()


@app.post("/portfolio/add")
def add_stock(entry: PortfolioEntry):
    pf = load_portfolio()
    pf[entry.symbol] = entry.dict()
    save_portfolio(pf)
    return {"status": "added", "portfolio": pf}


@app.post("/portfolio/delete")
def delete_stock(symbol: str):
    pf = load_portfolio()
    pf.pop(symbol, None)
    save_portfolio(pf)
    return {"status": "deleted", "portfolio": pf}
