# utils.py
import requests

API_URL = "http://localhost:8000"

def chat_with_llm(provider, message, symbol=None, use_crew=True):
    r = requests.post(f"{API_URL}/chat", json={
        "provider": provider,
        "message": message,
        "symbol": symbol,
        "use_crew": use_crew,
    })
    return r.json()["response"]

def get_portfolio():
    return requests.get(f"{API_URL}/portfolio").json()

def add_to_portfolio(symbol, avg_buy, avg_sell, qty):
    return requests.post(f"{API_URL}/portfolio/add", json={
        "symbol": symbol,
        "avg_buy": avg_buy,
        "avg_sell": avg_sell,
        "quantity": qty
    }).json()

def delete_from_portfolio(symbol):
    return requests.post(f"{API_URL}/portfolio/delete", params={"symbol": symbol}).json()
