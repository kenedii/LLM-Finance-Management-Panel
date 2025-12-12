# utils.py
import requests

API_URL = "http://localhost:8000"

def chat_with_llm(provider, message, symbol=None, use_crew=True, use_history=False, session_id=None):
    r = requests.post(f"{API_URL}/chat", json={
        "provider": provider,
        "message": message,
        "symbol": symbol,
        "use_crew": use_crew,
        "use_history": use_history,
        "session_id": session_id,
    })
    return r.json()["response"]


def create_chat(title=None):
    r = requests.post(f"{API_URL}/chats", json={"title": title})
    return r.json()


def list_chats():
    r = requests.get(f"{API_URL}/chats")
    return r.json().get("sessions", [])


def rename_chat(chat_id, title):
    requests.post(f"{API_URL}/chats/{chat_id}/rename", json={"title": title})


def append_message(chat_id, role, content):
    requests.post(f"{API_URL}/chats/{chat_id}/message", json={"role": role, "content": content})


def delete_chat(chat_id):
    requests.post(f"{API_URL}/chats/{chat_id}/delete")

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
