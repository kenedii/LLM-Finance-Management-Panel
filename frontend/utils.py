# utils.py
import requests

API_URL = "http://localhost:8000"

def chat_with_llm(provider, message, symbol=None, use_autogen=True, use_history=False, session_id=None, use_tools=False,
                  openai_model=None, deepseek_model=None, anthropic_model=None, gemini_model=None, xai_model=None, local_model_path=None):
    r = requests.post(f"{API_URL}/chat", json={
        "provider": provider,
        "message": message,
        "symbol": symbol,
        "use_crew": False if use_autogen else True,
        "use_history": use_history,
        "session_id": session_id,
        "use_tools": use_tools,
        "use_autogen": bool(use_autogen),
        "openai_model": openai_model,
        "deepseek_model": deepseek_model,
        "anthropic_model": anthropic_model,
        "gemini_model": gemini_model,
        "xai_model": xai_model,
        "local_model_path": local_model_path,
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

def open_env_in_notepad():
    """Trigger backend to open the .env file in Notepad (Windows)."""
    requests.post(f"{API_URL}/settings/open_env")

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

# ---- Agents config (Autogen) ----
def get_agents_config():
    return requests.get(f"{API_URL}/agents/config").json()

def set_agents_config(agents: list[dict]):
    return requests.post(f"{API_URL}/agents/config", json={"agents": agents}).json()
