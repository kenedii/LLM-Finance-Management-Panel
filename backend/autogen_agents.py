# autogen_agents.py

import autogen
import json
import os
from typing import List, Dict, Any

# --- Module Imports (Adjust as necessary for your project structure) ---
try:
    from backend.llm_provider import LLMProvider
    from backend.tools import get_current_price, get_historical_data
except ImportError:
    from llm_provider import LLMProvider
    from tools import get_current_price, get_historical_data

def load_agents_config() -> List[Dict[str, str]]:
    """Loads agent configurations from agents_config.json."""
    from pathlib import Path
    # Assumes AGENTS_FILE path logic is the same as in main.py
    AGENTS_FILE = Path(__file__).with_name("agents_config.json")
    try:
        with AGENTS_FILE.open("r", encoding="utf-8") as f:
            return json.load(f).get("agents", [])
    except Exception:
        # Fallback agents
        return [
            {"name": "Prediction Analyst", "role": "Analyst", "goal": "Analyze financial data and answer questions.", "backstory": "Expert financial analyst."},
        ]

def run_autogen_conversation(provider: str, user_message: str, history: List[Dict[str, str]]) -> str:
    """
    Runs the Autogen group chat, captures the full transcript, and formats agent contributions.
    """
    
    # 1. Configuration Setup
    # Ensure LLMProvider.get_autogen_config() returns a dict usable by Autogen
    llm_config = LLMProvider.get_autogen_config(provider) 
    agent_configs = load_agents_config()
    
    # 2. Define Tools and Functions
    available_tools = {
        "get_current_price": get_current_price,
        "get_historical_data": get_historical_data
    }
    
    # 3. Create Agents
    financial_agents = []
    
    # Instantiate the User Proxy Agent (initiates the conversation and executes code)
    user_proxy = autogen.UserProxyAgent(
        name="User_Proxy",
        max_consecutive_auto_reply=10, 
        # Crucial: Termination message must be explicit and unique
        is_termination_msg=lambda x: "TERMINATE" in x.get("content", "").upper(),
        human_input_mode="NEVER",
        code_execution_config={"work_dir": "coding", "use_docker": False},
        llm_config=llm_config,
        function_map=available_tools
    )
    
    # Create the specialized agents
    for cfg in agent_configs:
        agent = autogen.AssistantAgent(
            name=cfg["name"].replace(" ", "_"), # Ensure valid Autogen name
            system_message=(
                f"You are the **{cfg['name']}** (Role: {cfg['role']}). "
                f"Your Goal: {cfg['goal']}. "
                f"Backstory: {cfg['backstory']}. "
                "Collaborate and provide your contribution to the final response. "
                "When a clear consensus is reached, the FINAL agent to speak MUST end their full response with the keyword 'TERMINATE' to signal completion."
            ),
            llm_config=llm_config,
        )
        financial_agents.append(agent)
    
    # 4. Create Group Chat and Manager
    all_agents = [user_proxy] + financial_agents
    
    group_chat = autogen.GroupChat(
        agents=all_agents,
        messages=[],
        max_round=20,
        # The manager orchestrates the flow
        manager=autogen.GroupChatManager(
            groupchat=group_chat,
            llm_config=llm_config,
            # Set verbose to 1 or 2 if you want the manager's internal decisions printed to the console
            verbose=0 
        )
    )

    # 5. Initiate the chat
    # Autogen's `initiate_chat` expects a message to kick off the current round
    user_proxy.initiate_chat(group_chat.manager, message=user_message)

    # 6. Capture and Format the Output (The Enhanced Fix)
    
    # Get the full transcript from the group_chat object
    full_transcript = group_chat.messages
    
    formatted_output = "## 💬 Agent Collaboration Log\n\n"
    final_answer = ""
    
    # Iterate backwards to find the FINAL answer quickly, then format the full log
    for msg in reversed(full_transcript):
        content = msg.get("content", "").strip()
        if "TERMINATE" in content.upper() and not final_answer:
            # Capture the final clean answer
            final_answer = content.upper().split("TERMINATE", 1)[0].strip()
            
    # Fallback if TERMINATE was missed or if the last message is the answer
    if not final_answer and full_transcript:
        final_answer = full_transcript[-1].get("content", "No final response extracted.")


    # Now format the detailed log
    # We will build the log from the start of the conversation (forward)
    for msg in full_transcript:
        speaker = msg.get("name", "System")
        content = msg.get("content", "").strip()

        # Clean up the TERMINATE flag for display in the log
        clean_content = content.upper().split("TERMINATE", 1)[0].strip()
        
        # Skip the initial user message that started the chat and empty messages
        if msg["role"] == "user" and clean_content == user_message:
            continue
        
        # Check if the content is from a tool/function call (role='function')
        if msg.get("role") == "function":
            formatted_output += f"### 🛠️ Tool Result (Called by {speaker}):\n"
            # Limit tool result output for cleaner reading
            preview = clean_content[:500] + "..." if len(clean_content) > 500 else clean_content
            formatted_output += f"```json\n{preview}\n```\n\n---\n"
        
        # Check if the content is a message from an Agent
        elif clean_content:
            formatted_output += f"### 👤 {speaker}:\n"
            formatted_output += f"> {clean_content}\n\n---\n"
            
    # Structure the final response to ensure the log is included
    full_response = (
        f"## ✅ Final Consensus Recommendation\n"
        f"{final_answer}\n\n"
        f"---\n"
        f"{formatted_output}"
    )

    return full_response