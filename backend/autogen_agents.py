# autogen_agents.py
"""Autogen multi-agent orchestration using existing tools.

Creates an AnalystAgent that coordinates analysis and a ToolsAgent that executes
price tools. Agents are defined by a configurable JSON (agents_config.json).
"""

import os
import json
from pathlib import Path
import re
from typing import List, Dict, Any

# Reuse existing provider factory
try:
    from backend.llm_provider import LLMProvider
except Exception:
    from llm_provider import LLMProvider

# Reuse tools
try:
    from backend.tools import get_current_price, get_historical_data, get_asset_info
except Exception:
    from tools import get_current_price, get_historical_data, get_asset_info

AGENTS_FILE = Path(__file__).with_name("agents_config.json")
PORTFOLIO_FILE = Path(__file__).with_name("portfolio_db.json")


def load_agents_config():
    try:
        with AGENTS_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "agents": [
                {"name": "Prediction Analyst", "role": "Prediction Analyst", "goal": "Predict future performance of the asset.", "backstory": "Expert in quantitative forecasting."},
                {"name": "Short Term Market Analyst", "role": "Short Term Market Analyst", "goal": "Analyze short-term price action.", "backstory": "Expert in technical analysis."},
                {"name": "Long Term Market Analyst", "role": "Long Term Market Analyst", "goal": "Evaluate long-term viability.", "backstory": "Fundamental analyst."},
                {"name": "Risk Analyst", "role": "Risk Analyst", "goal": "Assess risk factors.", "backstory": "Risk modeling specialist."},
            ]
        }


def _load_portfolio_symbols() -> list[str]:
    """Helper to load symbols from the portfolio file."""
    try:
        with PORTFOLIO_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return list(data.keys())
    except Exception:
        return []

def execute_tools(tool_calls: List[Dict[str, Any]]) -> str:
    """Matches tool names to Python functions and returns a string of results."""
    results = []
    
    # Map tool names to callable functions
    tool_map = {
        "get_current_price": get_current_price,
        "get_historical_data": get_historical_data,
        "get_asset_info": get_asset_info,
    }

    for call in tool_calls:
        name = call.get("tool_name")
        params = call.get("parameters", {})
        
        func = tool_map.get(name)
        if func:
            try:
                # Assuming all tools take a single 'symbol' parameter for simplicity
                symbol = params.get("symbol")
                if not symbol:
                    res = {"error": f"Tool '{name}' called without 'symbol' parameter."}
                else:
                    # Execute the actual function
                    res = func(symbol)
                
                results.append({"tool": name, "parameters": params, "output": res})
            except Exception as e:
                results.append({"tool": name, "parameters": params, "output": f"Tool execution error: {e}"})
        else:
            results.append({"tool": name, "parameters": params, "output": f"Error: Tool '{name}' not found or implemented."})
            
    return json.dumps(results, ensure_ascii=False)


def run_autogen_conversation(provider: str, user_message: str, history: list[dict] | None = None) -> str:
    """Agentic tool-use conversation loop."""
    llm = LLMProvider.get(provider)
    
    # 1. System Prompt for the *first* turn
    # This guides the LLM to use the specified JSON format and tells it about the portfolio.
    portfolio_symbols = _load_portfolio_symbols()
    pf_str = f"Your user's portfolio assets are: {', '.join(portfolio_symbols)}. " if portfolio_symbols else ""
    
    system_prompt = (
        "You are an expert Finance Analyst. Your goal is to answer the user's query by gathering required data. "
        f"{pf_str}"
        "When you need asset data (prices, history, or info), you MUST output a JSON list of tools to execute, "
        "enclosed in triple backticks (```json...```). Use the format: "
        "[\n  { \"tool_name\": \"get_current_price\", \"parameters\": { \"symbol\": \"<TICKER>\" } },\n  { \"tool_name\": \"get_historical_data\", \"parameters\": { \"symbol\": \"<TICKER>\" } }\n] "
        "Available tool names: get_current_price, get_historical_data, get_asset_info. "
        "Do not include any other text besides the JSON in the tool-call step. "
        "After receiving the tool results (provided by the system), synthesize the final answer for the user."
    )
    
    # Initialize the message history
    messages = [{"role": "system", "content": system_prompt}]
    if history: 
        # Only use the user/assistant messages for history, ignoring previous system/tool messages
        messages.extend([m for m in history if m.get("role") in ["user", "assistant"]])
    
    messages.append({"role": "user", "content": user_message})

    # 2. Agent Execution Loop (up to 3 turns: initial query, tool call/results, final answer)
    for turn in range(3):
        
        # --- LLM Generates Response (or Tool Call) ---
        response_text = llm.chat(messages)
        
        # Add the LLM's response/thought to the history for context
        messages.append({"role": "assistant", "content": response_text})

        # --- Tool Call Detection ---
        # Search for JSON tool call pattern (triple backticks or raw JSON list)
        tool_match = re.search(r"```json\s*(\[.*?\])\s*```", response_text, re.DOTALL)
        if not tool_match:
             # Fallback: check for raw JSON list (like your example output showed)
             tool_match = re.search(r"(\[\s*\{\s*\"tool_name\":.*?\}\s*\])", response_text, re.DOTALL)

        if tool_match:
            if turn == 2:
                # If the agent is still calling tools on the final turn, force a stop/answer.
                return f"Agent attempted too many tool calls. Final thought: {response_text}"

            try:
                # Extract and parse the tool call JSON (group 1 for triple backticks, group 0 for raw)
                json_string = tool_match.group(1) if tool_match.groups() else tool_match.group(0)
                tool_calls = json.loads(json_string)
                
                # --- Tool Execution ---
                tool_output_json = execute_tools(tool_calls)
                
                # --- Feed Results Back to LLM ---
                # Add the tool results as a System message for the LLM to see
                tool_result_message = {
                    "role": "system", 
                    "content": f"TOOL_RESULTS:\n{tool_output_json}\n\nBased on these results, provide the final answer to the user's question."
                }
                messages.append(tool_result_message)
                
                # The loop continues to the next turn for the LLM to generate the final answer
                continue 
            except Exception as e:
                # Error in parsing or executing tool, tell the LLM and try again/finish
                messages.append({"role": "system", "content": f"Error during tool execution/parsing: {str(e)}. Please try to generate a final, non-tool-calling response."})
                continue
        else:
            # If no tool call was found, this is the final answer.
            return response_text

    # Fallback return if the loop finishes
    return messages[-1].get("content", "An unexpected error occurred or the agent reached the maximum analysis depth.")