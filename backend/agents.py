# agents.py
"""Builds the analysis crew. Safe to import on Windows.

We avoid importing `crewai` at module import time to prevent failures
on Windows where `signal.SIGHUP` may be missing. Instead, we patch
the `signal` module and import `crewai` lazily inside `build_agents`.
"""

import os
import signal

# Flexible imports: work whether 'backend' is a package or current dir
try:
    from backend.llm_provider import LLMProvider  # when running from repo root
    from backend.tools import get_current_price, get_historical_data
except Exception:
    from llm_provider import LLMProvider  # fallback when running inside backend folder
    from tools import get_current_price, get_historical_data


def build_agents(provider: str):
    # Patch missing SIGHUP on Windows before importing crewai
    if os.name == "nt" and not hasattr(signal, "SIGHUP"):
        signal.SIGHUP = 1  # dummy value

    # Lazy import to avoid startup crashes on Windows
    from crewai import Agent, Task, Crew

    llm = LLMProvider.get(provider)

    prediction_analyst = Agent(
        role="Prediction Analyst",
        goal="Predict future performance of the asset.",
        backstory="Expert in quantitative forecasting.",
        tools=[get_current_price, get_historical_data],
        llm=llm,
    )

    short_term = Agent(
        role="Short Term Market Analyst",
        goal="Analyze short-term price action.",
        backstory="Expert in technical analysis.",
        tools=[get_current_price, get_historical_data],
        llm=llm,
    )

    long_term = Agent(
        role="Long Term Market Analyst",
        goal="Evaluate long-term viability.",
        tools=[get_current_price, get_historical_data],
        llm=llm,
    )

    risk_analyst = Agent(
        role="Risk Analyst",
        goal="Assess risk factors.",
        tools=[get_current_price, get_historical_data],
        llm=llm,
    )

    crew = Crew(agents=[prediction_analyst, short_term, long_term, risk_analyst])

    return crew