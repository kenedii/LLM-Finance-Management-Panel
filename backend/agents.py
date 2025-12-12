# agents.py
from crewai import Agent, Task, Crew
from llm_provider import LLMProvider
from tools import get_current_price, get_historical_data

def build_agents(provider: str):
    llm = LLMProvider.get(provider)

    prediction_analyst = Agent(
        role="Prediction Analyst",
        goal="Predict future performance of the asset.",
        backstory="Expert in quantitative forecasting.",
        tools=[get_current_price, get_historical_data],
        llm=llm
    )

    short_term = Agent(
        role="Short Term Market Analyst",
        goal="Analyze short-term price action.",
        backstory="Expert in technical analysis.",
        tools=[get_current_price, get_historical_data],
        llm=llm
    )

    long_term = Agent(
        role="Long Term Market Analyst",
        goal="Evaluate long-term viability.",
        tools=[get_current_price, get_historical_data],
        llm=llm
    )

    risk_analyst = Agent(
        role="Risk Analyst",
        goal="Assess risk factors.",
        tools=[get_current_price, get_historical_data],
        llm=llm
    )

    crew = Crew(
        agents=[prediction_analyst, short_term, long_term, risk_analyst]
    )

    return crew
