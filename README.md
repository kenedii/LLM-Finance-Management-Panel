# LLM-Finance-Management-Panel
Frontend to setup a portfolio of owned assets, track the price of each asset, and use LLMs to research assets and provide portfolio advice.

Instructions to run:

1. pip install -r requirements.txt
2. uvicorn backend.main:app --port 8000
3. streamlit run frontend/streamlit_app.py
