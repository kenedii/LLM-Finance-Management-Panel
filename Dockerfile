# syntax=docker/dockerfile:1

# Use an official lightweight Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies required by some Python packages
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy dependency manifest first for caching
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY backend ./backend
COPY frontend ./frontend

# Expose API and Streamlit ports
EXPOSE 8000
EXPOSE 8501

# Default command: run backend API and frontend Streamlit in the same container
# Streamlit bound to 0.0.0.0:8501, Uvicorn on 0.0.0.0:8000
CMD bash -lc "uvicorn backend.main:app --host 0.0.0.0 --port 8000 & streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0"
