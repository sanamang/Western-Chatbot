#!/bin/bash
set -e
cd "$(dirname "$0")"

PYTHON=".venv/bin/python"
STREAMLIT=".venv/bin/streamlit"

if [ ! -f "$PYTHON" ]; then
  echo "Virtual environment not found. Run: python3 -m venv .venv && .venv/bin/pip install -r backend/requirements.txt"
  exit 1
fi

if [ ! -f ".venv/bin/streamlit" ]; then
  echo "Installing dependencies..."
  .venv/bin/pip install -r backend/requirements.txt -q
fi

echo "Starting Western University AI Chatbot..."
cd backend
../$STREAMLIT run streamlit_app.py --server.port 8501 --browser.gatherUsageStats false
