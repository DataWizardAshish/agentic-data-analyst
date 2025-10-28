# Agentic Data Analyst

## Overview
Single-page Streamlit app that runs a sequence of DSPy agents (schema, profile, quality, ML advisor, deployment, business) and can generate a PRD.

## Requirements
- Python 3.11.13 (your project env created via `uv`)
- `uv` manages the virtual environment (or use pip/venv)

## Quick start (using `uv`)
1. Ensure you activated the `uv` environment (or `python` from `uv`).
2. Install dependencies:
   ```bash
   # If using uv
   uv sync
   # or fallback: pip install -r requirements.txt
3. Copy .env.example to .env and add your OPENAI_API_KEY.
4. Run the App :

streamlit run app.py

5. Developer targets (Makefile)

make install — install dependencies

make run — run Streamlit locally

make test — run unit tests

6. Tests
Run
pytest -q
