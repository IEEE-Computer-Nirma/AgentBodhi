# Agent Bodhi API Chat

A FastAPI + plain HTML/JS research co-pilot! Upload a PDF inside a chat GUI and select multiple parallel agents (Citations, Methodology, Novelty, etc.) to query them simultaneously.

## Core Features

- **FastAPI Engine**: Powered by Uvicorn, replacing the cumbersome Streamlit flow.
- **Unified ReAct Orchestrator Plan-and-Execute Framework**: Select multiple expert agent personas. Instead of disconnected dummy outputs, the central ReAct Orchestrator creates an execution plan, performs necessary external searches (such as Glossary definitions and Conference requirements) dynamically using Selenium + DuckDuckGo, and synthesizes one coherent, collaborative response.
- **True Agentic AI Grounding**: Implemented real tool usage. No more "marionette" prompts masquerading as actions—Agent Bodhi uses headless Chromium and Beautiful Soup to fetch real web results to ensure accurate conference submissions with actual CFP verification.
- **HTML/CSS/JS Frontend**: Clean, responsive layout that feels like a native chat application.
- **No API Hurdles in UI**: Configuration implicitly loaded from `config.py`.

## Quick Start

1. Initialize a Python 3.11+ environment.
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies.
   ```powershell
   pip install fastapi uvicorn python-multipart google-generativeai tavily-python arxiv selenium beautifulsoup4 webdriver-manager
   ```
3. Ensure `config.py` has valid API credentials for `GEMINI_API_KEY` and `TAVILY_API_KEY`.
4. Run the server.
   ```powershell
   python app.py
   ```
5. Open `http://127.0.0.1:8000` in your web browser. Upload a paper using the button at the top header, choose your agents from the sidebar, and start chatting.

## File Layout

- `app.py` – FastAPI routes and application assembly.
- `static/index.html` – Lightweight HTML chat interface (with bundled minimal CSS/JS).
- `config.py` – Local secrets module.
- `agentbodhi/` – Backend orchestration logic and agent class definitions.
