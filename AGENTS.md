# Agent Bodhi Codebase Guide for AI Agents

Welcome, AI Agent! This file contains the essential context you need to be immediately productive in the Agent Bodhi codebase.

## 🏗️ Big Picture Architecture

- **Backend & Frontend Separation**: The app relies on FastAPI (`app.py`) for API routing and uses a lightweight vanilla HTML/JS single-page frontend served from `static/index.html`. It explicitly avoids Streamlit.
- **Plan-and-Execute ReAct Orchestrator**: The crux of the AI logic sits in `agentbodhi/core/orchestrator.py` (`ResearchOrchestrator`). The unified chat (`chat_with_agents`):
  1. Asks the LLM to generate an execution plan & search queries in JSON.
  2. Runs actual web searches via headless Selenium + DuckDuckGo.
  3. Synthesizes a unified final report adopting multiple chosen agent personas.
- **Sub-Agents**: Specific tasks (SOTA, Novelty, Methodology) are handled by specific agent classes in `agentbodhi/agents/`. All inherit from `ResearchAgent` (`agentbodhi/agents/base.py`), which abstracts LLM caching (`_cache_key`, `_get_cached`).

## 🛠️ Key Conventions & Project Patterns

- **LLM JSON Parsing**: ALWAYS use the `extract_json` utility from `agentbodhi.core.utils` when asking the LLM for JSON outputs. It safely strips markdown backticks (` ```json `), preventing `json.decoder.JSONDecodeError`.
- **Context Window Management**: To save tokens, avoid passing the original PDF to every agent. Use the 400-word `paper_summary` or crop strings like `paper_text[:8000]` (exemplified in `orchestrator.py`).
- **Adding New Chat Agents**: To add a theoretical new agent persona:
  1. Create its tool implementation in `agentbodhi/agents/`.
  2. Expose the agent's slug/description in `app.py` under the `@app.get("/api/agents")` endpoint.
  3. **Critical**: Register its specific prompt guidance in the `CHAT_AGENT_GUIDANCE` dictionary at the top of `ResearchOrchestrator`. Code will silently fail to adopt the persona if omitted.
- **Environment config**: Do not prompt the user for `.env` creation. Keys are managed by `ConfigManager` inside `agentbodhi/configuration.py`, reading directly from `config.py`.

## 🚀 Workflows

**Starting the Application**
Avoid verbose `uvicorn` terminal commands. The `app.py` file has an `if __name__ == "__main__":` block that handles host setup and auto-reload automatically.
```powershell
python app.py
```

**Web Tooling Dependency**
Web searches inherently use Tavily (`tavily-python`) for structured data or DuckDuckGo + Selenium for fallback. Ensure background Chromium instances are cleanly killed if debugging scraper logic in `_selenium_google_search`.
