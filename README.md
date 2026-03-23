# Agent Bodhi

A Streamlit-powered multi-agent research analysis tool that orchestrates specialized agents for citation validation, methodology critique, state-of-the-art comparison, novelty assessment, glossary building, and related work discovery. The refactor splits responsibilities into packages so `main.py` simply wires UI, configuration, and orchestrator logic.

## Setup

1. Create/activate a Python virtual environment (Python 3.11+ recommended).
   ```powershell
   python -m venv .venv; .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies (adjust versions as needed).
   ```powershell
   pip install streamlit google-generativeai tavily arxiv
   ```
3. Populate `config.py` with valid keys for `GEMINI_API_KEY` and `TAVILY_API_KEY`.

## Project Layout

- `main.py` – Streamlit entrypoint that configures the UI, validates keys, and triggers the orchestrator.
- `config.py` – Local module where you store API secrets (excluded from VCS).
- `agentbodhi/configuration.py` – Helper that loads/validates keys for UI reuse.
- `agentbodhi/core/` – Shared dataclasses (`models.py`), utilities (`utils.py`), and the orchestrator that coordinates the agents.
- `agentbodhi/agents/` – Each agent lives in its own module (citation, methodology, SOTA, novelty, glossary, related work) plus a base class for caching/extraction helpers.
- `agentbodhi/ui/renderers.py` – Sidebar controls and section renderers keep Streamlit layout logic separate from orchestration.

## Running Locally

With the virtual environment activated and dependencies installed:
```powershell
streamlit run main.py
```
Then open the provided localhost URL, enter API keys in the sidebar, upload a PDF, and click **Analyze Paper**.

## Testing

Quickly verify that the refactored modules compile:
```powershell
python -m py_compile main.py agentbodhi/core/orchestrator.py agentbodhi/ui/renderers.py
```

## Notes

- Sensitive keys remain in `config.py`; keep it out of git history in production use.
- The orchestrator runs agents concurrently using `ThreadPoolExecutor` and emits progress updates via callbacks stored in `st.session_state`.
- Extend or swap agents under `agentbodhi/agents/` without touching the UI by keeping shared data contracts in `agentbodhi/core/models.py`.

## Recommended Next Steps
1. Add automated tests for `ResearchOrchestrator` with mocked clients/agents and validate insight generation.
2. Consider a `pyproject.toml` or `requirements.txt` to pin dependencies.
3. Harden error handling for incomplete agent responses (e.g., missing JSON payloads).
