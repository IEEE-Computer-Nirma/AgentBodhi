# Agent Bodhi Codebase Guide for AI Agents

Welcome, AI Agent! This file contains the essential context and critical architecture constraints you need to be immediately productive in the Agent Bodhi codebase.

## 🏗️ Big Picture Architecture

- **Backend & Frontend Separation**: The application deliberately avoids heavier UI frameworks like Streamlit. It strictly relies on FastAPI (`app.py`) for API routing, serving a lightweight vanilla HTML/JS single-page frontend from `static/index.html`. 
- **Plan-and-Execute Orchestrator**: The crux of the AI logic sits within `agentbodhi/core/orchestrator.py` (`ResearchOrchestrator`). Both unified chats (`chat_with_agents`) and dashboard workflows (`analyze_paper`) coordinate operations here.
- **Sequential Execution Queues**: Orchestration heavily relies on `ThreadPoolExecutor(max_workers=1)`. The agents are executed serially to explicitly avoid triggering OS-level socket exhaustion (e.g., `WinError 10053`/`WinError 10060`/`Errno 11001`) and to circumvent stringent rate limits strictly enforced by external providers like ArXiv and Google Gemini.
- **Sub-Agents**: Specific tasks (SOTA, Novelty, Methodology) are segregated into dedicated agent files in the `agentbodhi/agents/` directory. All sub-agents inherit from `ResearchAgent` (`agentbodhi/agents/base.py`), which abstracts LLM usage and result caching (`_cache_key`, `_get_cached`).

## 🛠️ Key Conventions & Data Flow

- **LLM JSON Parsing**: ALWAYS use the `extract_json` utility from `agentbodhi/core/utils.py` when attempting to unpack JSON output from the LLM. It safely strips markdown backticks (` ```json `) to prevent `json.decoder.JSONDecodeError` exceptions.
- **Context Window Management**: To conserve tokens and reduce overhead, drastically avoid passing the entirety of a raw PDF payload to every agent. Agents should query the `paper_summary` (usually ~400 words) or cropped sections like `paper_text[:8000]` instead.
- **ArXiv API Interactions**: Never invoke massive concurrent requests against the ArXiv API. All ArXiv searches are mandated to use proper client rate-limiting parameters:
  ```python
  client = arxiv.Client(page_size=10, delay_seconds=3, num_retries=5)
  results = client.results(search)
  ```
- **Serialization Patterns**: Data models (`agentbodhi/core/models.py`) are strictly structured using Python `dataclasses`. Because they are native dataclasses (and not Pydantic BaseModels), ALWAYS serialize them via `dataclasses.asdict(report)` before transmitting them out of HTTP routes.
- **Environment & Config Management**: Do NOT prompt the user for `.env` creation. Variables and API keys are explicitly managed by `ConfigManager` inside `agentbodhi/configuration.py`, reading directly from flat strings housed in `config.py`.

## 🎨 UI & Markdown Formatting
When building strings destined for the HTML frontend, generate clean markdown templates entirely left-aligned (no 4-space indentation) to ensure that the client-side `marked.js` library properly registers them as headers, lists, and bold text blocks rather than rendering them incorrectly as pre-formatted text elements.

## 🚀 Adding New Chat Agents

To seamlessly add a theoretical new agent persona to the overall system:
1. **Tooling**: Create its logic file inside `agentbodhi/agents/` and ensure it wraps `ResearchAgent`.
2. **Registration**: Expose the new agent's slug, label, and description inside `app.py` under the `@app.get("/api/agents")` router endpoint. This forces it to render physically inside the frontend's left sidebar.
3. **Execution Logic**: Thread its execution hook manually into `agentbodhi/core/orchestrator.py` dynamically triggering it if its slug exists in the user's requested array query.
4. **Agent Personas**: Register the agent's explicit guidance strings in the `CHAT_AGENT_GUIDANCE` dictionary at the top of the orchestrator file. Code will quietly default and fail to adopt the correct behavioral persona without this map injection!
