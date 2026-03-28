# Agent Bodhi: AI Research Co-Pilot

Agent Bodhi is an advanced research co-pilot application powered by a FastAPI backend and a clean, responsive plain HTML/JS frontend. It allows users to upload academic research papers (PDFs) and interact with an ensemble of AI "agents" that each specialize in different aspects of paper analysis, such as validating citations, checking methodology, tracking State-of-the-Art (SOTA) developments, and scoring novelty.

## 🌟 Core Features

- **Specialized AI Agents**: Employs multiple collaborative agents (Citations, Methodology, SOTA, Novelty, Glossary, Related Work, and Conference Matchmaker) that dissect and review full-text PDFs.
- **Full Analytics Dashboard**: Generates comprehensive "Dashboard Orchestrator" reports, assessing the paper's Novelty, Reproducibility Score, high-level summaries, and providing key insights into the research context.
- **Robust API Integrations**: Grounded in real-world data utilizing Google Gemini for LLM processing, Tavily API for broad web searches, and ArXiv's official API for rigorous academic literature checks.
- **FastAPI Engine**: Powered directly by Uvicorn within `app.py`, ensuring a very fast backend replacing heavier frameworks like Streamlit.
- **Vanilla HTML/CSS/JS UI**: Provides an inherently responsive native chat experience that utilizes `marked.js` to flawlessly render markdown lists, emojis, and emphasis inside conversation bubbles.
- **Stable Execution Queue**: Employs serialized ThreadPool queues (`max_workers=1`) during intensive orchestration tasks to prevent local OS socket exhaustion (`WinError 10053/10060`) and third-party rate limits.
- **Graceful Error Handling**: Complete with dismissible UI notifications, robust fallback logic, and clear frontend error reporting.

## 🧠 Meet The Agents

The system leverages a specialized pool of personas:
*   **Methodology Reviewer**: Interrogates the experimental design and statistical rigor.
*   **Citation Auditor**: Verifies references and spots missing or improperly cited prior work using external ArXiv checks.
*   **SOTA Scout**: Compares the paper's claims with the absolute latest developments in the field.
*   **Novelty Analyst**: Probes the originality and potential overall impact of the research.
*   **Glossary Curator**: Extracts and breaks down dense, domain-specific terminology for faster comprehension.
*   **Related Work Scout**: Surfaces adjacent and highly relevant academic papers worth reading next.
*   **Conference Matchmaker**: Searches upcoming academic conferences and evaluates the paper's fit for submission.

## 🚀 Quick Start Guide

### 1. Environment Initialization
Ensure you have a Python 3.11+ environment ready.
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Dependency Installation
Install the required packages.
```powershell
pip install fastapi uvicorn python-multipart google-generativeai tavily-python arxiv selenium beautifulsoup4 webdriver-manager
```

### 3. Configuration
Agent Bodhi uses a `config.py` module to securely manage API keys. Create or update `config.py` in the root directory to include your necessary credentials:
```python
GEMINI_API_KEY = "your_google_gemini_key_here"
TAVILY_API_KEY = "your_tavily_api_key_here"
```

### 4. Running the Application
Spin up the server seamlessly by running the main Python file.
```powershell
python app.py
```

### 5. Start Reviewing
Open your web browser and navigate to `http://127.0.0.1:5000`. You can upload a PDF using the side panel, select which expert agents you'd like to consult, and begin talking or trigger a Full Analytics Dashboard run.

## 📂 Architecture & File Layout

- **`app.py`**: The FastAPI application entry point, containing API routes (`/api/upload`, `/api/chat`, `/api/analyze`, `/api/agents`) and backend server scaffolding.
- **`static/index.html`**: A tailored single-page HTML chat interface with bundled CSS/JS and markdown rendering capabilities.
- **`config.py`**: Configuration dictionary tracking critical secrets.
- **`agentbodhi/`**: The core package directory containing:
  - `core/orchestrator.py`: Orchestrates agent invocations and aggregates structured execution plans.
  - `core/models.py`: Python dataclasses representing complex analytical structures (Insights, Weaknesses, Dashboard Reports).
  - `agents/`: The dedicated directory housing the logic, specialized prompts, and toolsets for each individual agent (e.g., `sota.py`, `citation.py`).
