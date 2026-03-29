# Future Plan for Agent Bodhi

This document outlines the strategic roadmap for expanding Agent Bodhi's capabilities, focusing heavily on new AI agents, advanced analytical workflows, and feature expansions rather than just technical maintenance.

## 🤖 Proposed New Sub-Agents

The core strength of Agent Bodhi is its specialized, mixture-of-experts approach to reviewing research papers. The following agents will be added to provide a more holistic analysis:

### 1. **Reproducibility Agent** (`reproducibility.py`)
- **Role**: Auditor for empirical claims.
- **Tasks**: Scans the paper for necessary reproducibility artifacts: linked GitHub repositories, dataset accessibility/links, specific hyperparameter values, hardware configurations, and exact random seed mentions.
- **Output**: A "Reproducibility Scorecard" indicating how easily an independent researcher could replicate the study.

### 2. **Math & Theorem Agent** (`math_validator.py`)
- **Role**: Mathematical consistency checker.
- **Tasks**: Extracts core mathematical definitions, theorems, and proofs. Checks for notation consistency across the paper (e.g., "Is variable $\alpha$ redefined later?"). 
- **Output**: A standalone summary of the core equations, making heavy theoretical papers more accessible.

### 3. **Dataset & Bias Agent** (`dataset.py`)
- **Role**: Data evaluator.
- **Tasks**: Identifies the datasets utilized, interrogates their potential biases, checks licensing constraints, and flags if the data is synthetic or organic.
- **Output**: A section detailing the demographic or domain limitations of the data evaluated in the paper.

### 4. **Real-World Impact & Ethics Agent** (`impact.py`)
- **Role**: Industry and ethical analyst.
- **Tasks**: Projects the findings into real-world applications (e.g., healthcare, finance, consumer tech). Also performs an ethical review, identifying potential dual-use risks or negative societal impacts of the covered technology.

### 5. **Peer-Review / Critique Agent** (`reviewer.py`)
- **Role**: The "Devil's Advocate".
- **Tasks**: Actively looks for weak points in the paper such as unbacked claims, poor baseline selections, or conflated variables. It simulates a harsh but fair "Reviewer 2".

---

## 🚀 Advanced Workflows & Features

### 1. **Multi-Agent Debate / Synthesis**
Instead of agents just appending their output sequentially, introduce a **Debate Mode**. For example, the `novelty` agent might claim the paper is highly unique, while the `related_work` agent might find three identical papers. An overarching **Synthesis Agent** would resolve these conflicts and provide a nuanced final verdict.

### 2. **Multi-Paper Knowledge Graph**
Transition Agent Bodhi from a "Single Paper Analyzer" to a "Literature Review Engine".
- Users can upload a batch of 5-10 PDFs.
- Agents map out connections between the papers (e.g., Paper A improves on the dataset from Paper B).
- Output is an interactive, visual Knowledge Graph of citations and concept evolution.

### 3. **Dynamic / Custom Agents**
Provide a UI feature allowing users to spin up a custom agent on the fly. 
- **How it works**: The user inputs a Name (e.g., "Medical Expert") and a custom prompt instruction. The Orchestrator temporarily registers this agent in `app.py` for the duration of the session, executing it over the text using the standard base agent pipeline.

### 4. **Direct "Agent Chat" Mode**
Currently, queries in the left-hand chat are routed by the Orchestrator. A new feature would allow the user to "@mention" a specific agent in the chat to ask iterative questions directly to their specific context window (e.g., `@MathAgent can you explain Equation 3 in simpler terms?`).

### 5. **Rich Export Module**
Once the agents have finished their comprehensive dashboard analysis, users should be able to click **"Export as Review Report"**. 
- Generates a beautifully formatted LaTeX/PDF or Markdown document compiling all agent findings into a structured, formal AI Peer Review.

### 6. **Visual Citation Tree**
Enhance the UI by rendering the output of the `citation` and `related_work` agents into an interactive visual graph using libraries like D3.js or Cytoscape, mapping out how the current paper sits within the broader scientific tree.
