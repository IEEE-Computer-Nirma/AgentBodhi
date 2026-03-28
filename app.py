from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List
import uuid

from agentbodhi.configuration import ConfigManager
from agentbodhi.core.orchestrator import ResearchOrchestrator

app = FastAPI(title="Agent Bodhi Chat API")

# Serve the static frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

gemini_key, tavily_key = ConfigManager.get_api_keys()

try:
    orchestrator = ResearchOrchestrator(gemini_key or "", tavily_key or "")
except Exception as e:
    print(f"Warning: Could not initialize orchestrator properly. {e}")
    orchestrator = None

class ChatRequest(BaseModel):
    message: str
    agents: List[str]
    session_id: str

@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.get("/api/agents")
def get_agents():
    return [
        {"slug": "methodology", "label": "Methodology Reviewer", "description": "Interrogate the experimental design and statistical rigor."},
        {"slug": "citations", "label": "Citation Auditor", "description": "Verify references and spot missing prior work."},
        {"slug": "sota", "label": "SOTA Scout", "description": "Compare claims with the latest literature."},
        {"slug": "novelty", "label": "Novelty Analyst", "description": "Probe originality and impact potential."},
        {"slug": "glossary", "label": "Glossary Curator", "description": "Break down dense terminology."},
        {"slug": "related", "label": "Related Work Scout", "description": "Surface adjacent papers worth reading next."},
        {"slug": "conference", "label": "Conference Matchmaker", "description": "Search upcoming conferences and evaluate paper fit (Live Web Search)."},
    ]

@app.post("/api/upload")
async def upload_paper(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDFs are supported.")
    
    content = await file.read()
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Backend orchestrator is not configured with valid keys.")
    
    try:
        session_id = str(uuid.uuid4())
        orchestrator.load_pdf_context(content, session_id)
        return JSONResponse(content={"status": "success", "filename": file.filename, "session_id": session_id})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@app.post("/api/chat")
async def chat_with_agents(request: ChatRequest):
    if not request.agents:
        raise HTTPException(status_code=400, detail="Please select at least one agent.")
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Backend orchestrator is not configured with valid keys.")
    if not request.session_id:
        raise HTTPException(status_code=400, detail="Missing session_id.")
    
    try:
        results = orchestrator.chat_with_agents(request.session_id, request.agents, request.message)
        return JSONResponse(content={"responses": results})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze")
async def analyze_full_paper(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDFs are supported.")
    
    content = await file.read()
    if not orchestrator:
        raise HTTPException(status_code=500, detail="Backend orchestrator is not configured with valid keys.")
    
    try:
        # We need a file-like object for orchestrator.analyze_paper
        import io
        import dataclasses
        pdf_file = io.BytesIO(content)
        session_id = str(uuid.uuid4())
        report = orchestrator.analyze_paper(pdf_file, session_id)
        
        # Serialize report (it's a dataclass model)
        response_data = dataclasses.asdict(report)
        response_data["session_id"] = session_id
        return JSONResponse(content=response_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze PDF: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)
