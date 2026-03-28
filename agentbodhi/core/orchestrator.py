import hashlib
import json
import logging
import re
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Callable, Dict, List, Optional

from bs4 import BeautifulSoup
from google import genai
from google.genai import types
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from tavily import TavilyClient

from ..agents.citation import CitationAgent
from ..agents.glossary import GlossaryAgent
from ..agents.methodology import MethodologyAgent
from ..agents.novelty import NoveltyAgent
from ..agents.related_work import RelatedWorkAgent
from ..agents.sota import SOTAAgent
from ..agents.conference import ConferenceAgent
from .models import AnalysisReport, Insight
from .utils import extract_json

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float], None]


class ResearchOrchestrator:
    CHAT_AGENT_GUIDANCE = {
        "citations": "You audit references, cite arXiv IDs when possible, and highlight missing sources.",
        "methodology": "You critique experimental design, statistical rigor, and reproducibility gaps.",
        "sota": "You compare claims to the current state of the art using hybrid search results.",
        "novelty": "You assess originality, impact, and incremental aspects versus prior work.",
        "glossary": "You explain technical terms in plain but precise language for practitioners.",
        "related": "You surface nearby research directions and links for deeper reading.",
        "conference": "You find relevant upcoming conferences and evaluate if this paper fits their Call for Papers (CFP) requirements.",
    }

    def __init__(self, gemini_key: str, tavily_key: str):
        self.client = genai.Client(api_key=gemini_key)
        self.model = "gemma-3-27b-it"#"gemini-3-flash-preview"#gemini-3.1-flash-lite-preview" #"gemini-2.5-flash-lite"
        self.tavily = TavilyClient(api_key=tavily_key)
        
        self.citation_agent = CitationAgent(self.client, self.model)
        self.methodology_agent = MethodologyAgent(self.client, self.model)
        self.sota_agent = SOTAAgent(self.client, self.model, self.tavily)
        self.novelty_agent = NoveltyAgent(self.client, self.model, self.tavily)
        self.glossary_agent = GlossaryAgent(self.client, self.model, self.tavily)
        self.related_work_agent = RelatedWorkAgent(self.client, self.model, self.tavily)
        self.conference_agent = ConferenceAgent(self.client, self.model, self.tavily)

        self._contexts = {}

    def analyze_paper(
        self,
        pdf_file,
        session_id: str,
        progress_callback: Optional[ProgressCallback] = None,
        max_citations: int = 10,
        glossary_terms: int = 12,
    ) -> AnalysisReport:
        if progress_callback:
            progress_callback("Reading PDF...", 0.1)

        pdf_bytes = pdf_file.getvalue()

        if progress_callback:
            progress_callback("Extracting paper content & generating summary...", 0.2)

        summary_resp = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                "Provide a comprehensive 400-word technical summary of this paper including: research question, methodology, key findings, and contributions."
            ]
        )

        full_text_resp = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                "Extract all text from this paper including abstract, methodology, results, and discussion sections."
            ]
        )

        paper_summary = summary_resp.text
        paper_text = full_text_resp.text

        if progress_callback:
            progress_callback("Running specialized agents (Hybrid Search)...", 0.3)

        results: Dict[str, object] = {}
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = {
                "citations": executor.submit(self.citation_agent.execute, paper_text, max_citations),
                "weaknesses": executor.submit(self.methodology_agent.execute, paper_text),
                "sota": executor.submit(self.sota_agent.execute, paper_summary, paper_text[:5000]),
                "novelty": executor.submit(self.novelty_agent.execute, paper_summary, paper_text[:8000]),
                "glossary": executor.submit(self.glossary_agent.execute, paper_text, glossary_terms),
                "related": executor.submit(self.related_work_agent.execute, paper_summary, 5)
            }

            completed = 0
            total = len(futures)
            for key, future in futures.items():
                try:
                    results[key] = future.result(timeout=60)
                except Exception as exc:
                    logger.error("Agent %s failed: %s", key, exc)
                    results[key] = [] if key != "sota" else {}
                finally:
                    completed += 1
                    if progress_callback:
                        progress = 0.3 + 0.6 * (completed / total)
                        progress_callback(f"Completed {key} analysis...", progress)

        if progress_callback:
            progress_callback("Calculating reproducibility metrics...", 0.9)

        reproducibility_score = self._calculate_reproducibility(paper_text)
        insights = self._generate_insights(results, paper_summary)

        novelty_details = results.get("novelty", {})
        novelty_score = novelty_details.get("novelty_score", 0) if isinstance(novelty_details, dict) else 0

        report = AnalysisReport(
            paper_id=hashlib.md5(pdf_bytes).hexdigest()[:8],
            timestamp=datetime.now().isoformat(),
            summary=paper_summary,
            citations=results.get("citations", []),
            weaknesses=results.get("weaknesses", []),
            sota_analysis=results.get("sota", {}),
            glossary=results.get("glossary", {}),
            novelty_score=novelty_score,
            reproducibility_score=reproducibility_score,
            insights=insights,
            related_work=results.get("related", []),
            novelty_details=novelty_details if isinstance(novelty_details, dict) else {},
        )

        if progress_callback:
            progress_callback("Analysis complete!", 1.0)
            
        # Also store the context so the chat interface can be used right after
        self._contexts[session_id] = {
            "summary": paper_summary,
            "full_text": paper_text,
            "pdf_bytes": pdf_bytes
        }

        return report

    def _calculate_reproducibility(self, paper_text: str) -> float:
        try:
            prompt = f"""Assess the reproducibility of this research paper.
Check for:
1. Code/data availability
2. Hyperparameter specifications
3. Hardware/environment details
4. Statistical significance reporting
5. Clear methodology description

Return a score from 0-10 as JSON:
{{"score": 7.5, "reasoning": "..."}}

Paper:
{paper_text[:8000]}

Return ONLY the JSON."""

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            json_match = re.search(r"\{.*?\}", response.text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return data.get("score", 5.0)
            return 5.0
        except Exception as exc:
            logger.error("Reproducibility calculation error: %s", exc)
            return 5.0

    def _generate_insights(self, results: Dict[str, object], summary: str) -> List[Insight]:
        insights: List[Insight] = []

        citations = results.get("citations", [])
        verified = sum(1 for c in citations if getattr(c, "status", None) == "Verified")
        if citations:
            insights.append(Insight(
                type="Citations",
                content=f"{verified}/{len(citations)} citations verified on ArXiv",
                confidence=0.9,
                sources=["ArXiv"],
            ))

        sota = results.get("sota", {})
        analysis = sota.get("analysis") if isinstance(sota, dict) else None
        if analysis:
            insights.append(Insight(
                type="State-of-the-Art",
                content=f"{('Current SOTA' if analysis.get('is_sota') else 'Not current SOTA')} - {analysis.get('recommendation', 'N/A')}",
                confidence=analysis.get("confidence", 0.5),
                sources=sota.get("sources", []),
            ))

        novelty = results.get("novelty", {})
        nouve_score = novelty.get("novelty_score") if isinstance(novelty, dict) else None
        if nouve_score is not None:
            insights.append(Insight(
                type="Novelty",
                content=f"Novelty score: {nouve_score}/10 - {novelty.get('impact_potential', 'N/A')} impact potential",
                confidence=0.8,
                sources=["Comparative analysis"],
            ))

        weaknesses = results.get("weaknesses", [])
        if weaknesses:
            critical = sum(1 for w in weaknesses if getattr(w, "severity", "") == "Critical")
            insights.append(Insight(
                type="Methodology",
                content=f"Found {len(weaknesses)} methodological issues ({critical} critical)",
                confidence=0.85,
                sources=["Peer review analysis"],
            ))

        return insights

    def get_context_snapshot(self, session_id: str) -> Dict[str, object]:
        return dict(self._contexts.get(session_id, {}))

    def load_pdf_context(self, pdf_bytes: bytes, session_id: str) -> None:
        """Loads a PDF and extracts text/summary without running the full analysis pipeline."""
        summary_resp = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                "Provide a comprehensive 400-word technical summary of this paper including: research question, methodology, key findings, and contributions."
            ]
        )

        full_text_resp = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                "Extract all text from this paper including abstract, methodology, results, and discussion sections."
            ]
        )

        self._contexts[session_id] = {
            "summary": summary_resp.text,
            "full_text": full_text_resp.text,
            "pdf_bytes": pdf_bytes
        }

    def chat_with_agents(self, session_id: str, agent_slugs: List[str], instruction: str) -> Dict[str, str]:
        context = self.get_context_snapshot(session_id)
        if not context.get("full_text"):
            raise ValueError("No paper context available for this session. Upload a PDF first.")

        selected_roles = [self.CHAT_AGENT_GUIDANCE.get(s, "") for s in agent_slugs]
        
        # Step 1: Generate a Plan using LLM 
        plan_prompt = f"""You are the Master Orchestrator Agent. 
You need to fulfill this user instruction: "{instruction}"
Relevant Paper Summary: {context['summary'][:1000]}
You have the following expert roles available: {agent_slugs}

Decide on 1 to 3 search queries needed to gather external information required to fulfill the user instruction.
Do not search for things that the dedicated sub-agents would check (e.g. SOTA, conferences).
If no general search is needed, return an empty list for queries.

Return ONLY valid JSON in this format:
{{
    "plan_steps": ["step 1 description", "step 2 description"],
    "search_queries": ["query 1", "query 2"]
}}
"""
        plan_resp = self.client.models.generate_content(model=self.model, contents=plan_prompt).text
        plan_json = {}
        try:
            json_str = extract_json(plan_resp)
            plan_json = json.loads(json_str)
        except Exception:
            plan_json = {"plan_steps": ["Analyze paper text"], "search_queries": []}

        logs = ["### 🕵️‍♂️ Agent Live Log"]
        logs.append("📝 **Agent Action:** Formulating execution plan...")
        for step in plan_json.get("plan_steps", []):
            logs.append(f"  - {step}")

        # Step 2: Delegate to specialized AI Agents
        logs.append("⚙️ **Agent Action:** Delegating to invoked sub-agents...")
        agent_results = {}
        paper_text = str(context["full_text"])
        paper_summary = str(context["summary"])
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future_to_agent = {}
            if "citations" in agent_slugs:
                future_to_agent[executor.submit(self.citation_agent.execute, paper_text, 10)] = "Citations"
            if "methodology" in agent_slugs:
                future_to_agent[executor.submit(self.methodology_agent.execute, paper_text)] = "Methodology"
            if "sota" in agent_slugs:
                future_to_agent[executor.submit(self.sota_agent.execute, paper_summary, paper_text[:5000])] = "SOTA"
            if "novelty" in agent_slugs:
                future_to_agent[executor.submit(self.novelty_agent.execute, paper_summary, paper_text[:8000])] = "Novelty"
            if "glossary" in agent_slugs:
                future_to_agent[executor.submit(self.glossary_agent.execute, paper_text, 10)] = "Glossary"
            if "related" in agent_slugs:
                future_to_agent[executor.submit(self.related_work_agent.execute, paper_summary, 5)] = "Related Work"
            if "conference" in agent_slugs:
                future_to_agent[executor.submit(self.conference_agent.execute, paper_summary)] = "Conference Matchmaker"

            for future in as_completed(future_to_agent):
                agent_name = future_to_agent[future]
                try:
                    result = future.result(timeout=60)
                    agent_results[agent_name] = result
                    logs.append(f"✅ **{agent_name} Agent**: Execution completed successfully.")
                except Exception as exc:
                    logger.error("Agent %s failed: %s", agent_name, exc)
                    agent_results[agent_name] = {"error": str(exc)}
                    logs.append(f"❌ **{agent_name} Agent**: Execution failed.")

        # Step 3: Execute Search tools (Selenium/DuckDuckGo)
        search_results = ""
        for query in plan_json.get("search_queries", []):
            logs.append(f"🌐 **Agent Action:** Searching Web via Selenium for: `{query}`...")
            res = self._selenium_google_search(query)
            search_results += f"Search Query: {query}\nResults:\n{res}\n\n"
            logs.append(f"✅ **Agent Action:** Retrieved search results for `{query}`")

        logs.append("🧠 **Agent Action:** Synthesizing final unified collaborative report grounded in agent data...")

        # Convert agent results to JSON text
        def serialize_results(obj):
            if hasattr(obj, "model_dump"):
                return obj.model_dump()
            elif hasattr(obj, "__dict__"):
                return obj.__dict__
            return str(obj)

        agent_results_json = json.dumps(agent_results, indent=2, default=serialize_results)

        # Step 4: Final Synthesis
        synthesis_prompt = f"""You are the unified Master Agent synthesizing a response based on multiple expert perspectives.
You are embodying these roles: {selected_roles}

User Instruction: {instruction}

Paper Summary:
{context['summary']}

--- The Core Sub-Agent Findings (crucial strictly-gathered domain facts) ---
{agent_results_json}

--- External Web Search Results (general context) ---
{search_results if search_results else "No external search was performed."}

Synthesize a single, highly cohesive, comprehensive response combining the requested sub-agent data and factual findings directly into the report. Ground the answer on the concrete findings, citing papers, metrics, or glossary terms discovered. 
Do not separate arbitrarily by agent name, write one unified professional report addressing everything requested by the user, but heavily incorporate the sub-agent data naturally. Markdown format.
"""
        final_resp = self.client.models.generate_content(model=self.model, contents=synthesis_prompt).text.strip()
        
        formatted_logs = "\n".join(logs)
        return {"Synthesized Report": f"{formatted_logs}\n\n---\n\n{final_resp}"}

    def chat_with_agent(self, session_id: str, agent_slug: str, instruction: str, usage_hint: str) -> str:
        context = self.get_context_snapshot(session_id)
        if not context.get("full_text"):
            raise ValueError("No paper context available for this session. Upload a PDF first.")

        primer = self.CHAT_AGENT_GUIDANCE.get(agent_slug, "You are a helpful research agent.")
        prompt = f"{primer}\n\nPaper summary:\n{context['summary']}\n\nRelevant paper excerpt:\n{str(context['full_text'])[:8000]}\n\nResearcher instruction:\n{instruction}\n\n{usage_hint}\n\nRespond in Markdown."

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text.strip()

    def _selenium_google_search(self, query: str, num_results: int = 3) -> str:
        try:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            driver = webdriver.Chrome(options=options)
            
            # Using DuckDuckGo html as it's easier to parse without JS than Google
            url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            driver.get(url)
            html = driver.page_source
            
            soup = BeautifulSoup(html, 'html.parser')
            results = []
            for a in soup.find_all('a', class_='result__snippet', limit=num_results):
                results.append(a.text.strip())
                
            if not results:
                return "No obvious results found or blocked by search engine."
                
            return "\n".join([f"- {r}" for r in results])
        except Exception as e:
            return f"Search error: {str(e)}"
        finally:
            if 'driver' in locals():
                try:
                    driver.quit()
                except Exception:
                    pass
