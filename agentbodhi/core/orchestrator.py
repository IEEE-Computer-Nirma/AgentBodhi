import hashlib
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Callable, Dict, List, Optional

from google import genai
from google.genai import types
from tavily import TavilyClient

from ..agents.citation import CitationAgent
from ..agents.glossary import GlossaryAgent
from ..agents.methodology import MethodologyAgent
from ..agents.novelty import NoveltyAgent
from ..agents.related_work import RelatedWorkAgent
from ..agents.sota import SOTAAgent
from .models import AnalysisReport, Insight

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
    }

    def __init__(self, gemini_key: str, tavily_key: str):
        self.client = genai.Client(api_key=gemini_key)
        self.model = "gemini-3.1-flash-lite-preview" #"gemini-2.5-flash-lite" "gemma-3-27b-it"#
        self.tavily = TavilyClient(api_key=tavily_key)

        self.citation_agent = CitationAgent(self.client, self.model)
        self.methodology_agent = MethodologyAgent(self.client, self.model)
        self.sota_agent = SOTAAgent(self.client, self.model, self.tavily)
        self.novelty_agent = NoveltyAgent(self.client, self.model, self.tavily)
        self.glossary_agent = GlossaryAgent(self.client, self.model)
        self.related_work_agent = RelatedWorkAgent(self.client, self.model, self.tavily)

        self._context = {}

    def analyze_paper(
        self,
        pdf_file,
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
        with ThreadPoolExecutor(max_workers=4) as executor:
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

    def get_context_snapshot(self) -> Dict[str, object]:
        return dict(self._context)

    def load_pdf_context(self, pdf_bytes: bytes) -> None:
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

        self._context = {
            "summary": summary_resp.text,
            "full_text": full_text_resp.text,
            "pdf_bytes": pdf_bytes
        }

    def chat_with_agents(self, agent_slugs: List[str], instruction: str) -> Dict[str, str]:
        context = self.get_context_snapshot()
        if not context.get("full_text"):
            raise ValueError("No paper context available. Upload a PDF first.")

        responses = {}
        # We process each agent's response
        for slug in agent_slugs:
            primer = self.CHAT_AGENT_GUIDANCE.get(slug, "You are a helpful research agent.")
            prompt = f"{primer}\n\nPaper summary:\n{context['summary']}\n\nRelevant paper excerpt:\n{str(context['full_text'])[:8000]}\n\nResearcher instruction:\n{instruction}\n\nRespond in Markdown with concise bullet points."

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            responses[slug] = response.text.strip()

        return responses

    def chat_with_agent(self, agent_slug: str, instruction: str, usage_hint: str) -> str:
        context = self.get_context_snapshot()
        if not context.get("full_text"):
            raise ValueError("No paper context available. Upload a PDF first.")

        primer = self.CHAT_AGENT_GUIDANCE.get(agent_slug, "You are a helpful research agent.")
        prompt = f"{primer}\n\nPaper summary:\n{context['summary']}\n\nRelevant paper excerpt:\n{str(context['full_text'])[:8000]}\n\nResearcher instruction:\n{instruction}\n\n{usage_hint}\n\nRespond in Markdown."

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )
        return response.text.strip()
