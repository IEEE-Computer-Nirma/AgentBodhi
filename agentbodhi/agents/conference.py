import json
import logging
from typing import Dict

from .base import ResearchAgent

logger = logging.getLogger(__name__)


class ConferenceAgent(ResearchAgent):
    def execute(self, paper_summary: str) -> Dict:
        try:
            # First, extract the core topics and keywords to form a search query
            query_prompt = f"""Extract 2-3 keywords representing the core domain of this paper to search for relevant upcoming academic conferences.
Summary: {paper_summary[:1000]}
Return ONLY a short web search query string, like 'machine learning conferences 2026' or 'computer vision call for papers'."""
            
            query_response = self.client.models.generate_content(
                model=self.model,
                contents=query_prompt
            )
            search_query = query_response.text.strip().replace('"', '')

            # Search the web for upcoming conferences
            tavily_results = {}
            if self.tavily:
                try:
                    tavily_results = self.tavily.search(
                        f"upcoming academic conferences call for papers {search_query}",
                        search_depth="advanced",
                        max_results=3
                    )
                except Exception as e:
                    logger.error(f"Tavily search error: {e}")

            context_results = tavily_results.get('results', [])

            analysis_prompt = f"""Based on the paper summary and the retrieved conference data, recommend 1-3 conferences that fit this paper.
Paper Summary:
{paper_summary[:1000]}

Conference Search Results:
{json.dumps(context_results, indent=2)}

Provide the recommendations as JSON:
{{
    "recommended_conferences": [
        {{
            "name": "Conference Name",
            "reasoning": "Why it's a good fit",
            "url": "Conference URL if available"
        }}
    ]
}}

Return ONLY the JSON."""

            analysis_response = self.client.models.generate_content(
                model=self.model,
                contents=analysis_prompt
            )
            analysis_data = json.loads(self._extract_json(analysis_response.text))
            
            return {
                "search_query": search_query,
                "recommendations": analysis_data.get("recommended_conferences", []),
                "sources": [r.get("url") for r in context_results]
            }

        except Exception as e:
            logger.error(f"Conference analysis error: {e}")
            return {"error": str(e), "recommendations": []}

