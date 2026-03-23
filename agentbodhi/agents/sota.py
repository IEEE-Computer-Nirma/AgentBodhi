import json
import logging
from typing import Dict

import arxiv

from .base import ResearchAgent

logger = logging.getLogger(__name__)


class SOTAAgent(ResearchAgent):
    def execute(self, paper_summary: str, paper_claims: str) -> Dict:
        try:
            claims_prompt = f"""Extract the main performance claims from this paper summary.
Return as JSON:
{{
    "task": "what problem/task",
    "metrics": ["metric1", "metric2"],
    "reported_performance": {{"metric1": "value"}},
    "dataset": "dataset name"
}}

Summary:
{paper_summary}
Return ONLY the JSON."""

            claims_response = self.client.models.generate_content(
                model=self.model,
                contents=claims_prompt
            )
            claims_data = json.loads(self._extract_json(claims_response.text))
            task = claims_data.get('task', '')

            arxiv_results = []
            try:
                search = arxiv.Search(
                    query=f'all:"{task}" AND (all:"state of the art" OR all:"benchmark")',
                    max_results=3,
                    sort_by=arxiv.SortCriterion.SubmittedDate
                )
                arxiv_results = [
                    {"title": p.title, "url": p.entry_id, "year": p.published.year}
                    for p in search.results()
                ]
            except Exception as e:
                logger.error(f"Arxiv search error: {e}")

            tavily_results = {}
            if self.tavily:
                try:
                    tavily_results = self.tavily.search(
                        f"state of the art {task} 2025 2026 benchmark leaderboard",
                        search_depth="advanced",
                        max_results=2
                    )
                except Exception as e:
                    logger.error(f"Tavily search error: {e}")

            combined_context = {
                "academic_papers": arxiv_results,
                "web_trends": tavily_results.get('results', [])
            }

            comparison_prompt = f"""Compare this paper's claims against current SOTA findings.
Paper claims:
{json.dumps(claims_data, indent=2)}

Current research context (Academic & Web):
{json.dumps(combined_context, indent=2)}

Provide analysis as JSON:
{{
    "is_sota": true/false,
    "confidence": 0.0-1.0,
    "comparison": "detailed comparison based on academic/web data",
    "newer_methods": ["list of newer approaches if any"],
    "recommendation": "verdict"
}}

Return ONLY the JSON."""

            analysis = self.client.models.generate_content(
                model=self.model,
                contents=comparison_prompt
            )
            analysis_data = json.loads(self._extract_json(analysis.text))

            sources = [r.get('url') for r in combined_context['web_trends']] + [p.get('url') for p in
                                                                                combined_context['academic_papers']]

            return {
                'claims': claims_data,
                'analysis': analysis_data,
                'sources': list(set(sources))
            }
        except Exception as e:
            logger.error(f"SOTA analysis error: {e}")
            return {'error': str(e)}

