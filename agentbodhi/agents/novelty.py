import json
import logging

import arxiv

from .base import ResearchAgent

logger = logging.getLogger(__name__)


class NoveltyAgent(ResearchAgent):
    def execute(self, paper_summary: str, paper_content: str) -> dict:
        try:
            arxiv_results = []
            try:
                query_prompt = f"Extract a 4-word keyword search query from this summary to find similar papers:\n{paper_summary[:500]}"
                q_resp = self.client.models.generate_content(model=self.model, contents=query_prompt)
                search = arxiv.Search(
                    query=q_resp.text.strip().replace('"', ''),
                    max_results=3,
                    sort_by=arxiv.SortCriterion.Relevance
                )
                client = arxiv.Client(page_size=10, delay_seconds=3, num_retries=5)
                arxiv_results = [{"title": p.title, "summary": p.summary[:200]} for p in client.results(search)]
            except Exception:
                pass

            tavily_results = {}
            if self.tavily:
                try:
                    tavily_results = self.tavily.search(
                        f"{paper_summary[:200]} similar research new approaches 2024 2025",
                        search_depth="basic",
                        max_results=2
                    )
                except Exception:
                    pass

            context = {
                "similar_academic": arxiv_results,
                "similar_web": tavily_results.get('results', [])
            }

            novelty_prompt = f"""Assess the novelty of this research paper using the context.
Paper summary:
{paper_summary}

Similar recent work found:
{json.dumps(context, indent=2)}

Evaluate:
1. Novelty score (0-10)
2. What's genuinely new vs incremental
3. Relationship to prior work
4. Potential impact

Return as JSON:
{{
    "novelty_score": 7.5,
    "new_contributions": ["..."],
    "incremental_aspects": ["..."],
    "related_work_gaps": "...",
    "impact_potential": "High/Medium/Low",
    "reasoning": "..."
}}
Return ONLY the JSON."""

            response = self.client.models.generate_content(
                model=self.model,
                contents=novelty_prompt
            )
            return json.loads(self._extract_json(response.text))
        except Exception as e:
            logger.error(f"Novelty assessment error: {e}")
            return {'novelty_score': 0, 'error': str(e)}

