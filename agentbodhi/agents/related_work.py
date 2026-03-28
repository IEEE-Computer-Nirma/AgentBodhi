import logging

import arxiv

from ..core.utils import clean_query
from .base import ResearchAgent

logger = logging.getLogger(__name__)


class RelatedWorkAgent(ResearchAgent):
    """Uses Arxiv primarily for high-quality academic connections."""

    def execute(self, paper_summary: str, num_papers: int = 5):
        try:
            query_prompt = f"Extract a concise 3-4 word search query to find related academic papers based on this summary:\n{paper_summary[:500]}"
            query_response = self.client.models.generate_content(
                model=self.model,
                contents=query_prompt
            )
            search_query = clean_query(query_response.text.strip())

            search = arxiv.Search(
                query=f'all:{search_query}',
                max_results=num_papers,
                sort_by=arxiv.SortCriterion.Relevance
            )
            client = arxiv.Client(page_size=10, delay_seconds=3, num_retries=5)

            related = []
            for paper in client.results(search):
                related.append({
                    'title': paper.title,
                    'url': paper.entry_id,
                    'snippet': paper.summary[:200] + '...',
                    'relevance': 0.9
                })

            return related
        except Exception as e:
            logger.error(f"Related work search error: {e}")
            return []


