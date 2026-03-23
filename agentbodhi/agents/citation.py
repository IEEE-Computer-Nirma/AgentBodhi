import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import arxiv

from ..core.models import Citation
from .base import ResearchAgent

logger = logging.getLogger(__name__)


class CitationAgent(ResearchAgent):
    def execute(self, paper_text: str, max_citations: int = 10) -> List[Citation]:
        cache_key = self._cache_key(paper_text[:1000], max_citations)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        try:
            extraction_prompt = f"""Analyze this research paper and extract the {max_citations} most important citations.
For each citation, identify:
1. Full paper title
2. Authors (if mentioned)
3. Publication year (if mentioned)

Return as JSON array with structure:
[{{"title": "...", "authors": ["..."], "year": 2024}}]

Paper text:
{paper_text[:8000]}

Return ONLY the JSON array, no explanation."""

            response = self.client.models.generate_content(
                model=self.model,
                contents=extraction_prompt
            )

            json_text = self._extract_json(response.text)
            citations_data = json.loads(json_text)

            citations = []
            with ThreadPoolExecutor(max_workers=3) as executor:
                future_to_citation = {
                    executor.submit(self._verify_citation, cit): cit
                    for cit in citations_data
                }
                for future in as_completed(future_to_citation):
                    citation = future.result()
                    if citation:
                        citations.append(citation)

            self._set_cache(cache_key, citations)
            return citations
        except Exception as e:
            logger.error(f"Citation extraction error: {e}")
            return []

    def _verify_citation(self, citation_data: Dict) -> Optional[Citation]:
        try:
            title = citation_data.get('title', '').strip()
            if not title or len(title) < 10:
                return None

            search = arxiv.Search(query=f'ti:"{title}"', max_results=1,
                                  sort_by=arxiv.SortCriterion.Relevance)
            client = arxiv.Client()
            results = list(client.results(search))

            if results:
                paper = results[0]
                return Citation(
                    title=paper.title,
                    authors=[author.name for author in paper.authors],
                    url=paper.entry_id,
                    year=paper.published.year,
                    status="Verified",
                    confidence=0.9,
                    abstract=paper.summary[:300]
                )
            return Citation(
                title=title,
                authors=citation_data.get('authors', []),
                url="#",
                year=citation_data.get('year'),
                status="Not Found",
                confidence=0.3
            )
        except Exception as e:
            logger.error(f"Citation verification error: {e}")
            return None

