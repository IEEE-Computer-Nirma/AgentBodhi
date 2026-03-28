import json
import logging
from typing import Dict

from .base import ResearchAgent

logger = logging.getLogger(__name__)


class GlossaryAgent(ResearchAgent):
    def execute(self, paper_content: str, num_terms: int = 10) -> Dict[str, str]:
        try:
            prompt = f"""Extract the {num_terms} most important technical terms from this research paper.
For each term, provide:
- The exact term
- A clear, concise definition (1-2 sentences)
- Why it's important to understanding the paper

Return as JSON:
{{
    "term1": {{"definition": "...", "importance": "..."}},
    "term2": {{"definition": "...", "importance": "..."}}
}}

Paper content:
{paper_content[:10000]}

Return ONLY the JSON object."""

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            terms = json.loads(self._extract_json(response.text))

            # Cross-reference terms with external web definitions if tavily is available
            if self.tavily:
                for term, info in terms.items():
                    try:
                        web_result = self.tavily.search(f"{term} definition in computer science or domain", max_results=1)
                        if web_result.get('results'):
                            info['web_context'] = web_result['results'][0].get('content', '')
                    except Exception as e:
                        logger.error(f"Tavily search error for glossary term {term}: {e}")

            return terms
        except Exception as e:
            logger.error(f"Glossary generation error: {e}")
            return {}
