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
            return json.loads(self._extract_json(response.text))
        except Exception as e:
            logger.error(f"Glossary generation error: {e}")
            return {}

