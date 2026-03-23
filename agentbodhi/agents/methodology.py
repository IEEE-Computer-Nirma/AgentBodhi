import json
import logging
from typing import List

from ..core.models import Weakness
from .base import ResearchAgent

logger = logging.getLogger(__name__)


class MethodologyAgent(ResearchAgent):
    def execute(self, paper_content: str) -> List[Weakness]:
        try:
            base_prompt = """You are an expert peer reviewer. Analyze this research paper's methodology.
Identify specific weaknesses in these categories:
1. Experimental Design (sample size, controls, bias)
2. Statistical Methods (validity, assumptions, p-hacking risks)
3. Reproducibility (missing details, dependencies)
4. Generalizability (limited scope, overfitting)
5. Ethical Considerations (data privacy, fairness)

For each weakness found, provide:
- Category
- Specific description
- Severity (Critical/Major/Minor)
- Constructive suggestion for improvement

Return as JSON array:
[
    {
        "category": "...",
        "description": "...",
        "severity": "Major",
        "suggestion": "..."
    }
]
"""

            prompt = (
                base_prompt
                + "\nPaper content:\n"
                + paper_content[:12000]
                + "\n\nReturn ONLY the JSON array."
            )

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            json_text = self._extract_json(response.text)
            weaknesses_data = json.loads(json_text)
            return [Weakness(**w) for w in weaknesses_data]
        except Exception as e:
            logger.error(f"Methodology analysis error: {e}")
            return []


