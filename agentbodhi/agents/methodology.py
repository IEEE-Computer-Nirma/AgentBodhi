import json
import logging
from typing import List

from ..core.models import Weakness
from .base import ResearchAgent

logger = logging.getLogger(__name__)


class MethodologyAgent(ResearchAgent):
    def execute(self, paper_content: str) -> List[Weakness]:
        try:
            # First attempt to extract just the methodology/experiments section for tighter focus
            extraction_prompt = "Extract the 'Methodology', 'Experimental Setup', and 'Datasets' sections from this text. If none explicitly exist, extract the middle 4000 words. \n\n" + paper_content[:20000]
            extraction_resp = self.client.models.generate_content(model=self.model, contents=extraction_prompt)
            methodology_text = extraction_resp.text

            base_prompt = """You are an expert peer reviewer. Analyze this research paper's methodology.
Identify specific weaknesses in these categories:
1. Experimental Design (sample size, controls, bias)
2. Statistical Methods (validity, assumptions, p-hacking risks)
3. Reproducibility (missing code/repo links, private datasets, missing hyperparameter details)
4. Generalizability (limited scope, overfitting)
5. Ethical Considerations (data privacy, fairness)

For each weakness found, provide:
- Category
- Specific description (e.g. "The paper claims to use an open dataset but provides no URL or accession number")
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
                + "\nExtracted Methodology context:\n"
                + methodology_text[:12000]
                + "\n\nReturn ONLY the JSON array."
            )

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            json_text = self._extract_json(response.text)
            weaknesses_data = json.loads(json_text)

            sanitized: List[Weakness] = []
            for raw in weaknesses_data if isinstance(weaknesses_data, list) else []:
                sanitized.append(
                    Weakness(
                        category=str(raw.get("category") or "Uncategorized").strip(),
                        description=str(raw.get("description") or "No description provided").strip(),
                        severity=self._normalize_severity(raw.get("severity")),
                        suggestion=str(raw.get("suggestion") or "Clarify methodology or add missing experimental detail.").strip(),
                    )
                )
            return sanitized
        except Exception as e:
            logger.error(f"Methodology analysis error: {e}")
            return []

    def _normalize_severity(self, raw_value: object) -> str:
        allowed = {"Critical", "Major", "Minor"}
        if not raw_value:
            return "Major"
        normalized = str(raw_value).strip().title()
        if normalized not in allowed:
            return "Major"
        return normalized
