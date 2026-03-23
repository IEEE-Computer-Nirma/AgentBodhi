from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Citation:
    title: str
    authors: List[str]
    url: str
    year: Optional[int]
    status: str
    confidence: float
    abstract: Optional[str] = None


@dataclass
class Weakness:
    category: str
    description: str
    severity: str
    suggestion: str


@dataclass
class Insight:
    type: str
    content: str
    confidence: float
    sources: List[str]


@dataclass
class AnalysisReport:
    paper_id: str
    timestamp: str
    summary: str
    citations: List[Citation]
    weaknesses: List[Weakness]
    sota_analysis: Dict[str, Any]
    glossary: Dict[str, Any]
    novelty_score: float
    reproducibility_score: float
    insights: List[Insight]
    related_work: List[Dict[str, Any]]
    novelty_details: Dict[str, Any] = field(default_factory=dict)

