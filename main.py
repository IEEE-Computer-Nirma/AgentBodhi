"""
INITIAL SETUP
"""

import streamlit as st
from google import genai
from google.genai import types
import arxiv
from tavily import TavilyClient
import os
import time
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# --- CONFIGURATION ---
st.set_page_config(
    page_title="Agent Bodhi",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fallback for config file if not present
try:
    import config
except ImportError:
    class config:
        GEMINI_API_KEY = None
        TAVILY_API_KEY = None


def clean_query(q: str) -> str:
    # Remove markdown, symbols, junk
    q = re.sub(r'[\*\[\]\(\)"\'`]', '', q)
    q = re.sub(r'\s+', ' ', q)
    return q.strip()

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
    sota_analysis: Dict
    glossary: Dict[str, str]
    novelty_score: float
    reproducibility_score: float
    insights: List[Insight]
    related_work: List[Dict]


# --- CONFIGURATION MANAGER ---

class ConfigManager:
    @staticmethod
    def get_api_keys():
        try:
            gemini_key = getattr(config, 'GEMINI_API_KEY', None)
            tavily_key = getattr(config, 'TAVILY_API_KEY', None)
            return gemini_key, tavily_key
        except Exception as e:
            logger.error(f"Error loading API keys: {e}")
            return None, None

    @staticmethod
    def validate_keys(gemini_key: str, tavily_key: str) -> Tuple[bool, str]:
        if not gemini_key or gemini_key == "YOUR_GEMINI_API_KEY":
            return False, "Invalid Gemini API key"
        if not tavily_key or tavily_key == "YOUR_TAVILY_API_KEY":
            return False, "Invalid Tavily API key"
        return True, "Keys validated"


# --- AGENT BASE CLASS ---

class ResearchAgent:
    """Base class for specialized research agents"""
    def __init__(self, client, model, tavily_client=None):
        self.client = client
        self.model = model
        self.tavily = tavily_client
        self.cache = {}

    def execute(self, *args, **kwargs):
        raise NotImplementedError

    def _cache_key(self, *args) -> str:
        return hashlib.md5(str(args).encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[any]:
        return self.cache.get(key)

    def _set_cache(self, key: str, value: any):
        self.cache[key] = value

    def _extract_json(self, text: str) -> str:
        try:
            # Try direct parse first
            json.loads(text)
            return text
        except:
            pass

        # Extract JSON block
        match = re.search(r'\{.*\}|\[.*\]', text, re.DOTALL)
        if match:
            return match.group(0)

        raise ValueError("No valid JSON found")

# --- SPECIALIZED AGENTS ---

class CitationAgent(ResearchAgent):
    def execute(self, paper_text: str, max_citations: int = 10) -> List[Citation]:
        cache_key = self._cache_key(paper_text[:1000], max_citations)
        cached = self._get_cached(cache_key)
        if cached: return cached

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
                    if citation: citations.append(citation)

            self._set_cache(cache_key, citations)
            return citations
        except Exception as e:
            logger.error(f"Citation extraction error: {e}")
            return []

    def _verify_citation(self, citation_data: Dict) -> Optional[Citation]:
        try:
            title = citation_data.get('title', '').strip()
            if not title or len(title) < 10: return None

            search = arxiv.Search(query=f'ti:"{title}"', max_results=1, sort_by=arxiv.SortCriterion.Relevance)
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
            else:
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


class MethodologyAgent(ResearchAgent):
    def execute(self, paper_content: str) -> List[Weakness]:
        try:
            prompt = f"""You are an expert peer reviewer. Analyze this research paper's methodology.
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
[{{
    "category": "...",
    "description": "...",
    "severity": "Major",
    "suggestion": "..."
}}]

Paper content:
{paper_content[:12000]}

Return ONLY the JSON array."""

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


class SOTAAgent(ResearchAgent):
    def execute(self, paper_summary: str, paper_claims: str) -> Dict:
        try:
            # Step 1: Extract claims
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

            # Step 2: Hybrid Search (Arxiv Latest + Tavily Trends)
            arxiv_results = []
            try:
                search = arxiv.Search(
                    query=f'all:"{task}" AND (all:"state of the art" OR all:"benchmark")',
                    max_results=3,
                    sort_by=arxiv.SortCriterion.SubmittedDate
                )
                arxiv_results = [{"title": p.title, "url": p.entry_id, "year": p.published.year} for p in
                                 search.results()]
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

            # Step 3: Comparative analysis
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

            sources = [r.get('url') for r in combined_context["web_trends"]] + [p.get('url') for p in
                                                                                combined_context["academic_papers"]]

            return {
                'claims': claims_data,
                'analysis': analysis_data,
                'sources': list(set(sources))
            }
        except Exception as e:
            logger.error(f"SOTA analysis error: {e}")
            return {'error': str(e)}


class NoveltyAgent(ResearchAgent):
    def execute(self, paper_summary: str, paper_content: str) -> Dict:
        try:
            # Hybrid search
            arxiv_results = []
            try:
                query_prompt = f"Extract a 4-word keyword search query from this summary to find similar papers:\n{paper_summary[:500]}"
                q_resp = self.client.models.generate_content(model=self.model, contents=query_prompt)
                search = arxiv.Search(query=q_resp.text.strip().replace('"', ''), max_results=3,
                                      sort_by=arxiv.SortCriterion.Relevance)
                arxiv_results = [{"title": p.title, "summary": p.summary[:200]} for p in search.results()]
            except:
                pass

            tavily_results = {}
            if self.tavily:
                try:
                    tavily_results = self.tavily.search(
                        f"{paper_summary[:200]} similar research new approaches 2024 2025",
                        search_depth="basic",
                        max_results=2
                    )
                except:
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


class RelatedWorkAgent(ResearchAgent):
    """Upgraded: Uses Arxiv primarily for high-quality academic connections"""

    def execute(self, paper_summary: str, num_papers: int = 5) -> List[Dict]:
        try:
            # LLM generates a precise Arxiv query
            query_prompt = f"Extract a concise 3-4 word search query to find related academic papers based on this summary:\n{paper_summary[:500]}"
            query_response = self.client.models.generate_content(
                model=self.model,
                contents=query_prompt
            )
            search_query = query_response.text.strip().replace('"', '')
            search_query = clean_query(
                query_response.text.strip()
            )
            # Arxiv direct search
            search = arxiv.Search(
                query=f'all:{search_query}',
                max_results=num_papers,
                sort_by=arxiv.SortCriterion.Relevance
            )

            related = []
            for paper in search.results():
                related.append({
                    'title': paper.title,
                    'url': paper.entry_id,
                    'snippet': paper.summary[:200] + '...',
                    'relevance': 0.9  # Direct academic database hit
                })

            return related
        except Exception as e:
            logger.error(f"Related work search error: {e}")
            return []


# --- ORCHESTRATOR ---

class ResearchOrchestrator:
    def __init__(self, gemini_key: str, tavily_key: str):
        self.client = genai.Client(api_key=gemini_key)
        self.model = "gemini-3.1-flash-lite-preview"
        self.tavily = TavilyClient(api_key=tavily_key)

        # Initialize agents with new signature
        self.citation_agent = CitationAgent(self.client, self.model)
        self.methodology_agent = MethodologyAgent(self.client, self.model)
        self.sota_agent = SOTAAgent(self.client, self.model, self.tavily)
        self.novelty_agent = NoveltyAgent(self.client, self.model, self.tavily)
        self.glossary_agent = GlossaryAgent(self.client, self.model)
        self.related_work_agent = RelatedWorkAgent(self.client, self.model, self.tavily)

    def analyze_paper(self, pdf_file, progress_callback=None) -> AnalysisReport:
        try:
            if progress_callback:
                progress_callback("Reading PDF...", 0.1)

            # Direct PDF Byte Processing
            pdf_bytes = pdf_file.getvalue()

            if progress_callback:
                progress_callback("Extracting paper content & generating summary...", 0.2)

            summary_resp = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                    "Provide a comprehensive 400-word technical summary of this paper including: research question, methodology, key findings, and contributions."
                ]
            )

            full_text_resp = self.client.models.generate_content(
                model=self.model,
                contents=[
                    types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                    "Extract all text from this paper including abstract, methodology, results, and discussion sections."
                ]
            )

            paper_summary = summary_resp.text
            paper_text = full_text_resp.text

            if progress_callback:
                progress_callback("Running specialized agents (Hybrid Search)...", 0.3)

            results = {}
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    'citations': executor.submit(self.citation_agent.execute, paper_text, 10),
                    'weaknesses': executor.submit(self.methodology_agent.execute, paper_text),
                    'sota': executor.submit(self.sota_agent.execute, paper_summary, paper_text[:5000]),
                    'novelty': executor.submit(self.novelty_agent.execute, paper_summary, paper_text[:8000]),
                    'glossary': executor.submit(self.glossary_agent.execute, paper_text, 12),
                    'related': executor.submit(self.related_work_agent.execute, paper_summary, 5)
                }

                for key, future in futures.items():
                    try:
                        results[key] = future.result(timeout=60)
                        if progress_callback:
                            progress = 0.3 + (0.6 * (len([f for f in futures.values() if f.done()]) / len(futures)))
                            progress_callback(f"Completed {key} analysis...", progress)
                    except Exception as e:
                        logger.error(f"Agent {key} failed: {e}")
                        results[key] = [] if key != 'sota' else {}

            if progress_callback:
                progress_callback("Calculating reproducibility metrics...", 0.9)

            reproducibility_score = self._calculate_reproducibility(paper_text)
            insights = self._generate_insights(results, paper_summary)

            report = AnalysisReport(
                paper_id=hashlib.md5(pdf_bytes).hexdigest()[:8],
                timestamp=datetime.now().isoformat(),
                summary=paper_summary,
                citations=results.get('citations', []),
                weaknesses=results.get('weaknesses', []),
                sota_analysis=results.get('sota', {}),
                glossary=results.get('glossary', {}),
                novelty_score=results.get('novelty', {}).get('novelty_score', 0),
                reproducibility_score=reproducibility_score,
                insights=insights,
                related_work=results.get('related', [])
            )

            if progress_callback:
                progress_callback("Analysis complete!", 1.0)

            return report
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise

    def _calculate_reproducibility(self, paper_text: str) -> float:
        try:
            prompt = f"""Assess the reproducibility of this research paper.
Check for:
1. Code/data availability
2. Hyperparameter specifications
3. Hardware/environment details
4. Statistical significance reporting
5. Clear methodology description

Return a score from 0-10 as JSON:
{{"score": 7.5, "reasoning": "..."}}

Paper:
{paper_text[:8000]}

Return ONLY the JSON."""

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )
            json_match = re.search(r'\{.*?\}', response.text, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                return data.get('score', 5.0)
            return 5.0
        except Exception as e:
            logger.error(f"Reproducibility calculation error: {e}")
            return 5.0

    def _generate_insights(self, results: Dict, summary: str) -> List[Insight]:
        insights = []

        citations = results.get('citations', [])
        verified_count = sum(1 for c in citations if c.status == "Verified")
        if citations:
            insights.append(Insight(
                type="Citations",
                content=f"{verified_count}/{len(citations)} citations verified on ArXiv",
                confidence=0.9,
                sources=["ArXiv"]
            ))

        sota = results.get('sota', {})
        if 'analysis' in sota:
            is_sota = sota['analysis'].get('is_sota', False)
            insights.append(Insight(
                type="State-of-the-Art",
                content=f"{'Current SOTA' if is_sota else 'Not current SOTA'} - {sota['analysis'].get('recommendation', 'N/A')}",
                confidence=sota['analysis'].get('confidence', 0.5),
                sources=sota.get('sources', [])
            ))

        novelty = results.get('novelty', {})
        if 'novelty_score' in novelty:
            insights.append(Insight(
                type="Novelty",
                content=f"Novelty score: {novelty['novelty_score']}/10 - {novelty.get('impact_potential', 'N/A')} impact potential",
                confidence=0.8,
                sources=["Comparative analysis"]
            ))

        weaknesses = results.get('weaknesses', [])
        if weaknesses:
            critical = sum(1 for w in weaknesses if w.severity == "Critical")
            insights.append(Insight(
                type="Methodology",
                content=f"Found {len(weaknesses)} methodological issues ({critical} critical)",
                confidence=0.85,
                sources=["Peer review analysis"]
            ))

        return insights


# --- UI COMPONENTS ---

def render_sidebar():
    with st.sidebar:
        st.title("⚙️ Settings")

        st.markdown("### API Configuration")
        gemini_key = st.text_input("Gemini API Key", type="password", help="Get from Google AI Studio")
        tavily_key = st.text_input("Tavily API Key", type="password", help="Get from https://tavily.com")

        st.markdown("### Analysis Options")
        max_citations = st.slider("Max Citations to Verify", 5, 20, 10)
        max_glossary = st.slider("Glossary Terms", 5, 20, 12)

        st.markdown("### Export")
        if st.session_state.get('report'):
            report_json = json.dumps(asdict(st.session_state.report), indent=2, default=str)
            st.download_button(
                "📥 Download Report (JSON)",
                data=report_json,
                file_name=f"analysis_{st.session_state.report.paper_id}.json",
                mime="application/json"
            )

        return gemini_key, tavily_key, max_citations, max_glossary


def render_citation_analysis(citations: List[Citation]):
    st.markdown("### 📚 Citation Verification")
    if not citations:
        st.warning("No citations extracted")
        return

    verified = [c for c in citations if c.status == "Verified"]
    not_found = [c for c in citations if c.status == "Not Found"]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Citations", len(citations))
    col2.metric("Verified", len(verified))
    col3.metric("Not Found", len(not_found))

    st.markdown("#### Verified Citations")
    for cit in verified:
        with st.expander(f"✅ {cit.title[:80]}..."):
            st.markdown(f"**Authors:** {', '.join(cit.authors[:3])}")
            st.markdown(f"**Year:** {cit.year}")
            st.markdown(f"**URL:** [{cit.url}]({cit.url})")
            if cit.abstract:
                st.markdown(f"**Abstract:** {cit.abstract}")

    if not_found:
        st.markdown("#### Unverified Citations")
        for cit in not_found:
            st.warning(f"❌ {cit.title}")


def render_methodology_analysis(weaknesses: List[Weakness]):
    st.markdown("### ⚖️ Methodology Critique")
    if not weaknesses:
        st.info("No significant weaknesses identified")
        return

    critical = [w for w in weaknesses if w.severity == "Critical"]
    major = [w for w in weaknesses if w.severity == "Major"]
    minor = [w for w in weaknesses if w.severity == "Minor"]

    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 Critical", len(critical))
    col2.metric("🟡 Major", len(major))
    col3.metric("🟢 Minor", len(minor))

    for weakness in weaknesses:
        severity_emoji = {"Critical": "🔴", "Major": "🟡", "Minor": "🟢"}
        with st.expander(f"{severity_emoji.get(weakness.severity, '⚪')} {weakness.category}"):
            st.markdown(f"**Issue:** {weakness.description}")
            st.markdown(f"**Severity:** {weakness.severity}")
            st.markdown(f"**Suggestion:** {weakness.suggestion}")


def render_sota_analysis(sota_data: Dict):
    st.markdown("### 📈 State-of-the-Art Analysis")
    if not sota_data or 'error' in sota_data:
        st.error("SOTA analysis failed")
        return

    analysis = sota_data.get('analysis', {})

    col1, col2 = st.columns(2)
    with col1:
        is_sota = analysis.get('is_sota', False)
        st.metric("SOTA Status", "✅ Current" if is_sota else "❌ Outdated",
                  f"{analysis.get('confidence', 0) * 100:.0f}% confidence")
    with col2:
        st.markdown("**Recommendation**")
        st.info(analysis.get('recommendation', 'N/A'))

    st.markdown("**Detailed Comparison**")
    st.write(analysis.get('comparison', 'No comparison available'))

    newer_methods = analysis.get('newer_methods', [])
    if newer_methods:
        st.markdown("**Newer Methods Found:**")
        for method in newer_methods:
            st.markdown(f"- {method}")

    if 'sources' in sota_data:
        st.markdown("**Sources (Arxiv & Web):**")
        for url in sota_data['sources']:
            st.markdown(f"- [{url}]({url})")


def render_novelty_analysis(novelty_data: Dict):
    st.markdown("### 💡 Novelty Assessment")
    if not novelty_data or 'error' in novelty_data:
        st.error("Novelty assessment failed")
        return

    score = novelty_data.get('novelty_score', 0)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Novelty Score", f"{score}/10")
        st.progress(score / 10)
        impact = novelty_data.get('impact_potential', 'Unknown')
        impact_color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
        st.metric("Impact Potential", f"{impact_color.get(impact, '⚪')} {impact}")
    with col2:
        st.markdown("**Analysis**")
        st.write(novelty_data.get('reasoning', 'No reasoning provided'))

    contributions = novelty_data.get('new_contributions', [])
    if contributions:
        st.markdown("**Novel Contributions:**")
        for contrib in contributions:
            st.markdown(f"✨ {contrib}")

    incremental = novelty_data.get('incremental_aspects', [])
    if incremental:
        st.markdown("**Incremental Aspects:**")
        for aspect in incremental:
            st.markdown(f"📊 {aspect}")


def render_glossary(glossary: Dict):
    st.markdown("### 📖 Technical Glossary")
    if not glossary:
        st.info("No technical terms extracted")
        return

    for term, data in glossary.items():
        with st.expander(f"**{term}**"):
            if isinstance(data, dict):
                st.markdown(f"**Definition:** {data.get('definition', 'N/A')}")
                st.markdown(f"**Importance:** {data.get('importance', 'N/A')}")
            else:
                st.write(data)


def render_insights(insights: List[Insight]):
    st.markdown("### 🎯 Key Insights")
    if not insights:
        st.info("No insights generated")
        return

    for insight in insights:
        confidence_bar = "🟢" * int(insight.confidence * 5) + "⚪" * (5 - int(insight.confidence * 5))
        with st.container():
            st.markdown(f"**{insight.type}**")
            st.write(insight.content)
            st.caption(f"Confidence: {confidence_bar} ({insight.confidence * 100:.0f}%)")
            st.markdown("---")


def render_related_work(related: List[Dict]):
    st.markdown("### 🔗 Related Research (Powered by ArXiv)")
    if not related:
        st.info("No related work found")
        return

    for paper in related:
        with st.expander(paper.get('title', 'Unknown Title')):
            st.markdown(f"**Relevance:** {paper.get('relevance', 0) * 100:.0f}%")
            st.write(paper.get('snippet', 'No description'))
            st.markdown(f"[Read more]({paper.get('url', '#')})")


# --- MAIN APP ---

def main():
    st.title("🧠 OmniResearch: Multi-Agent Research Analysis")
    st.markdown("""
    **Advanced AI-powered research paper analysis using specialized agents**
    This system employs multiple AI agents to provide:
    - Citation verification via ArXiv
    - Methodology critique
    - State-of-the-art comparison (Hybrid: ArXiv + Web Search)
    - Novelty assessment
    - Technical glossary
    - Related work discovery
    """)

    gemini_key, tavily_key, max_citations, max_glossary = render_sidebar()

    if not gemini_key or not tavily_key:
        st.warning("⚠️ Please configure API keys in the sidebar")
        return

    valid, message = ConfigManager.validate_keys(gemini_key, tavily_key)
    if not valid:
        st.error(f"❌ {message}")
        return

    st.markdown("---")
    uploaded_file = st.file_uploader(
        "📄 Upload Research Paper (PDF)",
        type="pdf",
        help="Upload a research paper in PDF format for comprehensive analysis"
    )

    if uploaded_file:
        orchestrator = ResearchOrchestrator(gemini_key, tavily_key)

        if st.button("🚀 Analyze Paper", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(message, progress):
                status_text.text(message)
                progress_bar.progress(progress)

            try:
                with st.spinner("Analyzing paper..."):
                    report = orchestrator.analyze_paper(
                        uploaded_file,
                        progress_callback=update_progress
                    )
                    st.session_state.report = report

                progress_bar.empty()
                status_text.empty()
                st.success("✅ Analysis complete!")

            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                logger.error(f"Analysis error: {e}", exc_info=True)
                return

        if st.session_state.get('report'):
            report = st.session_state.report

            st.markdown("---")
            st.markdown("## 📊 Analysis Report")

            with st.expander("📝 Executive Summary", expanded=True):
                st.write(report.summary)
                col1, col2, col3 = st.columns(3)
                col1.metric("Novelty Score", f"{report.novelty_score:.1f}/10")
                col2.metric("Reproducibility", f"{report.reproducibility_score:.1f}/10")
                col3.metric("Citations Found", len(report.citations))

            render_insights(report.insights)

            tabs = st.tabs([
                "📚 Citations",
                "⚖️ Methodology",
                "📈 SOTA",
                "💡 Novelty",
                "📖 Glossary",
                "🔗 Related Work"
            ])

            with tabs[0]: render_citation_analysis(report.citations)
            with tabs[1]: render_methodology_analysis(report.weaknesses)
            with tabs[2]: render_sota_analysis(report.sota_analysis)
            with tabs[3]: render_novelty_analysis({
                'novelty_score': report.novelty_score,
                'impact_potential': 'High' if report.novelty_score > 7 else 'Medium' if report.novelty_score > 4 else 'Low'
            })
            with tabs[4]: render_glossary(report.glossary)
            with tabs[5]: render_related_work(report.related_work)


if __name__ == "__main__":
    if 'report' not in st.session_state:
        st.session_state.report = None
    main()