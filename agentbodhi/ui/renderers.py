import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import streamlit as st

from ..core.models import Citation, Insight, Weakness


def render_sidebar() -> tuple[Optional[str], Optional[str], int, int]:
    with st.sidebar:
        st.title("⚙️ Settings")

        st.markdown("### API Configuration")
        gemini_key = st.text_input("Gemini API Key", type="password", help="Get from Google AI Studio")
        tavily_key = st.text_input("Tavily API Key", type="password", help="Get from https://tavily.com")

        st.markdown("### Analysis Options")
        max_citations = st.slider("Max Citations to Verify", 5, 20, 10)
        max_glossary = st.slider("Glossary Terms", 5, 20, 12)

        st.markdown("### Export")
        report = st.session_state.get("report")
        if report:
            report_json = json.dumps(asdict(report), indent=2, default=str)
            st.download_button(
                "📥 Download Report (JSON)",
                data=report_json,
                file_name=f"analysis_{report.paper_id}.json",
                mime="application/json"
            )

    return gemini_key, tavily_key, max_citations, max_glossary


def render_citation_analysis(citations: List[Citation]) -> None:
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


def render_methodology_analysis(weaknesses: List[Weakness]) -> None:
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
        emoji = {"Critical": "🔴", "Major": "🟡", "Minor": "🟢"}
        with st.expander(f"{emoji.get(weakness.severity, '⚪')} {weakness.category}"):
            st.markdown(f"**Issue:** {weakness.description}")
            st.markdown(f"**Severity:** {weakness.severity}")
            st.markdown(f"**Suggestion:** {weakness.suggestion}")


def render_sota_analysis(sota_data: Dict[str, Any]) -> None:
    st.markdown("### 📈 State-of-the-Art Analysis")
    if not sota_data or sota_data.get("error"):
        st.error("SOTA analysis failed")
        return

    analysis = sota_data.get("analysis", {})

    col1, col2 = st.columns(2)
    with col1:
        is_sota = analysis.get("is_sota", False)
        st.metric(
            "SOTA Status",
            "✅ Current" if is_sota else "❌ Outdated",
            f"{analysis.get('confidence', 0) * 100:.0f}% confidence"
        )
    with col2:
        st.markdown("**Recommendation**")
        st.info(analysis.get("recommendation", "N/A"))

    st.markdown("**Detailed Comparison**")
    st.write(analysis.get("comparison", "No comparison available"))

    newer_methods = analysis.get("newer_methods", [])
    if newer_methods:
        st.markdown("**Newer Methods Found:**")
        for method in newer_methods:
            st.markdown(f"- {method}")

    sources = sota_data.get("sources", [])
    if sources:
        st.markdown("**Sources (Arxiv & Web):**")
        for url in sources:
            st.markdown(f"- [{url}]({url})")


def render_novelty_analysis(novelty_data: Dict[str, Any]) -> None:
    st.markdown("### 💡 Novelty Assessment")
    if not novelty_data or novelty_data.get("error"):
        st.error("Novelty assessment failed")
        return

    score = novelty_data.get("novelty_score", 0)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric("Novelty Score", f"{score}/10")
        st.progress(score / 10)
        impact = novelty_data.get("impact_potential", "Unknown")
        impact_color = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}
        st.metric("Impact Potential", f"{impact_color.get(impact, '⚪')} {impact}")
    with col2:
        st.markdown("**Analysis**")
        st.write(novelty_data.get("reasoning", "No reasoning provided"))

    contributions = novelty_data.get("new_contributions", [])
    if contributions:
        st.markdown("**Novel Contributions:**")
        for contrib in contributions:
            st.markdown(f"✨ {contrib}")

    incremental = novelty_data.get("incremental_aspects", [])
    if incremental:
        st.markdown("**Incremental Aspects:**")
        for aspect in incremental:
            st.markdown(f"📊 {aspect}")


def render_glossary(glossary: Dict[str, Any]) -> None:
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


def render_insights(insights: List[Insight]) -> None:
    st.markdown("### 🎯 Key Insights")
    if not insights:
        st.info("No insights generated")
        return

    for insight in insights:
        confidence = int(insight.confidence * 5)
        confidence_bar = "🟢" * confidence + "⚪" * (5 - confidence)
        with st.container():
            st.markdown(f"**{insight.type}**")
            st.write(insight.content)
            st.caption(f"Confidence: {confidence_bar} ({insight.confidence * 100:.0f}%)")
            st.markdown("---")


def render_related_work(related: List[Dict[str, Any]]) -> None:
    st.markdown("### 🔗 Related Research (Powered by ArXiv)")
    if not related:
        st.info("No related work found")
        return

    for paper in related:
        with st.expander(paper.get("title", "Unknown Title")):
            st.markdown(f"**Relevance:** {paper.get('relevance', 0) * 100:.0f}%")
            st.write(paper.get("snippet", "No description"))
            st.markdown(f"[Read more]({paper.get('url', '#')})")

