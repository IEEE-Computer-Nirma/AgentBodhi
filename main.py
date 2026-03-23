import streamlit as st

from agentbodhi.configuration import ConfigManager
from agentbodhi.core.orchestrator import ResearchOrchestrator
from agentbodhi.ui.renderers import (
    render_sidebar,
    render_citation_analysis,
    render_methodology_analysis,
    render_sota_analysis,
    render_novelty_analysis,
    render_glossary,
    render_insights,
    render_related_work,
)


st.set_page_config(
    page_title="Agent Bodhi",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🧠",
)


def _ensure_session_state() -> None:
    if "report" not in st.session_state:
        st.session_state.report = None
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = None
    if "orchestrator_keys" not in st.session_state:
        st.session_state.orchestrator_keys = (None, None)


def _get_orchestrator(gemini_key: str, tavily_key: str) -> ResearchOrchestrator:
    current_keys = st.session_state.orchestrator_keys
    if current_keys != (gemini_key, tavily_key) or st.session_state.orchestrator is None:
        st.session_state.orchestrator = ResearchOrchestrator(gemini_key, tavily_key)
        st.session_state.orchestrator_keys = (gemini_key, tavily_key)
    return st.session_state.orchestrator


def main() -> None:
    _ensure_session_state()

    st.title("🧠 OmniResearch: Multi-Agent Research Analysis")
    st.markdown(
        """
        **Advanced AI-powered research paper analysis using specialized agents**
        This system employs multiple AI agents to provide:
        - Citation verification via ArXiv
        - Methodology critique
        - State-of-the-art comparison (Hybrid: ArXiv + Web Search)
        - Novelty assessment
        - Technical glossary
        - Related work discovery
        """
    )

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
        help="Upload a research paper in PDF format for comprehensive analysis",
    )

    if not uploaded_file:
        return

    orchestrator = _get_orchestrator(gemini_key, tavily_key)

    if st.button("🚀 Analyze Paper", type="primary"):
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_progress(message: str, value: float) -> None:
            status_text.text(message)
            progress_bar.progress(value)

        try:
            with st.spinner("Analyzing paper..."):
                report = orchestrator.analyze_paper(
                    uploaded_file,
                    progress_callback=update_progress,
                    max_citations=max_citations,
                    glossary_terms=max_glossary,
                )
                st.session_state.report = report
        except Exception as exc:
            st.error(f"❌ Analysis failed: {exc}")
            return
        finally:
            progress_bar.empty()
            status_text.empty()

        st.success("✅ Analysis complete!")

    report = st.session_state.report
    if not report:
        return

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
        "🔗 Related Work",
    ])

    with tabs[0]:
        render_citation_analysis(report.citations)
    with tabs[1]:
        render_methodology_analysis(report.weaknesses)
    with tabs[2]:
        render_sota_analysis(report.sota_analysis)
    with tabs[3]:
        novelty_payload = dict(report.novelty_details or {})
        if "novelty_score" not in novelty_payload:
            novelty_payload["novelty_score"] = report.novelty_score
        if "impact_potential" not in novelty_payload:
            score = report.novelty_score
            novelty_payload["impact_potential"] = (
                "High" if score > 7 else "Medium" if score > 4 else "Low"
            )
        render_novelty_analysis(novelty_payload)
    with tabs[4]:
        render_glossary(report.glossary)
    with tabs[5]:
        render_related_work(report.related_work)


if __name__ == "__main__":
    main()

