import streamlit as st
import time
from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchMind · AI Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #e8e4dc;
}

.stApp {
    background: #0a0a0f;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(255,140,50,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(255,80,30,0.08) 0%, transparent 55%);
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1200px; }

.hero { text-align: center; padding: 3.5rem 0 2.5rem; position: relative; }
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem; font-weight: 500; letter-spacing: 0.25em;
    text-transform: uppercase; color: #ff8c32; margin-bottom: 1rem; opacity: 0.9;
}
.hero h1 {
    font-family: 'Syne', sans-serif; font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 800; line-height: 1.0; letter-spacing: -0.03em;
    color: #f0ebe0; margin: 0 0 1rem;
}
.hero h1 span { color: #ff8c32; }
.hero-sub {
    font-size: 1.05rem; font-weight: 300; color: #a09890;
    max-width: 520px; margin: 0 auto; line-height: 1.65;
}

.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,140,50,0.3), transparent);
    margin: 2rem 0;
}

.section-heading {
    font-family: 'Syne', sans-serif; font-size: 1.4rem; font-weight: 700;
    color: #f0ebe0; margin-bottom: 1.25rem;
}

.steps-grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 2rem 0;
}
.step-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,140,50,0.15);
    border-radius: 12px; padding: 1.25rem; transition: border-color 0.2s;
}
.step-card.running { border-color: #ff8c32; box-shadow: 0 0 20px rgba(255,140,50,0.15); }
.step-card.done { border-color: rgba(80,200,120,0.4); }
.step-num { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #ff8c32; letter-spacing: 0.15em; }
.step-title { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.95rem; margin: 0.4rem 0 0.25rem; }
.step-desc { font-size: 0.78rem; color: #a09890; line-height: 1.4; }
.step-status { font-family: 'DM Mono', monospace; font-size: 0.65rem; margin-top: 0.6rem; text-transform: uppercase; letter-spacing: 0.1em; }
.step-status.waiting { color: #555; }
.step-status.running { color: #ff8c32; }
.step-status.done { color: #50c878; }

.report-panel, .feedback-panel, .result-panel {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,140,50,0.15);
    border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
}
.panel-label {
    font-family: 'DM Mono', monospace; font-size: 0.7rem; font-weight: 500;
    letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 1rem;
}
.panel-label.orange { color: #ff8c32; }
.panel-label.green { color: #50c878; }
.result-panel-title { font-family: 'Syne', sans-serif; font-weight: 600; margin-bottom: 0.75rem; }
.result-content { font-size: 0.9rem; line-height: 1.7; color: #c8c0b8; white-space: pre-wrap; }

.notice {
    text-align: center; font-family: 'DM Mono', monospace; font-size: 0.65rem;
    color: #555; letter-spacing: 0.1em; margin-top: 3rem; padding-top: 2rem;
    border-top: 1px solid rgba(255,255,255,0.05);
}

div[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,140,50,0.2) !important;
    border-radius: 8px !important; color: #e8e4dc !important; font-family: 'DM Sans', sans-serif !important;
}
div[data-testid="stButton"] button {
    background: linear-gradient(135deg, #ff8c32, #ff5020) !important; color: #fff !important;
    border: none !important; border-radius: 8px !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; letter-spacing: 0.05em !important; padding: 0.6rem 2rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = {}
if "running" not in st.session_state:
    st.session_state.running = False
if "done" not in st.session_state:
    st.session_state.done = False
if "topic_input" not in st.session_state:
    st.session_state.topic_input = ""

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Multi-Agent Research Pipeline</div>
    <h1>Research<span>Mind</span></h1>
    <p class="hero-sub">Enter any topic and watch four AI agents search, read, write, and critique a full research report.</p>
</div>
""", unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([5, 1])
with col1:
    topic = st.text_input(
        "Research topic",
        placeholder="e.g. Impact of quantum computing on cryptography",
        label_visibility="collapsed",
    )
with col2:
    st.markdown("<div style='margin-top:0.35rem'>", unsafe_allow_html=True)
    run_btn = st.button("Run Research", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ── Pipeline progress cards ───────────────────────────────────────────────────
def step_card(num, title, status, desc):
    st.markdown(f"""
    <div class="step-card {status}">
        <div class="step-num">{num}</div>
        <div class="step-title">{title}</div>
        <div class="step-desc">{desc}</div>
        <div class="step-status {status}">{status}</div>
    </div>
    """, unsafe_allow_html=True)

r = st.session_state.results

def step_status(step):
    steps = ["search", "reader", "writer", "critic"]
    if step in r:
        return "done"
    if st.session_state.running:
        for k in steps:
            if k not in r:
                return "running" if k == step else "waiting"
    return "waiting"

cols = st.columns(4)
steps_info = [
    ("01", "Search Agent", "search", "Gathers recent web information"),
    ("02", "Reader Agent", "reader", "Scrapes & extracts deep content"),
    ("03", "Writer Chain", "writer", "Drafts the full research report"),
    ("04", "Critic Chain", "critic", "Reviews & scores the report"),
]
for col, (num, title, key, desc) in zip(cols, steps_info):
    with col:
        step_card(num, title, step_status(key), desc)

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        st.session_state.results = {}
        st.session_state.topic_input = topic.strip()
        st.session_state.running = True
        st.session_state.done = False
        st.rerun()

if st.session_state.running and not st.session_state.done:
    results = dict(st.session_state.results)
    topic_val = st.session_state.topic_input

    if "search" not in results:
        with st.spinner("🔍  Search Agent is working…"):
            search_agent = build_search_agent()
            sr = search_agent.invoke({
                "messages": [("user", f"Find recent, reliable and detailed information about: {topic_val}")]
            })
            results["search"] = sr["messages"][-1].content
            st.session_state.results = dict(results)
        st.rerun()

    if "reader" not in results:
        with st.spinner("📄  Reader Agent is scraping top resources…"):
            reader_agent = build_reader_agent()
            rr = reader_agent.invoke({
                "messages": [("user",
                    f"Based on the following search results about '{topic_val}', "
                    f"pick the most relevant URL and scrape it for deeper content.\n\n"
                    f"Search Results:\n{results['search'][:800]}"
                )]
            })
            results["reader"] = rr["messages"][-1].content
            st.session_state.results = dict(results)
        st.rerun()

    if "writer" not in results:
        with st.spinner("✍️  Writer is drafting the report…"):
            research_combined = (
                f"SEARCH RESULTS:\n{results['search']}\n\n"
                f"DETAILED SCRAPED CONTENT:\n{results['reader']}"
            )
            results["writer"] = writer_chain.invoke({
                "topic": topic_val,
                "research": research_combined,
            })
            st.session_state.results = dict(results)
        st.rerun()

    if "critic" not in results:
        with st.spinner("🧐  Critic is reviewing the report…"):
            results["critic"] = critic_chain.invoke({
                "report": results["writer"],
            })
            st.session_state.results = dict(results)
        st.session_state.running = False
        st.session_state.done = True
        st.rerun()

# ── Results display ───────────────────────────────────────────────────────────
r = st.session_state.results

if r:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Results</div>', unsafe_allow_html=True)

    if "search" in r:
        with st.expander("🔍 Search Results (raw)", expanded=False):
            st.markdown(
                f'<div class="result-panel"><div class="result-panel-title">Search Agent Output</div>'
                f'<div class="result-content">{r["search"]}</div></div>',
                unsafe_allow_html=True,
            )

    if "reader" in r:
        with st.expander("📄 Scraped Content (raw)", expanded=False):
            st.markdown(
                f'<div class="result-panel"><div class="result-panel-title">Reader Agent Output</div>'
                f'<div class="result-content">{r["reader"]}</div></div>',
                unsafe_allow_html=True,
            )

    if "writer" in r:
        st.markdown("""
        <div class="report-panel">
            <div class="panel-label orange">📝 Final Research Report</div>
        """, unsafe_allow_html=True)
        st.markdown(r["writer"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.download_button(
            label="⬇  Download Report (.md)",
            data=r["writer"],
            file_name=f"research_report_{int(time.time())}.md",
            mime="text/markdown",
        )

    if "critic" in r:
        st.markdown("""
        <div class="feedback-panel">
            <div class="panel-label green">🧐 Critic Feedback</div>
        """, unsafe_allow_html=True)
        st.markdown(r["critic"])
        st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="notice">
    ResearchMind · Powered by LangChain multi-agent pipeline · Built with Streamlit
</div>
""", unsafe_allow_html=True)
