import time

import streamlit as st

from agents import build_reader_agent, build_search_agent, critic_chain, writer_chain
from config import READER_CONTEXT_CHARS, get_logger, validate_config

logger = get_logger(__name__)

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

.step-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,140,50,0.15);
    border-radius: 12px; padding: 1.25rem; transition: border-color 0.2s;
    height: 100%;
}
.step-card.running { border-color: #ff8c32; box-shadow: 0 0 20px rgba(255,140,50,0.15); }
.step-card.done { border-color: rgba(80,200,120,0.4); }
.step-card.error { border-color: rgba(220,80,80,0.5); }
.step-num { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #ff8c32; letter-spacing: 0.15em; }
.step-title { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.95rem; margin: 0.4rem 0 0.25rem; }
.step-desc { font-size: 0.78rem; color: #a09890; line-height: 1.4; }
.step-status { font-family: 'DM Mono', monospace; font-size: 0.65rem; margin-top: 0.6rem; text-transform: uppercase; letter-spacing: 0.1em; }
.step-status.waiting { color: #555; }
.step-status.running { color: #ff8c32; }
.step-status.done { color: #50c878; }
.step-status.error { color: #dc5050; }

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

# ── Pipeline definition ───────────────────────────────────────────────────────
STEPS = [
    ("01", "Search Agent", "search", "Gathers recent web information"),
    ("02", "Reader Agent", "reader", "Scrapes & extracts deep content"),
    ("03", "Writer Chain", "writer", "Drafts the full research report"),
    ("04", "Critic Chain", "critic", "Reviews & scores the report"),
]
STEP_KEYS = [key for _, _, key, _ in STEPS]

EXAMPLE_TOPICS = [
    "Impact of quantum computing on cryptography",
    "How CRISPR gene editing is reshaping medicine",
    "The economics of nuclear fusion power",
]

# ── Cached resources (built once per session) ─────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_search_agent():
    return build_search_agent()


@st.cache_resource(show_spinner=False)
def get_reader_agent():
    return build_reader_agent()


# ── Session state ─────────────────────────────────────────────────────────────
def init_state() -> None:
    defaults = {
        "results": {},
        "running": False,
        "done": False,
        "topic_input": "",
        "error": None,
        "started_at": None,
        "elapsed": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def reset_run() -> None:
    st.session_state.results = {}
    st.session_state.running = False
    st.session_state.done = False
    st.session_state.error = None
    st.session_state.started_at = None
    st.session_state.elapsed = None


init_state()

# ── Config check ──────────────────────────────────────────────────────────────
config_status = validate_config()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Multi-Agent Research Pipeline</div>
    <h1>Research<span>Mind</span></h1>
    <p class="hero-sub">Enter any topic and watch four AI agents search, read, write, and critique a full research report.</p>
</div>
""", unsafe_allow_html=True)

if not config_status.ok:
    st.error(
        f"⚠️  {config_status.message}\n\n"
        "Add the missing keys to your `.env` file (see `.env.example`) and restart."
    )

# ── Input ─────────────────────────────────────────────────────────────────────
busy = st.session_state.running and not st.session_state.done

col1, col2 = st.columns([5, 1])
with col1:
    topic = st.text_input(
        "Research topic",
        placeholder="e.g. Impact of quantum computing on cryptography",
        label_visibility="collapsed",
        disabled=busy,
    )
with col2:
    run_btn = st.button(
        "Run Research",
        use_container_width=True,
        disabled=busy or not config_status.ok,
    )

# Example topic chips
ex_cols = st.columns(len(EXAMPLE_TOPICS) + 1)
with ex_cols[0]:
    st.caption("Try:")
for i, example in enumerate(EXAMPLE_TOPICS):
    with ex_cols[i + 1]:
        if st.button(example, key=f"ex_{i}", use_container_width=True, disabled=busy):
            reset_run()
            st.session_state.topic_input = example
            st.session_state.running = True
            st.session_state.started_at = time.time()
            st.rerun()

# ── Pipeline progress cards ───────────────────────────────────────────────────
def step_status(step: str) -> str:
    r = st.session_state.results
    if st.session_state.error and step not in r and st.session_state.error_step == step:
        return "error"
    if step in r:
        return "done"
    if st.session_state.running:
        for k in STEP_KEYS:
            if k not in r:
                return "running" if k == step else "waiting"
    return "waiting"


def step_card(num, title, status, desc) -> None:
    st.markdown(f"""
    <div class="step-card {status}">
        <div class="step-num">{num}</div>
        <div class="step-title">{title}</div>
        <div class="step-desc">{desc}</div>
        <div class="step-status {status}">{status}</div>
    </div>
    """, unsafe_allow_html=True)


st.session_state.setdefault("error_step", None)

cols = st.columns(4)
for col, (num, title, key, desc) in zip(cols, STEPS):
    with col:
        step_card(num, title, step_status(key), desc)

# ── Trigger a run from the main button ────────────────────────────────────────
if run_btn:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        reset_run()
        st.session_state.topic_input = topic.strip()
        st.session_state.running = True
        st.session_state.started_at = time.time()
        st.rerun()

# ── Run pipeline (one step per rerun) ─────────────────────────────────────────
def run_next_step() -> None:
    """Execute the next unfinished pipeline step. Resets state on failure."""
    results = dict(st.session_state.results)
    topic_val = st.session_state.topic_input

    try:
        if "search" not in results:
            with st.spinner("🔍  Search Agent is gathering information…"):
                agent = get_search_agent()
                resp = agent.invoke({
                    "messages": [("user", f"Find recent, reliable and detailed information about: {topic_val}")]
                })
                results["search"] = resp["messages"][-1].content

        elif "reader" not in results:
            with st.spinner("📄  Reader Agent is scraping top resources…"):
                agent = get_reader_agent()
                resp = agent.invoke({
                    "messages": [("user",
                        f"Based on the following search results about '{topic_val}', "
                        f"pick the most relevant URL and scrape it for deeper content.\n\n"
                        f"Search Results:\n{results['search'][:READER_CONTEXT_CHARS]}"
                    )]
                })
                results["reader"] = resp["messages"][-1].content

        elif "writer" not in results:
            with st.spinner("✍️  Writer is drafting the report…"):
                research_combined = (
                    f"SEARCH RESULTS:\n{results['search']}\n\n"
                    f"DETAILED SCRAPED CONTENT:\n{results['reader']}"
                )
                results["writer"] = writer_chain.invoke({
                    "topic": topic_val,
                    "research": research_combined,
                })

        elif "critic" not in results:
            with st.spinner("🧐  Critic is reviewing the report…"):
                results["critic"] = critic_chain.invoke({"report": results["writer"]})
                st.session_state.running = False
                st.session_state.done = True
                if st.session_state.started_at:
                    st.session_state.elapsed = time.time() - st.session_state.started_at

        st.session_state.results = results
    except Exception as exc:  # noqa: BLE001 — surface any agent/network failure to the UI
        logger.exception("Pipeline step failed")
        # Identify which step failed (first key still missing).
        failed = next((k for k in STEP_KEYS if k not in results), "unknown")
        st.session_state.results = results
        st.session_state.error = str(exc)
        st.session_state.error_step = failed
        st.session_state.running = False
        st.session_state.done = False
    finally:
        st.rerun()


if st.session_state.running and not st.session_state.done:
    run_next_step()

# ── Error display ─────────────────────────────────────────────────────────────
if st.session_state.error:
    st.error(
        f"❌  The **{st.session_state.error_step}** step failed:\n\n"
        f"```\n{st.session_state.error}\n```"
    )
    if st.button("↻  Try again"):
        reset_run()
        st.rerun()

# ── Results display ───────────────────────────────────────────────────────────
r = st.session_state.results

if r:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    head_col, action_col = st.columns([4, 1])
    with head_col:
        st.markdown('<div class="section-heading">Results</div>', unsafe_allow_html=True)
    with action_col:
        if not busy and st.button("✚  New research", use_container_width=True):
            reset_run()
            st.rerun()

    if st.session_state.done and st.session_state.elapsed:
        m1, m2 = st.columns(2)
        m1.metric("Stages completed", f"{len(r)}/4")
        m2.metric("Total time", f"{st.session_state.elapsed:.1f}s")

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
        st.markdown(
            '<div class="report-panel"><div class="panel-label orange">📝 Final Research Report</div>',
            unsafe_allow_html=True,
        )
        st.markdown(r["writer"])
        st.markdown("</div>", unsafe_allow_html=True)

        safe_topic = "".join(
            c if c.isalnum() else "_" for c in st.session_state.topic_input
        ).strip("_")[:40] or "report"
        st.download_button(
            label="⬇  Download Report (.md)",
            data=r["writer"],
            file_name=f"research_{safe_topic}_{int(time.time())}.md",
            mime="text/markdown",
        )

    if "critic" in r:
        st.markdown(
            '<div class="feedback-panel"><div class="panel-label green">🧐 Critic Feedback</div>',
            unsafe_allow_html=True,
        )
        st.markdown(r["critic"])
        st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="notice">
    ResearchMind · Powered by LangChain multi-agent pipeline · Built with Streamlit
</div>
""", unsafe_allow_html=True)
