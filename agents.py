"""Agent and chain factory for the ResearchMind pipeline.

Four stages: search agent, reader agent, writer chain, critic chain. The LLM is
built once and reused across all stages.

LLM backend: Ollama (local) running gemma3. Make sure Ollama is running and the
model is pulled before starting the app::

    ollama pull gemma3
    ollama serve          # already running if you opened the Ollama app

"""
from __future__ import annotations

from functools import lru_cache

from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

from config import MODEL_NAME, MODEL_TEMPERATURE, get_logger
from tools import scrape_url, web_search

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    """Return a shared Ollama chat model (built once, cached)."""
    logger.info("Initialising LLM: %s (temp=%s)", MODEL_NAME, MODEL_TEMPERATURE)
    return ChatOllama(model=MODEL_NAME, temperature=MODEL_TEMPERATURE)


# ── Agents ────────────────────────────────────────────────────────────────────
def build_search_agent():
    """Agent 1 — gathers recent web information via Tavily search."""
    return create_agent(model=get_llm(), tools=[web_search])


def build_reader_agent():
    """Agent 2 — scrapes the most relevant URL for deeper content."""
    return create_agent(model=get_llm(), tools=[scrape_url])


# ── Writer chain ──────────────────────────────────────────────────────────────
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | get_llm() | StrOutputParser()

# ── Critic chain ──────────────────────────────────────────────────────────────
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | get_llm() | StrOutputParser()
