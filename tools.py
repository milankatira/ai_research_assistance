"""LangChain tools for web search (Tavily) and URL scraping (requests + bs4)."""
from __future__ import annotations

import os

import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from tavily import TavilyClient
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    MAX_SEARCH_RESULTS,
    REQUEST_USER_AGENT,
    SCRAPE_MAX_CHARS,
    SCRAPE_TIMEOUT,
    SEARCH_SNIPPET_CHARS,
    get_logger,
)

logger = get_logger(__name__)

# Lazily constructed so importing this module never requires a valid key.
_tavily_client: TavilyClient | None = None


def _get_tavily_client() -> TavilyClient:
    global _tavily_client
    if _tavily_client is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise RuntimeError(
                "TAVILY_API_KEY is not set. Add it to your .env file."
            )
        _tavily_client = TavilyClient(api_key=api_key)
    return _tavily_client


@tool
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=8))
def web_search(query: str) -> str:
    """Search the web for recent, reliable information on a topic.

    Returns the title, URL, and a short snippet for each result.
    """
    logger.info("web_search: %s", query)
    response = _get_tavily_client().search(query, max_results=MAX_SEARCH_RESULTS)
    results = response.get("results", []) if isinstance(response, dict) else []

    if not results:
        return f"No search results found for: {query}"

    blocks = [
        (
            f"Title: {r.get('title', 'Untitled')}\n"
            f"URL: {r.get('url', 'N/A')}\n"
            f"Snippet: {r.get('content', '')[:SEARCH_SNIPPET_CHARS]}..."
        )
        for r in results
    ]
    return "\n--------------------------------\n".join(blocks)


@tool
def scrape_url(url: str) -> str:
    """Scrape and return clean text content from a given URL for deeper reading."""
    logger.info("scrape_url: %s", url)
    try:
        resp = requests.get(
            url,
            timeout=SCRAPE_TIMEOUT,
            headers={"User-Agent": REQUEST_USER_AGENT},
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("scrape_url failed for %s: %s", url, exc)
        return f"Could not scrape URL ({url}): {exc}"

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    if not text:
        return f"No readable text content found at {url}."
    return text[:SCRAPE_MAX_CHARS]
