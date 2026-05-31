"""Central configuration, environment validation, and logging for ResearchMind.

Importing this module is side-effect-light: it loads the .env file and wires up
logging, but performs no network calls. Call ``validate_config()`` to check that
the required API keys are present.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# ── Provider key normalisation ────────────────────────────────────────────────
# langchain-google-genai reads GOOGLE_API_KEY. Some users set GEMINI_API_KEY
# instead, so mirror it across. setdefault never clobbers an existing value.
if os.getenv("GEMINI_API_KEY"):
    os.environ.setdefault("GOOGLE_API_KEY", os.environ["GEMINI_API_KEY"])

# ── Tunable constants ─────────────────────────────────────────────────────────
MODEL_NAME: str = os.getenv("RESEARCHMIND_MODEL", "gemini-2.5-flash")
MODEL_TEMPERATURE: float = float(os.getenv("RESEARCHMIND_TEMPERATURE", "0"))

MAX_SEARCH_RESULTS: int = int(os.getenv("RESEARCHMIND_MAX_RESULTS", "5"))
SCRAPE_TIMEOUT: int = int(os.getenv("RESEARCHMIND_SCRAPE_TIMEOUT", "10"))
SCRAPE_MAX_CHARS: int = int(os.getenv("RESEARCHMIND_SCRAPE_MAX_CHARS", "3000"))
SEARCH_SNIPPET_CHARS: int = 300
READER_CONTEXT_CHARS: int = 800

REQUEST_USER_AGENT: str = (
    "Mozilla/5.0 (compatible; ResearchMind/1.0; +https://github.com/researchmind)"
)

# ── Logging ───────────────────────────────────────────────────────────────────
_LOG_LEVEL = os.getenv("RESEARCHMIND_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, _LOG_LEVEL, logging.INFO),
    format="%(asctime)s · %(levelname)-7s · %(name)s · %(message)s",
    datefmt="%H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger."""
    return logging.getLogger(name)


# ── Environment validation ────────────────────────────────────────────────────
@dataclass(frozen=True)
class ConfigStatus:
    """Result of a configuration check."""

    ok: bool
    missing: tuple[str, ...]

    @property
    def message(self) -> str:
        if self.ok:
            return "All required API keys are configured."
        return "Missing required environment variables: " + ", ".join(self.missing)


def validate_config() -> ConfigStatus:
    """Check that all required API keys are present.

    Returns a :class:`ConfigStatus` instead of raising, so the UI can render a
    friendly banner rather than crashing on a traceback.
    """
    required = {
        "TAVILY_API_KEY": os.getenv("TAVILY_API_KEY"),
        # Either GOOGLE_API_KEY or GEMINI_API_KEY satisfies the Gemini provider.
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
    }
    missing = tuple(name for name, value in required.items() if not value)
    return ConfigStatus(ok=not missing, missing=missing)
