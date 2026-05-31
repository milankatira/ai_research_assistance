# 🧠 ResearchMind — AI Research Assistant

> A multi-agent AI pipeline that searches the web, reads sources, writes structured research reports, and critiques its own output — powered by Google Gemini and LangChain.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?style=flat-square&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-1C3C3C?style=flat-square)
![Ollama](https://img.shields.io/badge/Ollama-Gemma3-black?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## ✨ Overview

**ResearchMind** automates the full research workflow in four sequential agent stages:

| Stage | Agent | What it does |
|-------|-------|-------------|
| 1 | 🔍 **Search Agent** | Queries the web via Tavily to find recent, relevant sources |
| 2 | 📄 **Reader Agent** | Picks the best URL and scrapes it for deep contextual content |
| 3 | ✍️ **Writer Chain** | Drafts a structured report: Introduction → Key Findings → Conclusion → Sources |
| 4 | 🧐 **Critic Chain** | Scores the report (X/10), lists strengths, areas to improve, and a one-line verdict |

You can run it as a **Streamlit web app** (interactive UI) or directly as a **CLI script**.

> **LLM runs 100% locally** via [Ollama](https://ollama.com) — no cloud API key or internet connection needed for inference. Only web search (Tavily) requires an API key.

---

## 🖥️ Screenshots

| Interactive UI | Report Output |
|---|---|
| *Dark-themed Streamlit app with step-by-step progress cards* | *Structured markdown report with critic feedback panel* |

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/your-username/ai_research_assistance.git
cd ai_research_assistance
```

### 2. Set up a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

Open `.env` and fill in your Tavily key:

```env
TAVILY_API_KEY=your-tavily-key-here
```

> **Get your key:** 🔑 **Tavily** — [https://app.tavily.com](https://app.tavily.com) (free tier available)
>
> No Gemini / Google key needed — the LLM runs locally!

### 5. Pull the model and start Ollama

```bash
ollama pull gemma3
# Ollama server usually starts automatically; if not:
ollama serve
```

### 6. Run the app

**Web UI (recommended):**
```bash
streamlit run app.py
```

**CLI mode:**
```bash
python pipeline.py
```

---

## 📁 Project Structure

```
ai_research_assistance/
├── app.py            # Streamlit web UI — main entry point
├── agents.py         # LangChain agent & chain factory functions
├── pipeline.py       # CLI pipeline runner (no Streamlit dependency)
├── tools.py          # LangChain @tool definitions (web_search, scrape_url)
├── config.py         # Central config, constants, and environment validation
├── requirements.txt  # Python dependencies
├── .env.example      # Template for required environment variables
└── .gitignore
```

### File Responsibilities

- **[`app.py`](app.py)** — Streamlit UI, session state management, step-by-step progress rendering, report and critic output display.
- **[`agents.py`](agents.py)** — Builds and caches the Search Agent, Reader Agent, Writer Chain, and Critic Chain using LangChain + Gemini.
- **[`pipeline.py`](pipeline.py)** — Standalone script that runs all 4 stages sequentially with terminal output. No Streamlit required.
- **[`tools.py`](tools.py)** — `web_search` (Tavily, with retry/exponential backoff) and `scrape_url` (BeautifulSoup, removes noise tags).
- **[`config.py`](config.py)** — Loads `.env`, exposes typed constants, configures Ollama base URL, and provides `validate_config()` (only checks `TAVILY_API_KEY`).

---

## ⚙️ Configuration

All settings are controlled via environment variables. Defaults are shown below:

| Variable | Default | Description |
|---|---|---|
| `TAVILY_API_KEY` | *(required)* | Tavily search API key |
| `RESEARCHMIND_MODEL` | `gemma3` | Ollama model to use |
| `RESEARCHMIND_TEMPERATURE` | `0` | LLM temperature (0 = deterministic) |
| `RESEARCHMIND_MAX_RESULTS` | `5` | Number of Tavily search results |
| `RESEARCHMIND_SCRAPE_TIMEOUT` | `10` | HTTP timeout (seconds) for scraping |
| `RESEARCHMIND_SCRAPE_MAX_CHARS` | `3000` | Max characters extracted per page |
| `RESEARCHMIND_LOG_LEVEL` | `INFO` | Python logging level |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address (change if running remotely) |

> **Tip:** You can switch to any model available in your local Ollama install (e.g. `llama3`, `mistral`, `phi4`) by setting `RESEARCHMIND_MODEL` in your `.env` file.

---

## 🛠️ How It Works

```
User Topic
    │
    ▼
┌──────────────┐
│ Search Agent │  → Tavily web search → top 5 results (title, URL, snippet)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Reader Agent │  → Selects best URL → scrapes full page content
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Writer Chain │  → Gemini prompt → structured Markdown report
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Critic Chain │  → Scores report (X/10), strengths, improvements, verdict
└──────────────┘
```

### Agent Details

**Search Agent**
- Tool: `web_search` via Tavily Python client
- Retries up to 3 times with exponential backoff on failure
- Returns formatted blocks: Title / URL / Snippet (≤ 300 chars each)

**Reader Agent**
- Tool: `scrape_url` via `requests` + BeautifulSoup
- Strips `<script>`, `<style>`, `<nav>`, `<footer>`, `<header>`, `<aside>`, `<form>` tags
- Returns clean plain text (up to 3,000 characters)
- Custom `User-Agent: ResearchMind/1.0`

**Writer Chain**
- Prompt persona: *"expert research writer"*
- Output sections: Introduction · Key Findings (min 3) · Conclusion · Sources
- Powered by **Gemma3 via Ollama** (local inference, no API calls)

**Critic Chain**
- Prompt persona: *"sharp and constructive research critic"*
- Output format: `Score: X/10` · Strengths · Areas to Improve · One-line verdict
- Powered by **Gemma3 via Ollama** (local inference, no API calls)

---

## 🎨 UI Features

- **Dark theme** with orange accent colors (`#ff8c32`)
- **Custom typography**: Syne (headings), DM Mono (code), DM Sans (body)
- **Step progress cards** with waiting / running / done / error states
- **Collapsible panels** for raw search results and scraped content
- **Styled report viewer** with a **Markdown download button**
- **Critic feedback panel** with score display
- **Elapsed time tracker** showing total research duration

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `langchain` + `langchain-core` + `langchain-community` | Agent orchestration |
| `langchain-ollama` | Ollama LLM integration |
| `tavily-python` | Web search API client |
| `beautifulsoup4` + `lxml` + `html5lib` | HTML parsing and scraping |
| `requests` | HTTP client for web scraping |
| `python-dotenv` | `.env` file loading |
| `tenacity` | Retry logic with backoff |
| `pydantic` | Data validation |
| `rich` | Terminal output formatting (CLI mode) |

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

1. **Fork** the repository
2. **Create a branch**: `git checkout -b feature/your-feature-name`
3. **Commit your changes**: `git commit -m "feat: add your feature"`
4. **Push to your branch**: `git push origin feature/your-feature-name`
5. **Open a Pull Request**

Please keep code style consistent and add comments for any non-obvious logic.

---

## 📋 Roadmap

- [ ] Support for multiple URLs per research topic (parallel reading)
- [ ] Export reports as PDF
- [ ] Configurable number of search results in the UI
- [ ] Chat-style follow-up questions on the generated report
- [ ] Support for additional LLM providers (OpenAI, Anthropic)
- [ ] Persistent research history and session management

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [LangChain](https://www.langchain.com/) for the agent framework
- [Ollama](https://ollama.com/) for local LLM inference
- [Gemma](https://ai.google.dev/gemma) by Google DeepMind for the open-weights model
- [Tavily](https://www.tavily.com/) for AI-optimized web search
- [Streamlit](https://streamlit.io/) for the rapid UI development experience
