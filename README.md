# KnowledgePulse

A RAG (retrieval-augmented generation) platform, built entirely on open-source
tools. Upload documents or point it at a website, then ask questions in a
chat UI and get answers with citations back to the source.

## Stack

| Layer            | Tool                                                    |
|-------------------|----------------------------------------------------------|
| Orchestration     | LangChain (loaders, splitter, retriever, prompt template) |
| File parsing      | pypdf, docx2txt (via LangChain loaders)                 |
| Website fetching  | LangChain `WebBaseLoader` (requests + BeautifulSoup)    |
| Chunking          | LangChain `RecursiveCharacterTextSplitter`              |
| Retrieval         | LangChain `TFIDFRetriever` (scikit-learn TF-IDF — offline, no model download) |
| Generation        | Local open-source LLM via **Ollama**, with a zero-dependency extractive fallback |
| Backend           | FastAPI                                                  |
| Frontend          | Single HTML/JS page — file upload, website ingestion, chat |

Everything above is free and runs locally. No API key is required out of the box.

## Setup

```bash
pip install -r requirements.txt
```

### Run

```bash
cd backend
uvicorn app:app --reload --port 8000
```

Open **http://localhost:8000** — a demo knowledge base (3 sample HR policy
docs in `data/sample_docs/`) is indexed automatically on first run.

### Using it

- **Upload documents**: drag `.txt`, `.md`, `.pdf`, or `.docx` files onto the
  sidebar, or click to browse.
- **Add a website**: paste a URL into the sidebar and click Add — the page
  is fetched, parsed, and indexed the same way as an uploaded file.
- **Chat**: ask a question in the main panel. Each answer shows which
  document or URL it drew from, as clickable-looking source pills.

## Enabling a real LLM (Ollama — recommended, still free)

By default there's no LLM running, so answers are composed directly from the
retrieved text ("extractive" mode — always works, lower fluency). To get
fluent, synthesized answers with a real open-source model:

1. Install [Ollama](https://ollama.com)
2. Pull a model: `ollama pull llama3`
3. Start Ollama (it runs a local server on port 11434)
4. Restart the KnowledgePulse backend — it auto-detects Ollama on startup
   and switches providers, no config needed

Check `GET /health` to see which provider is active.

## Optional backup: OpenAI

`backend/generation.py` has a commented-out `ChatOpenAI` integration for
using a hosted OpenAI model as a backup or alternative to Ollama:

```bash
pip install langchain-openai openai
export OPENAI_API_KEY=sk-...
```

Then uncomment the `get_openai_llm()` block and the try/except in `get_llm()`
in `backend/generation.py` to fail over from Ollama to OpenAI automatically.
This is optional and off by default, the project has no paid dependencies
unless you opt in here.

## Architecture

```
Documents/URLs → LangChain loaders → RecursiveCharacterTextSplitter
              → TFIDFRetriever (index)
                     ↓
Question → TFIDFRetriever.search() → top-k chunks
                     ↓
         Ollama LLM (or extractive fallback) → cited answer
```

## Swapping components later

Every module talks through a narrow interface, so upgrading a piece doesn't
touch the rest:

- **Retrieval**: swap `TFIDFRetriever` in `backend/retrieval.py` for an
  embeddings-based retriever (e.g. `sentence-transformers` + FAISS/Chroma/
  Qdrant) once you have model hosting in place — better for semantic
  (rather than keyword) matching.
- **Generation**: swap or add providers in `backend/generation.py` (Ollama
  model name, OpenAI, or any other LangChain-compatible LLM).
- **Frontend**: `frontend/index.html` is a single self-contained file, no
  build step, so it's easy to restyle or extend with the current chat
  history, a Slack bot, etc.

## Project layout

```
knowledgepulse/
├── backend/
│   ├── app.py          # FastAPI routes: /ingest/files, /ingest/url, /chat, /health
│   ├── ingestion.py     # LangChain loaders + text splitter
│   ├── retrieval.py      # LangChain TFIDFRetriever wrapper
│   └── generation.py     # Ollama LLM wrapper, extractive fallback, OpenAI backup (commented)
├── frontend/
│   └── index.html       # Upload + website + chat UI
├── data/
│   └── sample_docs/     # Seed documents for first run
└── requirements.txt
```
