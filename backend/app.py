"""
KnowledgePulse API.

Run with:  uvicorn app:app --reload --port 8000
Then open  http://localhost:8000
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from generation import answer_question, get_llm
from ingestion import load_and_split_directory, load_and_split_file, load_and_split_url
from retrieval import KnowledgeStore

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_PATH = BASE_DIR / "data" / "index.pkl"
SAMPLE_DOCS = BASE_DIR / "data" / "sample_docs"
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="KnowledgePulse")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

store = KnowledgeStore()
llm = get_llm()  # None => extractive fallback, see generation.py


@app.on_event("startup")
def load_or_seed_index():
    if INDEX_PATH.exists():
        store.load(INDEX_PATH)
    elif SAMPLE_DOCS.exists():
        store.add(load_and_split_directory(SAMPLE_DOCS))
        store.save(INDEX_PATH)


class UrlRequest(BaseModel):
    url: str


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[dict]
    provider: str


@app.get("/")
def home():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/ingest/files")
async def ingest_files(files: list[UploadFile]):
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            dest = Path(tmp) / f.filename
            with open(dest, "wb") as out:
                shutil.copyfileobj(f.file, out)
            try:
                chunks = load_and_split_file(dest)
            except Exception as exc:  # noqa: BLE001 - surface to the caller
                raise HTTPException(400, f"Couldn't process {f.filename}: {exc}") from exc
            store.add(chunks)
            results.append({"file": f.filename, "chunks": len(chunks)})
    store.save(INDEX_PATH)
    return {"ingested": results, "total_chunks": len(store)}


@app.post("/ingest/url")
def ingest_url(req: UrlRequest):
    try:
        chunks = load_and_split_url(req.url)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"Couldn't fetch {req.url}: {exc}") from exc
    store.add(chunks)
    store.save(INDEX_PATH)
    return {"ingested": {"url": req.url, "chunks": len(chunks)}, "total_chunks": len(store)}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    result = answer_question(store, req.message, llm)
    return ChatResponse(
        answer=result.text,
        citations=[c.__dict__ for c in result.citations],
        provider=result.provider,
    )


@app.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": len(store), "provider": "ollama" if llm else "extractive"}
