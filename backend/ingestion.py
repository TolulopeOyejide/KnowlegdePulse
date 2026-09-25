"""
Ingestion layer, built on LangChain.

- File loading: LangChain's TextLoader / PyPDFLoader / Docx2txtLoader
  (thin open-source wrappers around pypdf and docx2txt).
- Website loading: LangChain's WebBaseLoader (requests + BeautifulSoup).
- Chunking: LangChain's RecursiveCharacterTextSplitter, which splits on
  paragraph/sentence boundaries before falling back to raw characters —
  better than a naive word-window split for mixed document types.

Every loader returns LangChain `Document` objects (page_content + metadata),
which flow unchanged into the retriever.
"""
from __future__ import annotations

import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

# WebBaseLoader (via requests) 
os.environ.setdefault("USER_AGENT", "KnowledgePulse/1.0")

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx"}

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def _loader_for(path: Path):
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return TextLoader(str(path), autodetect_encoding=True)
    if ext == ".pdf":
        return PyPDFLoader(str(path))
    if ext == ".docx":
        return Docx2txtLoader(str(path))
    raise ValueError(f"Unsupported file type: {ext}")


def load_and_split_file(path: Path) -> list[Document]:
    """Load one uploaded file and split it into retrieval-ready chunks."""
    loader = _loader_for(path)
    raw_docs = loader.load()
    for doc in raw_docs:
        doc.metadata["source"] = path.name
        doc.metadata["source_type"] = "file"
    return _SPLITTER.split_documents(raw_docs)


def load_and_split_url(url: str) -> list[Document]:
    """Fetch a web page and split it into retrieval-ready chunks."""
    loader = WebBaseLoader(web_paths=[url])
    raw_docs = loader.load()
    for doc in raw_docs:
        doc.metadata["source"] = doc.metadata.get("title") or url
        doc.metadata["source_type"] = "url"
        doc.metadata["url"] = url
    return _SPLITTER.split_documents(raw_docs)


def load_and_split_directory(directory: Path) -> list[Document]:
    """Bulk-load every supported file in a directory (used to seed the
    demo knowledge base from data/sample_docs on first boot)."""
    chunks: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            chunks.extend(load_and_split_file(path))
    return chunks
