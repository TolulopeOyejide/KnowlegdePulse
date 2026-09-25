"""
Generation layer, built on LangChain.

Primary path: an open-source model served locally by Ollama (Llama 3,
Mistral, Qwen, etc.), wrapped as a proper LangChain `LLM` so it plugs into
any LangChain prompt/chain.

Fallback path: if Ollama isn't running, answers are composed directly from
the retrieved chunks (no model call at all) so the app always returns a
cited answer instead of failing outright.

Optional backup: OpenAI, via LangChain's `ChatOpenAI`, is wired in below
but commented out. Uncomment it (and set OPENAI_API_KEY) to use a hosted
model instead of / in addition to Ollama — see README.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import requests
from langchain_core.documents import Document
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import PromptTemplate

OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "llama3"

_QA_PROMPT = PromptTemplate.from_template(
    "You are KnowledgePulse, an assistant that answers questions using only "
    "the provided context from internal company documents. Cite the source "
    "name in brackets after any claim you use from it. If the context doesn't "
    "contain the answer, say so plainly instead of guessing.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


class OllamaLLM(LLM):
    """Minimal LangChain LLM wrapper around Ollama's local HTTP API.
    Requires Ollama installed and running, with a model pulled:
        ollama pull llama3
    """

    model: str = OLLAMA_MODEL
    host: str = OLLAMA_HOST

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        resp = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False},
            timeout=600,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()


# ---------------------------------------------------------------------------
# Optional backup: OpenAI via LangChain. Commented out by default so the
# project has zero paid dependencies unless you opt in.
#
#   pip install langchain-openai openai
#   export OPENAI_API_KEY=sk-...
#
# from langchain_openai import ChatOpenAI
#
# def get_openai_llm() -> ChatOpenAI:
#     return ChatOpenAI(model="gpt-4o-mini", temperature=0)
#
# To use it instead of Ollama, swap the provider selection in get_llm()
# below, or try Ollama first and fall back to OpenAI:
#
# def get_llm():
#     try:
#         requests.get(f"{OLLAMA_HOST}/api/tags", timeout=1)
#         return OllamaLLM()
#     except requests.exceptions.RequestException:
#         return get_openai_llm()   # requires OPENAI_API_KEY
# ---------------------------------------------------------------------------


def get_llm() -> Optional[OllamaLLM]:
    """Probe for a running Ollama instance. Returns None (triggering the
    extractive fallback) if it isn't reachable."""
    try:
        requests.get(f"{OLLAMA_HOST}/api/tags", timeout=1)
        return OllamaLLM()
    except requests.exceptions.RequestException:
        return None


def _format_context(docs: List[Document]) -> str:
    return "\n\n".join(f"[{d.metadata.get('source', 'unknown')}] {d.page_content}" for d in docs)


def _extractive_answer(question: str, docs: List[Document]) -> str:
    if not docs:
        return "I couldn't find anything relevant to that in the knowledge base yet."
    top = docs[0]
    lead = f"Based on {top.metadata.get('source', 'the knowledge base')}: {top.page_content.strip()[:600]}"
    other_sources = sorted({d.metadata.get("source", "unknown") for d in docs[1:]})
    if other_sources:
        lead += f" (related: {', '.join(other_sources)})"
    return lead


@dataclass
class Citation:
    source: str
    excerpt: str
    source_type: str
    url: str | None = None


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    provider: str


def answer_question(store, question: str, llm: Optional[OllamaLLM]) -> Answer:
    docs = store.search(question)

    if llm is not None:
        prompt = _QA_PROMPT.format(context=_format_context(docs), question=question)
        text = llm.invoke(prompt)
        provider = "ollama"
    else:
        text = _extractive_answer(question, docs)
        provider = "extractive"

    citations = [
        Citation(
            source=d.metadata.get("source", "unknown"),
            excerpt=d.page_content[:220],
            source_type=d.metadata.get("source_type", "file"),
            url=d.metadata.get("url"),
        )
        for d in docs
    ]
    return Answer(text=text, citations=citations, provider=provider)
