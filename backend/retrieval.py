"""
Retrieval layer, built on LangChain's TFIDFRetriever.

TF-IDF (scikit-learn under the hood) is used instead of a neural embedding
model so the whole platform runs offline with zero model downloads and no
GPU. It's a real, LangChain-native retriever — swap it for an embeddings +
FAISS/Chroma/Qdrant retriever later without touching any other file; every
other module only talks to the `KnowledgeStore` interface below.
"""
from __future__ import annotations

import pickle
from pathlib import Path

from langchain_community.retrievers import TFIDFRetriever
from langchain_core.documents import Document


class KnowledgeStore:
    def __init__(self, k: int = 5):
        self.k = k
        self.documents: list[Document] = []
        self._retriever: TFIDFRetriever | None = None

    def _rebuild(self) -> None:
        if self.documents:
            self._retriever = TFIDFRetriever.from_documents(self.documents, k=self.k)
        else:
            self._retriever = None

    def add(self, docs: list[Document]) -> None:
        self.documents.extend(docs)
        self._rebuild()

    def search(self, query: str) -> list[Document]:
        if self._retriever is None:
            return []
        return self._retriever.invoke(query)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            self.documents = pickle.load(f)
        self._rebuild()

    def __len__(self) -> int:
        return len(self.documents)
