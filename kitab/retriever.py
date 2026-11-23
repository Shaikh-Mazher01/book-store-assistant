"""Index building and hybrid retrieval: vector + BM25, fused with RRF, optional rerank."""
import json
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document

from .config import COLLECTION, EMBED_MODEL, FETCH_K, RRF_K
from .query import STOPWORDS, matches, to_chroma, tokenize


def _safe_metadata(meta: dict) -> dict:
    return {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}


def load_documents(path: Path) -> list[Document]:
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            meta = _safe_metadata(row.get("metadata", {}))
            meta["id"] = row["id"]
            docs.append(Document(page_content=row["text"], metadata=meta, id=row["id"]))
    return docs


def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})


def open_store(db_dir: Path, embeddings):
    from langchain_chroma import Chroma
    return Chroma(embedding_function=embeddings, persist_directory=str(db_dir),
                  collection_name=COLLECTION)


def build_index(docs: list[Document], db_dir: Path, embeddings,
                batch_size: int = 256, on_progress: Callable[[int, int], None] | None = None):
    store = open_store(db_dir, embeddings)
    for start in range(0, len(docs), batch_size):
        batch = docs[start:start + batch_size]
        store.add_documents(batch, ids=[d.metadata["id"] for d in batch])
        if on_progress:
            on_progress(min(start + batch_size, len(docs)), len(docs))
    return store


class HybridRetriever:
    def __init__(self, store, docs: list[Document], reranker_loader: Callable | None = None):
        self.store, self.docs = store, docs
        self._reranker_loader = reranker_loader
        self._bm25 = None

    @property
    def bm25(self):
        if self._bm25 is None:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([tokenize(d.page_content) for d in self.docs])
        return self._bm25

    def _vector(self, query, f):
        return self.store.similarity_search(query, k=FETCH_K, filter=to_chroma(f))

    def _bm25_search(self, query, f):
        tokens = [t for t in tokenize(query) if t not in STOPWORDS]
        if not tokens:
            return []
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
        out = []
        for i in ranked:
            if scores[i] <= 0:
                break
            if matches(self.docs[i].metadata, f):
                out.append(self.docs[i])
                if len(out) == FETCH_K:
                    break
        return out

    @staticmethod
    def _rrf(result_lists):
        scores, by_id = {}, {}
        for results in result_lists:
            for rank, doc in enumerate(results):
                did = doc.metadata.get("id") or doc.page_content
                by_id[did] = doc
                scores[did] = scores.get(did, 0.0) + 1.0 / (RRF_K + rank + 1)
        return [by_id[i] for i in sorted(scores, key=scores.get, reverse=True)]

    def invoke(self, query, filters, k=5, rerank=False):
        fused = self._rrf([self._vector(query, filters), self._bm25_search(query, filters)])
        if rerank and fused and self._reranker_loader:
            fused = fused[:FETCH_K]
            scores = self._reranker_loader().predict([(query, d.page_content) for d in fused])
            fused = [d for _, d in sorted(zip(scores, fused), key=lambda x: x[0], reverse=True)]
        return fused[:k]
