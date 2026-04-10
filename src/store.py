from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        # In tests we usually inject a deterministic embedding function.
        # Keep that mode fully in-memory to avoid persistent Chroma state.
        if embedding_fn is not None:
            return

        try:
            import chromadb

            self._client = chromadb.Client()
            self._collection = self._client.get_or_create_collection(name=collection_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        """Build one normalized in-memory record."""
        return {
            "id": f"id_{self._next_index}",
            "doc_id": doc.id,
            "content": doc.content,
            "embedding": self._embedding_fn(doc.content),
            "metadata": dict(doc.metadata or {}),
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        """Run in-memory similarity search over records."""
        query_vec = self._embedding_fn(query)
        scored_records: list[dict[str, Any]] = []
        for rec in records:
            score = _dot(query_vec, rec["embedding"])
            scored_records.append(
                {
                    "id": rec["id"],
                    "doc_id": rec["doc_id"],
                    "content": rec["content"],
                    "metadata": rec["metadata"],
                    "score": score,
                }
            )
        scored_records.sort(key=lambda x: x["score"], reverse=True)
        return scored_records[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if not docs:
            return

        if self._use_chroma:
            ids = [f"id_{i + self._next_index}" for i in range(len(docs))]
            contents = [doc.content for doc in docs]
            embeddings = [self._embedding_fn(c) for c in contents]
            metadatas = [{**(doc.metadata or {}), "doc_id": doc.id} for doc in docs]

            self._collection.add(
                ids=ids,
                documents=contents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        else:
            for doc in docs:
                record = self._make_record(doc)
                self._store.append(record)
                self._next_index += 1

        if self._use_chroma:
            self._next_index += len(docs)

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        if self._use_chroma:
            query_vec = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_vec],
                n_results=top_k,
            )
            formatted: list[dict[str, Any]] = []
            ids = results.get("ids", [[]])[0]
            docs = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            for i in range(len(ids)):
                distance = distances[i] if i < len(distances) else 0.0
                metadata = metadatas[i] if i < len(metadatas) else {}
                formatted.append(
                    {
                        "id": ids[i],
                        "doc_id": (metadata or {}).get("doc_id"),
                        "content": docs[i],
                        "metadata": metadata or {},
                        "score": -float(distance),
                    }
                )
            formatted.sort(key=lambda x: x["score"], reverse=True)
            return formatted

        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        if not metadata_filter:
            return self.search(query, top_k)

        if self._use_chroma:
            query_vec = self._embedding_fn(query)
            results = self._collection.query(
                query_embeddings=[query_vec],
                n_results=top_k,
                where=metadata_filter,
            )
            formatted: list[dict[str, Any]] = []
            ids = results.get("ids", [[]])[0]
            docs = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            for i in range(len(ids)):
                distance = distances[i] if i < len(distances) else 0.0
                metadata = metadatas[i] if i < len(metadatas) else {}
                formatted.append(
                    {
                        "id": ids[i],
                        "doc_id": (metadata or {}).get("doc_id"),
                        "content": docs[i],
                        "metadata": metadata or {},
                        "score": -float(distance),
                    }
                )
            formatted.sort(key=lambda x: x["score"], reverse=True)
            return formatted[:top_k]

        filtered_records = [
            rec
            for rec in self._store
            if all(rec["metadata"].get(k) == v for k, v in metadata_filter.items())
        ]
        return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        initial_count = self.get_collection_size()

        if self._use_chroma:
            self._collection.delete(where={"doc_id": doc_id})
        else:
            self._store = [r for r in self._store if r["doc_id"] != doc_id]

        return self.get_collection_size() < initial_count
