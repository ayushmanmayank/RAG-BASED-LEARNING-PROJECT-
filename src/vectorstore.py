"""Vector store for chronology-scoped concept retrieval.

Stores ONLY embeddings + metadata (playlist index, timestamps, concept id, tag) - never
the transcript or concept text itself. Text is re-fetched live by rag_engine when an
answer actually needs the words; this store only ever answers "which video/timestamp is
relevant", never "what was said there".
"""
from pathlib import Path
from typing import List, Optional, Union

import chromadb
from sentence_transformers import SentenceTransformer

from src import config
from src.concept_extractor import Concept

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    return _model


class VectorStore:
    def __init__(self, path: Optional[Union[str, Path]] = None):
        self._client = chromadb.PersistentClient(path=str(path or config.CHROMA_DIR))
        self._collection = self._client.get_or_create_collection(
            name="concepts", metadata={"hnsw:space": "cosine"}
        )

    def add_concepts(self, video_id: str, playlist_index: int, concepts: List[Concept]) -> None:
        if not concepts:
            return
        texts = [c.point for c in concepts]
        embeddings = _get_model().encode(texts, show_progress_bar=False).tolist()
        ids = [f"{video_id}:{c.id}" for c in concepts]
        metadatas = [
            {
                "video_id": video_id,
                "playlist_index": playlist_index,
                "concept_id": c.id,
                "tag": c.tag,
                "start_seconds": c.start_seconds,
                "end_seconds": c.end_seconds,
            }
            for c in concepts
        ]
        self._collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def search(self, query: str, max_playlist_index: int, n_results: int = 6) -> List[dict]:
        embedding = _get_model().encode([query], show_progress_bar=False).tolist()
        result = self._collection.query(
            query_embeddings=embedding,
            n_results=n_results,
            where={"playlist_index": {"$lte": max_playlist_index}},
        )
        hits = []
        metadatas = result.get("metadatas") or [[]]
        distances = result.get("distances") or [[]]
        for meta, distance in zip(metadatas[0], distances[0]):
            hits.append({**meta, "distance": distance})
        return hits

    def has_video(self, video_id: str) -> bool:
        existing = self._collection.get(where={"video_id": video_id}, limit=1)
        return bool(existing["ids"])
