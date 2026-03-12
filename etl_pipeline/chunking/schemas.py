"""
chunking/schemas.py
-------------------
Data classes for the chunking module output.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    """A single chunk of text with provenance metadata."""
    text: str
    method: str                           # line | paragraph | section | context
    chunk_index: int
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    # context mode extras
    similarity_score: float = 0.0         # cosine similarity to topic centroid
    related_chunk_ids: List[str] = field(default_factory=list)


@dataclass
class ContextGroup:
    """
    One semantic topic group produced by BERTopic.

    merged_chunk  — all paragraphs joined as a single SLM-ready block
    source_chunks — the individual paragraphs that belong to this topic
    coherence_score — intra-group cohesion (0–1, higher is better)
    """
    group_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    topic_id: int = -1
    topic_label: str = ""
    topic_words: List[str] = field(default_factory=list)
    merged_chunk: Optional[Chunk] = None
    source_chunks: List[Chunk] = field(default_factory=list)
    coherence_score: float = 0.0


@dataclass
class ChunkingResult:
    """Top-level result returned by Chunker.chunk()."""
    method: str
    chunks: List[Chunk] = field(default_factory=list)
    context_groups: List[ContextGroup] = field(default_factory=list)
    total_chunks: int = 0
    coherence_score: float = 0.0     # overall silhouette score (context mode only)

    # Payload ready to send to the downstream SLM extractor
    @property
    def slm_payload(self) -> Dict[str, Any]:
        if self.method == "context" and self.context_groups:
            return {
                "method": self.method,
                "coherence_score": self.coherence_score,
                "groups": [
                    {
                        "group_id": g.group_id,
                        "topic_label": g.topic_label,
                        "topic_words": g.topic_words,
                        "coherence_score": g.coherence_score,
                        "text": g.merged_chunk.text if g.merged_chunk else "",
                        "source_chunks": [
                            {"chunk_id": c.chunk_id, "text": c.text,
                             "similarity_score": c.similarity_score}
                            for c in g.source_chunks
                        ],
                    }
                    for g in self.context_groups
                ],
            }
        return {
            "method": self.method,
            "total_chunks": self.total_chunks,
            "chunks": [
                {"chunk_id": c.chunk_id, "chunk_index": c.chunk_index, "text": c.text}
                for c in self.chunks
            ],
        }
