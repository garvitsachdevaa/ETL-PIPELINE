"""
chunking/schemas.py
-------------------
Data classes that represent the output of every chunking strategy.
These are downstream-ready for the SLM entity/relation extractor.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class Chunk:
    """
    A single text chunk produced by any chunking strategy.

    chunk_id           — unique identifier (uuid4)
    text               — the actual text content
    method             — one of: 'line' | 'paragraph' | 'section' | 'context'
    chunk_index        — 0-based position in the original document
    similarity_score   — cosine similarity to its topic centroid (context mode)
    related_chunk_ids  — chunk_ids of other chunks sharing the same topic (context mode)
    metadata           — source provenance + optional BERTopic fields:
                           page_number  (binary docs)
                           block_id     (binary docs)
                           block_label  (binary docs)
                           section_id   (text docs)
                           format_type  (text docs)
                           topic_id     (context mode)
                           topic_label  (context mode)
                           topic_words  (context mode — top 5 keywords)
    """
    chunk_id: str
    text: str
    method: str
    chunk_index: int
    similarity_score: float = 0.0
    related_chunk_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "method": self.method,
            "chunk_index": self.chunk_index,
            "metadata": self.metadata,
        }
        if self.similarity_score:
            d["similarity_score"] = round(self.similarity_score, 4)
        if self.related_chunk_ids:
            d["related_chunk_ids"] = self.related_chunk_ids
        return d


@dataclass
class ContextGroup:
    """
    Used exclusively by the 'context' (BERTopic) chunking mode.

    topic_id      — integer topic ID assigned by BERTopic (-1 = outlier)
    topic_label   — human-readable label derived from top keywords
    topic_words   — list of (word, score) tuples from BERTopic
    merged_chunk  — ONE Chunk whose text = all source paragraphs joined;
                    this is what gets sent to the SLM
    source_chunks — the individual paragraph Chunks that were merged;
                    each has similarity_score + related_chunk_ids populated
    """
    topic_id: int
    topic_label: str
    topic_words: List[Any]
    merged_chunk: Optional["Chunk"] = None
    source_chunks: List["Chunk"] = field(default_factory=list)

    # keep .chunks as alias for backwards compat with the UI
    @property
    def chunks(self) -> List["Chunk"]:
        return [self.merged_chunk] if self.merged_chunk else []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "topic_label": self.topic_label,
            "topic_words": [[w, round(s, 4)] for w, s in self.topic_words]
                           if self.topic_words else [],
            "paragraph_count": len(self.source_chunks),
            "merged_chunk": self.merged_chunk.to_dict() if self.merged_chunk else None,
            "source_chunks": [c.to_dict() for c in self.source_chunks],
        }


@dataclass
class ChunkingResult:
    """
    Top-level output returned by Chunker.chunk().

    method          — chunking method used
    total_chunks    — total number of chunks produced
    chunks          — flat list of ALL individual Chunk objects in document order
                      (context mode: individual paragraphs, each tagged with
                       topic_id, similarity_score, related_chunk_ids)
    context_groups  — context mode only: one ContextGroup per topic,
                      each holding a merged_chunk (sent to SLM) and
                      source_chunks (individual paragraphs for traceability)
    slm_payload     — ready-to-send payload for the SLM extractor
    """
    method: str
    total_chunks: int
    chunks: List[Chunk] = field(default_factory=list)
    context_groups: List[ContextGroup] = field(default_factory=list)

    # ── SLM-ready payload ───────────────────────────────────────────────────
    @property
    def slm_payload(self) -> Dict[str, Any]:
        """
        Serialisable dict for the fine-tuned SLM (entity + relation extraction).

        In context mode this includes BOTH:
        - individual_chunks : every paragraph tagged with topic + similarity +
                              related_chunk_ids  (Scenario B)
        - context_groups    : one merged block per topic (Scenario A)
        Together this is Scenario C — full context with full traceability.
        """
        base: Dict[str, Any] = {
            "method": self.method,
            "total_chunks": self.total_chunks,
        }
        if self.context_groups:
            # Context mode: send merged groups + individual chunks separately
            base["context_groups"] = [g.to_dict() for g in self.context_groups]
            base["individual_chunks"] = [c.to_dict() for c in self.chunks]
        else:
            base["chunks"] = [c.to_dict() for c in self.chunks]
        return base

    def to_dict(self) -> Dict[str, Any]:
        return self.slm_payload
