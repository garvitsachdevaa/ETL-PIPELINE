"""
chunking/strategies.py
----------------------
Four chunking strategies exposed as pure functions.

All strategies accept a list of (text, source_metadata) tuples — the
"segments" extracted from a document — and return a flat list of Chunk
objects.  The caller (chunker.py) is responsible for building those
segments from either a TextDocument or a BinaryDocument.

Strategies
----------
chunk_by_line      — split each segment on newline boundaries
chunk_by_paragraph — split each segment on blank-line boundaries
chunk_by_section   — split on detected structural headers / dividers
chunk_by_context   — paragraph-level base split → sentence-transformers
                     embeddings → BERTopic topic assignment →
                     ContextGroup clustering
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Dict, List, Tuple

from chunking.schemas import Chunk, ContextGroup

logger = logging.getLogger(__name__)

# ── Type alias ───────────────────────────────────────────────────────────────
# Each segment is (text_content, source_metadata_dict)
Segment = Tuple[str, Dict[str, Any]]


# ═══════════════════════════════════════════════════════════════════════════
# 1. LINE CHUNKING
# ═══════════════════════════════════════════════════════════════════════════

def chunk_by_line(segments: List[Segment]) -> List[Chunk]:
    """
    Split every segment on newline boundaries.
    Each non-empty line becomes one Chunk.
    """
    chunks: List[Chunk] = []
    idx = 0

    for text, meta in segments:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                text=line,
                method="line",
                chunk_index=idx,
                metadata={**meta},
            ))
            idx += 1

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# 2. PARAGRAPH CHUNKING
# ═══════════════════════════════════════════════════════════════════════════

def chunk_by_paragraph(segments: List[Segment]) -> List[Chunk]:
    """
    Split every segment on blank-line boundaries (one or more \\n).
    Each non-empty paragraph becomes one Chunk.
    """
    chunks: List[Chunk] = []
    idx = 0

    for text, meta in segments:
        for para in re.split(r"\n{2,}", text):
            para = para.strip()
            if not para:
                continue
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                text=para,
                method="paragraph",
                chunk_index=idx,
                metadata={**meta},
            ))
            idx += 1

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# 3. SECTION CHUNKING
# ═══════════════════════════════════════════════════════════════════════════

# Patterns that indicate a new section boundary
_SECTION_PATTERN = re.compile(
    r"(?m)^(?:"
    r"\d+[\.\)]\s+\S"          # "1. Intro" or "2) Methods"
    r"|#{1,4}\s+\S"             # Markdown headings  ## Title
    r"|[A-Z][A-Z\s]{3,}$"      # ALL-CAPS lines
    r"|(?:Chapter|Section|Part|Appendix)\s+\S"  # Explicit section keywords
    r"|={3,}|-{3,}|_{3,}"      # Horizontal rules
    r")"
)


def chunk_by_section(segments: List[Segment]) -> List[Chunk]:
    """
    Split every segment at detected structural headers / dividers.
    Each section (header + body) becomes one Chunk.
    Falls back to paragraph splitting if no structural markers are found.
    """
    chunks: List[Chunk] = []
    idx = 0

    for text, meta in segments:
        # Find all header positions
        boundaries = [m.start() for m in _SECTION_PATTERN.finditer(text)]

        if not boundaries:
            # No structural markers → fall back to paragraph split
            for para in re.split(r"\n{2,}", text):
                para = para.strip()
                if para:
                    chunks.append(Chunk(
                        chunk_id=str(uuid.uuid4()),
                        text=para,
                        method="section",
                        chunk_index=idx,
                        metadata={**meta, "section_detected": False},
                    ))
                    idx += 1
            continue

        # Include text before the first boundary as a preamble section
        boundaries = [0] + boundaries

        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
            section_text = text[start:end].strip()
            if not section_text:
                continue
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                text=section_text,
                method="section",
                chunk_index=idx,
                metadata={**meta, "section_detected": True, "section_number": i},
            ))
            idx += 1

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# 4. CONTEXT CHUNKING  (BERTopic)
# ═══════════════════════════════════════════════════════════════════════════

_MIN_CHUNKS_FOR_BERTOPIC = 5   # BERTopic needs a reasonable corpus


def _target_nr_topics(num_paragraphs: int, granularity: str) -> int | str:
    """
    Map granularity + corpus size → desired number of topic groups.

    BERTopic is always fitted at maximum granularity (min_topic_size=2),
    then reduce_topics() merges down to this target.  This is the only
    reliable control lever: UMAP flattens the embedding space before
    HDBSCAN runs, making cluster_selection_epsilon essentially useless.

    Fractions of num_paragraphs (floored, min 2):
        Very Fine  → 60%  e.g. 12 para → 7 groups
        Fine       → 40%  e.g. 12 para → 4 groups
        Auto       → BERTopic decides (no reduction)
        Broad      → 25%  e.g. 12 para → 3 groups
        Very Broad → 15%  e.g. 12 para → 2 groups
    """
    if granularity == "Auto":
        return "auto"
    fracs = {
        "Very Fine":  0.60,
        "Fine":       0.40,
        "Broad":      0.25,
        "Very Broad": 0.15,
    }
    frac = fracs.get(granularity, 0.40)
    return max(2, int(num_paragraphs * frac))


def chunk_by_context(
    segments: List[Segment],
    embedding_model_name: str = "all-MiniLM-L6-v2",
    granularity: str = "Auto",
) -> Tuple[List[Chunk], List[ContextGroup]]:
    """
    Semantic context chunking — Scenario C (individual + merged).

    Pipeline
    --------
    1.  Paragraph-level base split across all segments
    2.  sentence-transformers embeddings (one vector per paragraph)
    3.  BERTopic topic assignment per paragraph
    4.  Cosine similarity of each paragraph to its topic centroid
    5.  Tag every individual paragraph Chunk with:
            topic_id, topic_label, topic_words,
            similarity_score, related_chunk_ids
    6.  MERGE all paragraphs of the same topic → one merged Chunk
        (joined with blank-line separator; source_paragraph_indices kept)
    7.  Build ContextGroup per topic:
            merged_chunk  = the one big SLM-ready block
            source_chunks = individual paragraphs with full provenance

    Returns
    -------
    (individual_chunks, context_groups)
    - individual_chunks : all paragraphs in original document order,
                          each tagged with topic + similarity + related IDs
    - context_groups    : one group per topic, each having:
                              merged_chunk  (what the SLM receives)
                              source_chunks (drill-down traceability)
    """
    import numpy as np

    # ── Step 1: base paragraph split ────────────────────────────────────────
    base_chunks: List[Chunk] = chunk_by_paragraph(segments)

    if len(base_chunks) < _MIN_CHUNKS_FOR_BERTOPIC:
        logger.warning(
            "chunk_by_context: only %d base chunks — "
            "BERTopic needs ≥%d; falling back to paragraph chunking.",
            len(base_chunks), _MIN_CHUNKS_FOR_BERTOPIC,
        )
        for c in base_chunks:
            c.method = "context"
            c.metadata.update({"topic_id": 0, "topic_label": "context_0 (fallback)",
                                "topic_words": [], "bertopic_used": False})
        fallback_group = ContextGroup(
            topic_id=0,
            topic_label="context_0 (fallback)",
            topic_words=[],
            merged_chunk=Chunk(
                chunk_id=str(uuid.uuid4()),
                text="\n\n".join(c.text for c in base_chunks),
                method="context",
                chunk_index=0,
                metadata={"topic_id": 0, "topic_label": "context_0 (fallback)",
                           "bertopic_used": False, "paragraph_count": len(base_chunks),
                           "source_paragraph_indices": list(range(len(base_chunks)))},
            ),
            source_chunks=base_chunks,
        )
        return base_chunks, [fallback_group]

    # ── Step 2: embeddings ──────────────────────────────────────────────────
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for context chunking. "
            "Install with: pip install sentence-transformers"
        ) from exc

    try:
        from bertopic import BERTopic
    except ImportError as exc:
        raise ImportError(
            "bertopic is required for context chunking. "
            "Install with: pip install bertopic"
        ) from exc

    texts = [c.text for c in base_chunks]

    logger.info("chunk_by_context: loading embedding model '%s'…", embedding_model_name)
    st_model = SentenceTransformer(embedding_model_name)
    embeddings = np.array(st_model.encode(texts, show_progress_bar=False))

    # ── Step 3: BERTopic — always fit at max granularity, then reduce ─────────
    nr_topics_target = _target_nr_topics(len(base_chunks), granularity)
    logger.info(
        "chunk_by_context: granularity='%s' → target nr_topics=%s on %d paragraphs",
        granularity, nr_topics_target, len(texts),
    )
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(
        stop_words="english",
        min_df=1,
        ngram_range=(1, 2),
    )
    # Always fit at max granularity (min_topic_size=2) — reduce_topics() handles
    # merging afterward.  This gives BERTopic the best chance to find all real
    # topics first, then we collapse them to the desired count.
    topic_model = BERTopic(
        embedding_model=st_model,
        vectorizer_model=vectorizer,
        min_topic_size=2,
        nr_topics="auto",
        verbose=False,
        calculate_probabilities=False,
    )
    topics, _ = topic_model.fit_transform(texts, embeddings)

    # Reduce if the user asked for fewer topics than BERTopic produced
    unique_real_topics = [t for t in set(topics) if t != -1]
    if nr_topics_target != "auto" and len(unique_real_topics) > nr_topics_target:
        logger.info(
            "chunk_by_context: reducing %d topics → %d via reduce_topics()",
            len(unique_real_topics), nr_topics_target,
        )
        topics, _ = topic_model.reduce_topics(texts, nr_topics=nr_topics_target)
    elif nr_topics_target != "auto" and len(unique_real_topics) <= nr_topics_target:
        logger.info(
            "chunk_by_context: already at %d topics ≤ target %d, no reduction needed",
            len(unique_real_topics), nr_topics_target,
        )

    # ── Step 4: compute topic centroids + cosine similarity ─────────────────
    # Group embedding indices by topic
    topic_indices: Dict[int, List[int]] = {}
    for idx, tid in enumerate(topics):
        topic_indices.setdefault(tid, []).append(idx)

    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    # centroid per topic = mean of all its paragraph embeddings
    topic_centroids: Dict[int, np.ndarray] = {
        tid: embeddings[idxs].mean(axis=0)
        for tid, idxs in topic_indices.items()
    }
    # cosine similarity of each paragraph to its topic centroid
    similarities: List[float] = [
        _cosine(embeddings[i], topic_centroids[topics[i]])
        for i in range(len(base_chunks))
    ]

    # ── Step 5: tag individual paragraph chunks ──────────────────────────────
    # First pass: collect chunk_ids per topic for related_chunk_ids
    topic_chunk_ids: Dict[int, List[str]] = {}
    for chunk, tid in zip(base_chunks, topics):
        chunk.method = "context"
        topic_chunk_ids.setdefault(tid, []).append(chunk.chunk_id)

    raw_words_by_topic: Dict[int, List] = {
        tid: (topic_model.get_topic(tid) or [])
        for tid in set(topics)
    }
    label_by_topic: Dict[int, str] = {}
    for tid, words in raw_words_by_topic.items():
        kws = [w for w, _ in words[:3]]
        label_by_topic[tid] = (
            "context_" + "_".join(kws) if kws
            else ("context_outlier" if tid == -1 else f"context_{tid}")
        )

    # Second pass: attach all metadata to individual chunks
    for i, (chunk, tid) in enumerate(zip(base_chunks, topics)):
        raw_words = raw_words_by_topic[tid]
        top_keywords = [w for w, _ in raw_words[:5]]
        label = label_by_topic[tid]

        chunk.similarity_score = similarities[i]
        chunk.related_chunk_ids = [
            cid for cid in topic_chunk_ids[tid] if cid != chunk.chunk_id
        ]
        chunk.metadata.update({
            "topic_id": tid,
            "topic_label": label,
            "topic_words": top_keywords,
            "bertopic_used": True,
        })

    # ── Step 6 + 7: build merged chunks + ContextGroups ──────────────────
    context_groups: List[ContextGroup] = []
    sorted_topic_ids = sorted(set(topics), key=lambda t: (t == -1, t))

    for merge_idx, tid in enumerate(sorted_topic_ids):
        idxs = sorted(topic_indices[tid])
        source = [base_chunks[i] for i in idxs]

        raw_words = raw_words_by_topic[tid]
        label = label_by_topic[tid]
        top_keywords = [w for w, _ in raw_words[:5]]

        merged_text = "\n\n".join(c.text for c in source)
        avg_sim = float(np.mean([similarities[i] for i in idxs]))

        merged_chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            text=merged_text,
            method="context",
            chunk_index=merge_idx,
            similarity_score=avg_sim,
            metadata={
                "topic_id": tid,
                "topic_label": label,
                "topic_words": top_keywords,
                "bertopic_used": True,
                "paragraph_count": len(source),
                "avg_similarity_score": round(avg_sim, 4),   # ← UI reads this key
                "source_paragraph_indices": idxs,
                "source": source[0].metadata.get("source", ""),
                "format_type": source[0].metadata.get("format_type", ""),
            },
        )

        context_groups.append(ContextGroup(
            topic_id=tid,
            topic_label=label,
            topic_words=raw_words,
            merged_chunk=merged_chunk,
            source_chunks=source,
        ))

    logger.info(
        "chunk_by_context: %d paragraphs → %d topic group(s).",
        len(base_chunks), len(context_groups),
    )
    # Return individual chunks in original doc order + context groups
    return base_chunks, context_groups
