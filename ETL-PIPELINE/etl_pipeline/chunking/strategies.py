"""
chunking/strategies.py
----------------------
Four pure-function chunking strategies.

All strategies accept a list of (text, source_metadata) tuples and return
a flat list of Chunk objects plus (for context mode) a list of ContextGroups.

Strategies
----------
chunk_by_line      — finest: split on newlines, then sentence boundaries
chunk_by_paragraph — medium: split on blank lines
chunk_by_section   — structural: split on detected headers
chunk_by_context   — semantic: BERTopic topic clustering (Auto only)
                     includes silhouette coherence scoring
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any, Dict, List, Tuple

from chunking.schemas import Chunk, ContextGroup

logger = logging.getLogger(__name__)

# Type alias: each segment is (text_content, source_metadata_dict)
Segment = Tuple[str, Dict[str, Any]]

# Sentence boundary: after . ! ? followed by whitespace or end-of-string
_SENTENCE_END = re.compile(r'(?<=[.!?])(?:\s+|$)')

# Lines that look like structural headings (must occupy a full line)
_SECTION_HEADER = re.compile(
    r'(?m)^(?:'
    r'#{1,6}\s+\S.*'                               # ## Markdown heading
    r'|\d+[.)]\s+[A-Z]\S*.*'                       # 1. Title  /  2) Title
    r'|(?:Chapter|Section|Part|Appendix)\s+\S.*'   # Chapter One / Section 2
    r'|[A-Z][A-Z ]{3,60}$'                         # ALL CAPS TITLE LINE
    r')$'
)

_BERTOPIC_RANDOM_SEED = 42
_MIN_PARAGRAPHS_FOR_BERTOPIC = 4


# ═══════════════════════════════════════════════════════════════════════════
# 1. LINE  (sentence-level — finest granularity)
# ═══════════════════════════════════════════════════════════════════════════

def chunk_by_line(segments: List[Segment]) -> List[Chunk]:
    """
    Split each segment on newlines first so headings stay separate,
    then split each non-empty line further by sentence boundaries.
    """
    chunks: List[Chunk] = []
    idx = 0

    for text, meta in segments:
        text = text.replace('\r\n', '\n').replace('\r', '\n').strip()
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            for sent in _SENTENCE_END.split(line):
                sent = sent.strip()
                if not sent:
                    continue
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=sent,
                    method='line',
                    chunk_index=idx,
                    metadata={**meta},
                ))
                idx += 1

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# 2. PARAGRAPH  (blank-line boundaries)
# ═══════════════════════════════════════════════════════════════════════════

def chunk_by_paragraph(segments: List[Segment]) -> List[Chunk]:
    """Each blank-line-separated block becomes one Chunk."""
    chunks: List[Chunk] = []
    idx = 0

    for text, meta in segments:
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        for para in re.split(r'\n{2,}', text):
            para = para.strip()
            if not para:
                continue
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                text=para,
                method='paragraph',
                chunk_index=idx,
                metadata={**meta},
            ))
            idx += 1

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# 3. SECTION  (header-guided grouping)
# ═══════════════════════════════════════════════════════════════════════════

def chunk_by_section(segments: List[Segment]) -> List[Chunk]:
    """
    Detect structural headers (## Markdown, numbered, ALL-CAPS, Chapter/Section).
    Each header + its following body becomes one Chunk.
    Falls back to paragraph chunking when no headers are detected.
    """
    chunks: List[Chunk] = []
    idx = 0

    for text, meta in segments:
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        boundaries = [m.start() for m in _SECTION_HEADER.finditer(text)]

        if not boundaries:
            # No structural markers → paragraph fallback
            for para in re.split(r'\n{2,}', text):
                para = para.strip()
                if para:
                    chunks.append(Chunk(
                        chunk_id=str(uuid.uuid4()),
                        text=para,
                        method='section',
                        chunk_index=idx,
                        metadata={**meta, 'section_detected': False},
                    ))
                    idx += 1
            continue

        # Capture any preamble before the first header
        if boundaries[0] > 0:
            preamble = text[:boundaries[0]].strip()
            if preamble:
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=preamble,
                    method='section',
                    chunk_index=idx,
                    metadata={**meta, 'section_detected': False, 'section_number': 0},
                ))
                idx += 1

        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
            section_text = text[start:end].strip()
            if not section_text:
                continue
            chunks.append(Chunk(
                chunk_id=str(uuid.uuid4()),
                text=section_text,
                method='section',
                chunk_index=idx,
                metadata={**meta, 'section_detected': True, 'section_number': i + 1},
            ))
            idx += 1

    return chunks


# ═══════════════════════════════════════════════════════════════════════════
# 4. CONTEXT  (BERTopic semantic clustering — Auto, with coherence score)
# ═══════════════════════════════════════════════════════════════════════════

def chunk_by_context(
    segments: List[Segment],
    embedding_model_name: str = 'all-MiniLM-L6-v2',
) -> Tuple[List[Chunk], List[ContextGroup], float]:
    """
    Semantic context chunking using BERTopic (Auto mode only).

    Pipeline
    --------
    1. Paragraph-level base split across all segments
    2. sentence-transformers embeddings (one vector per paragraph)
    3. BERTopic topic assignment (fully automatic — no manual nr_topics)
    4. Silhouette score computed on embeddings → overall coherence score
    5. Cosine similarity of each paragraph to its topic centroid
    6. Build ContextGroup per topic:
         merged_chunk  = all paragraphs joined (SLM-ready)
         source_chunks = individual paragraphs with similarity scores

    Returns
    -------
    (individual_chunks, context_groups, overall_coherence_score)
    """
    import numpy as np

    # ── Step 1: paragraph-level base split ──────────────────────────────────
    paragraphs: List[str] = []
    para_metas: List[Dict[str, Any]] = []

    for text, meta in segments:
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        for para in re.split(r'\n{2,}', text):
            para = para.strip()
            if not para:
                continue
            # Skip bare headings (## Foo, 1. Foo, ALL CAPS short lines) —
            # they carry no semantic content for BERTopic and form noise clusters.
            word_count = len(para.split())
            is_heading = bool(_SECTION_HEADER.match(para)) or (
                word_count <= 3 and not re.search(r'[.!?]', para)
            )
            if is_heading:
                continue
            paragraphs.append(para)
            para_metas.append(meta)

    if len(paragraphs) < _MIN_PARAGRAPHS_FOR_BERTOPIC:
        logger.warning(
            'chunk_by_context: only %d paragraph(s) — too few for BERTopic, '
            'falling back to paragraph chunking.', len(paragraphs)
        )
        fallback = chunk_by_paragraph(segments)
        # Wrap each chunk in its own ContextGroup for UI consistency
        groups = [
            ContextGroup(
                topic_id=i,
                topic_label=f'Group {i + 1}',
                topic_words=[],
                merged_chunk=c,
                source_chunks=[c],
                coherence_score=1.0,
            )
            for i, c in enumerate(fallback)
        ]
        return fallback, groups, 1.0

    # ── Step 2: embeddings ───────────────────────────────────────────────────
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise RuntimeError('sentence-transformers is required for context chunking. '
                           'Install it with: pip install sentence-transformers')

    model = SentenceTransformer(embedding_model_name)
    embeddings = model.encode(paragraphs, show_progress_bar=False, normalize_embeddings=True)
    embeddings_np = np.array(embeddings)

    # ── Step 3: BERTopic (Auto) ──────────────────────────────────────────────
    try:
        from bertopic import BERTopic
        from umap import UMAP
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError as e:
        raise RuntimeError(f'BERTopic dependencies missing: {e}. '
                           'Install with: pip install bertopic umap-learn hdbscan')

    n = len(paragraphs)
    n_components = max(2, min(5, n - 2))
    n_neighbors  = max(2, min(5, n - 1))

    umap_model  = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric='cosine',
        random_state=_BERTOPIC_RANDOM_SEED,
    )
    # Scale min_cluster_size with document size so large docs get coherent
    # groups instead of splitting into dozens of tiny clusters.
    min_cs = max(2, n // 5)
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cs,
        min_samples=1,
        metric='euclidean',
        cluster_selection_method='eom',
        prediction_data=True,
    )
    vectorizer = CountVectorizer(stop_words='english', min_df=1)

    topic_model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        verbose=False,
    )

    topics, _ = topic_model.fit_transform(paragraphs, embeddings=embeddings_np)

    # ── Reassign outliers (topic -1) to the nearest real topic ───────────────
    # Outliers are paragraphs HDBSCAN couldn't confidently cluster. Instead of
    # surfacing them as noise, assign each one to the topic whose centroid it
    # is closest to (cosine similarity).
    tmp_centroids: Dict[int, np.ndarray] = {}
    for tid in set(t for t in topics if t != -1):
        idxs = [i for i, t in enumerate(topics) if t == tid]
        tmp_centroids[tid] = embeddings_np[idxs].mean(axis=0)

    if tmp_centroids:  # only reassign when there are real topics
        for i, t in enumerate(topics):
            if t == -1:
                best_tid = max(
                    tmp_centroids,
                    key=lambda tid: float(np.dot(embeddings_np[i], tmp_centroids[tid])),
                )
                topics[i] = best_tid

    # ── Step 4: overall silhouette coherence score ───────────────────────────
    overall_coherence = _silhouette_score(embeddings_np, topics)

    # ── Step 5: per-paragraph similarity to topic centroid ──────────────────
    # Build centroids per topic (mean of embeddings)
    unique_topics = sorted(set(t for t in topics if t != -1))
    centroids: Dict[int, np.ndarray] = {}
    for tid in unique_topics:
        idxs = [i for i, t in enumerate(topics) if t == tid]
        centroids[tid] = embeddings_np[idxs].mean(axis=0)

    # Get topic info for labels/words
    try:
        topic_info = topic_model.get_topic_info()
    except Exception:
        topic_info = None

    def _topic_label(tid: int) -> str:
        """Return a clean human-readable label, stripping BERTopic's N_w_w_w prefix."""
        if topic_info is not None:
            row = topic_info[topic_info['Topic'] == tid]
            if not row.empty and 'Name' in row.columns:
                raw = str(row.iloc[0]['Name'])
                # BERTopic names look like "0_cancer_patient_study" — strip leading N_
                clean = re.sub(r'^-?\d+_', '', raw)
                clean = clean.replace('_', ' ').title()
                if clean:
                    return clean
        words = _topic_words(tid)
        return ', '.join(words[:3]).title() if words else f'Group {tid + 1}'

    def _topic_words(tid: int) -> List[str]:
        try:
            return [w for w, _ in (topic_model.get_topic(tid) or [])[:5]]
        except Exception:
            return []

    # ── Step 6: build Chunk objects per paragraph ────────────────────────────
    all_chunks: List[Chunk] = []
    topic_to_chunks: Dict[int, List[Chunk]] = {}

    outlier_group_id   = str(uuid.uuid4())
    outlier_topic_id   = -1

    for i, (para, meta, topic_id) in enumerate(zip(paragraphs, para_metas, topics)):
        # Similarity to centroid (cosine — embeddings already L2-normalised)
        if topic_id != -1 and topic_id in centroids:
            sim = float(np.dot(embeddings_np[i], centroids[topic_id]))
        else:
            sim = 0.0

        chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            text=para,
            method='context',
            chunk_index=i,
            metadata={
                **meta,
                'topic_id': int(topic_id),
                'topic_label': _topic_label(topic_id) if topic_id != -1 else 'outlier',
                'topic_words': _topic_words(topic_id) if topic_id != -1 else [],
            },
            similarity_score=round(sim, 4),
        )
        all_chunks.append(chunk)
        topic_to_chunks.setdefault(topic_id, []).append(chunk)

    # Cross-link related chunk IDs within each topic
    for tid, group_chunks in topic_to_chunks.items():
        ids = [c.chunk_id for c in group_chunks]
        for c in group_chunks:
            c.related_chunk_ids = [x for x in ids if x != c.chunk_id]

    # ── Step 7: build ContextGroups ──────────────────────────────────────────
    context_groups: List[ContextGroup] = []

    # Non-outlier topics
    for tid in unique_topics:
        group_chunks = topic_to_chunks.get(tid, [])
        if not group_chunks:
            continue

        merged_text = '\n\n'.join(c.text for c in group_chunks)
        avg_sim     = round(sum(c.similarity_score for c in group_chunks) / len(group_chunks), 4)

        # Intra-group coherence (silhouette of this group vs others)
        group_idxs = [i for i, t in enumerate(topics) if t == tid]
        group_coherence = _silhouette_score(embeddings_np, topics, subset=group_idxs)

        label = _topic_label(tid)
        words = _topic_words(tid)

        merged_chunk = Chunk(
            chunk_id=str(uuid.uuid4()),
            text=merged_text,
            method='context_merged',
            chunk_index=tid,
            metadata={
                'topic_id': tid,
                'topic_label': label,
                'topic_words': words,
                'avg_similarity_score': avg_sim,
                'source_paragraph_count': len(group_chunks),
            },
            similarity_score=avg_sim,
        )

        context_groups.append(ContextGroup(
            topic_id=tid,
            topic_label=label,
            topic_words=words,
            merged_chunk=merged_chunk,
            source_chunks=group_chunks,
            coherence_score=group_coherence,
        ))

    # Sort groups by topic_id for stable order
    context_groups.sort(key=lambda g: (g.topic_id == -1, g.topic_id))

    return all_chunks, context_groups, overall_coherence


# ── Helpers ──────────────────────────────────────────────────────────────────

def _silhouette_score(
    embeddings: 'np.ndarray',
    topics: List[int],
    subset: List[int] | None = None,
) -> float:
    """
    Silhouette score for topic assignments.
    Returns a value in [-1, 1]; higher is better.
    Returns 0.0 when there is not enough data to compute.
    """
    try:
        import numpy as np
        from sklearn.metrics import silhouette_score

        valid_idx = [i for i, t in enumerate(topics) if t != -1]
        if len(valid_idx) < 4:
            return 0.0
        valid_labels = [topics[i] for i in valid_idx]
        if len(set(valid_labels)) < 2:
            return 0.0
        valid_emb = embeddings[valid_idx]
        score = silhouette_score(valid_emb, valid_labels, metric='cosine')
        return round(float(score), 4)
    except Exception:
        return 0.0
