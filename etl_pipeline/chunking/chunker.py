"""
chunking/chunker.py
-------------------
Main entry-point for the chunking module.

Usage
-----
    from chunking import Chunker

    result = Chunker.chunk(document, method="paragraph")
    result.chunks          → List[Chunk]
    result.context_groups  → List[ContextGroup]  (context mode only)
    result.coherence_score → float               (context mode only)
    result.slm_payload     → dict ready for the SLM extractor
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Literal, Tuple

from chunking.schemas import Chunk, ChunkingResult, ContextGroup
from chunking.strategies import (
    Segment,
    chunk_by_context,
    chunk_by_line,
    chunk_by_paragraph,
    chunk_by_section,
)

logger = logging.getLogger(__name__)

ChunkMethod = Literal['line', 'paragraph', 'section', 'context']


class Chunker:
    """Stateless chunking dispatcher.  Call Chunker.chunk(doc, method)."""

    @classmethod
    def chunk(
        cls,
        document: Any,
        method: ChunkMethod = 'paragraph',
    ) -> ChunkingResult:
        """
        Chunk a processed document.

        Parameters
        ----------
        document : TextDocument | BinaryDocument
            The result object produced by text_handler or binary_handler.
        method : 'line' | 'paragraph' | 'section' | 'context'
            Chunking strategy to apply.

        Returns
        -------
        ChunkingResult
        """
        segments = cls._extract_segments(document)

        if not segments:
            logger.warning('Chunker: no text segments found.')
            return ChunkingResult(method=method, total_chunks=0)

        context_groups: List[ContextGroup] = []
        coherence_score: float = 0.0

        # line / paragraph / section all work on the full rejoined text
        full_segments = cls._as_full_text_segments(document, segments)

        if method == 'line':
            chunks = chunk_by_line(full_segments)

        elif method == 'paragraph':
            chunks = chunk_by_paragraph(full_segments)

        elif method == 'section':
            chunks = chunk_by_section(full_segments)

        elif method == 'context':
            # context works on pre-split paragraph segments
            chunks, context_groups, coherence_score = chunk_by_context(segments)

        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose: line, paragraph, section, context."
            )

        logger.info(
            "Chunker: method='%s' → %d chunk(s), coherence=%.4f",
            method, len(chunks), coherence_score,
        )

        return ChunkingResult(
            method=method,
            total_chunks=len(chunks),
            chunks=chunks,
            context_groups=context_groups,
            coherence_score=coherence_score,
        )

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _as_full_text_segments(
        document: Any, segments: List[Segment]
    ) -> List[Segment]:
        """
        Rejoin all TextDocument sections into one segment so that
        line / paragraph / section strategies can re-split on the full text.

        BinaryDocument per-block segments are returned unchanged.
        """
        if hasattr(document, 'pages'):          # BinaryDocument
            return segments

        if hasattr(document, 'sections'):       # TextDocument
            full_text = '\n\n'.join(text for text, _ in segments)
            if not full_text.strip():
                return segments
            base_meta = {**segments[0][1]} if segments else {'source': 'text'}
            return [(full_text, base_meta)]

        return segments

    @classmethod
    def _extract_segments(cls, document: Any) -> List[Segment]:
        if hasattr(document, 'pages'):
            return cls._segments_from_binary(document)
        if hasattr(document, 'sections'):
            return cls._segments_from_text(document)
        if hasattr(document, 'raw_text') and document.raw_text:
            return [(document.raw_text, {'source': 'raw_text'})]
        logger.warning("Chunker: unrecognised document type '%s'.", type(document))
        return []

    @staticmethod
    def _segments_from_binary(document: Any) -> List[Segment]:
        segments: List[Segment] = []
        for page in document.pages:
            for block in page.blocks:
                text = (block.corrected_text or block.raw_text or '').strip()
                if not text:
                    continue
                segments.append((text, {
                    'source': 'binary',
                    'page_number': page.page_number,
                    'block_id': block.block_id,
                    'block_label': block.label,
                    'block_title': block.title or '',
                    'confidence': block.confidence,
                }))
        return segments

    @staticmethod
    def _segments_from_text(document: Any) -> List[Segment]:
        segments: List[Segment] = []
        for section in document.sections:
            text = (section.content or '').strip()
            if not text:
                continue
            # Skip auto-generated summary sections
            if section.metadata.get('section_type') == 'summary':
                continue
            segments.append((text, {
                'source': 'text',
                'section_id': section.section_id,
                'format_type': section.format_type,
            }))
        if not segments and hasattr(document, 'raw_text') and document.raw_text:
            segments.append((document.raw_text.strip(), {'source': 'raw_text'}))
        return segments
