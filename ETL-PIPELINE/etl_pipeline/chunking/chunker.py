"""
chunking/chunker.py
-------------------
Main entry-point for the chunking module.

Accepts either a TextDocument or a BinaryDocument produced by the existing
ETL handlers, extracts source-aware segments, and dispatches to the
requested chunking strategy.

Usage
-----
    from chunking import Chunker

    result = Chunker.chunk(result_document, method="paragraph")
    # result  → ChunkingResult
    # result.chunks          → List[Chunk]
    # result.context_groups  → List[ContextGroup]  (context mode only)
    # result.slm_payload     → dict ready for the SLM extractor
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

ChunkMethod = Literal["line", "paragraph", "section", "context"]


class Chunker:
    """
    Stateless chunking dispatcher.

    All public logic lives in the class method `chunk()`.
    No need to instantiate — call `Chunker.chunk(doc, method)`.
    """

    # ── Public API ──────────────────────────────────────────────────────────

    @classmethod
    def chunk(
        cls,
        document: Any,
        method: ChunkMethod = "paragraph",
        nr_topics: int | None = None,
    ) -> ChunkingResult:
        """
        Chunk a processed document.

        Parameters
        ----------
        document : TextDocument | BinaryDocument
            The result object produced by text_handler or binary_handler.
        method : 'line' | 'paragraph' | 'section' | 'context'
            Chunking strategy to apply.
        nr_topics : int | None
            Context-mode only.
            None  → Auto: BERTopic decides the natural number of topics.
            int N → Manual: merge down to N groups (clamped to natural max).

        Returns
        -------
        ChunkingResult
            Contains flat chunks, optional context groups, and the
            SLM-ready payload.
        """
        segments = cls._extract_segments(document)

        if not segments:
            logger.warning("Chunker: no text segments found in document.")
            return ChunkingResult(method=method, total_chunks=0)

        # ── Dispatch ────────────────────────────────────────────────────────
        # Line and paragraph modes need the FULL raw text as one segment so
        # they can apply their own splitting.  The plain-text parser already
        # splits by blank line into individual sections, which would make all
        # three simple modes produce identical output if we passed pre-split
        # segments directly.
        context_groups: List[ContextGroup] = []

        if method == "line":
            chunks = chunk_by_line(cls._as_full_text_segments(document, segments))

        elif method == "paragraph":
            chunks = chunk_by_paragraph(cls._as_full_text_segments(document, segments))

        elif method == "section":
            # Section mode intentionally uses the parser-derived structural
            # segments (headings, chapters) — keep as-is.
            chunks = chunk_by_section(segments)

        elif method == "context":
            chunks, context_groups = chunk_by_context(segments, nr_topics=nr_topics)

        else:
            raise ValueError(
                f"Unknown chunking method '{method}'. "
                "Choose from: line, paragraph, section, context."
            )

        logger.info(
            "Chunker: method='%s' produced %d chunk(s) from %d segment(s).",
            method, len(chunks), len(segments),
        )

        return ChunkingResult(
            method=method,
            total_chunks=len(chunks),
            chunks=chunks,
            context_groups=context_groups,
        )

    @staticmethod
    def _as_full_text_segments(
        document: Any, segments: List[Segment]
    ) -> List[Segment]:
        """
        For line / paragraph chunking we need the complete raw text as ONE
        segment so the strategy can do its own splitting.

        For TextDocuments the parser has already split by blank lines, so we
        rejoin all section texts with double-newlines to restore the original
        structure.  For BinaryDocuments (page blocks) we keep the existing
        per-block segments unchanged — each block is already a natural unit.
        """
        # BinaryDocument — keep per-block granularity
        if hasattr(document, "pages"):
            return segments

        # TextDocument — join all section content back into full document text
        if hasattr(document, "sections"):
            full_text = "\n\n".join(text for text, _ in segments)
            if not full_text.strip():
                return segments
            # Use metadata from the first segment as representative provenance
            base_meta = {**segments[0][1]} if segments else {"source": "text"}
            return [(full_text, base_meta)]

        # Raw-text fallback — already a single segment
        return segments

    @classmethod
    def _extract_segments(cls, document: Any) -> List[Segment]:
        """
        Convert a document object into a flat list of (text, metadata) tuples.

        Handles both:
        - BinaryDocument  (pages → blocks → raw_text / corrected_text)
        - TextDocument    (sections → content)
        """
        # ── BinaryDocument ──────────────────────────────────────────────────
        if hasattr(document, "pages"):
            return cls._segments_from_binary(document)

        # ── TextDocument ────────────────────────────────────────────────────
        if hasattr(document, "sections"):
            return cls._segments_from_text(document)

        # ── Raw string fallback ─────────────────────────────────────────────
        if hasattr(document, "raw_text") and document.raw_text:
            return [(document.raw_text, {"source": "raw_text"})]

        logger.warning("Chunker: unrecognised document type '%s'.", type(document))
        return []

    @staticmethod
    def _segments_from_binary(document: Any) -> List[Segment]:
        """
        Extract segments from a BinaryDocument.
        One segment per OCR block (preserves page number and block metadata).
        """
        segments: List[Segment] = []

        for page in document.pages:
            for block in page.blocks:
                # Prefer corrected text if available, otherwise raw OCR
                text = (block.corrected_text or block.raw_text or "").strip()
                if not text:
                    continue
                meta: Dict[str, Any] = {
                    "source": "binary",
                    "page_number": page.page_number,
                    "block_id": block.block_id,
                    "block_label": block.label,
                    "block_title": block.title or "",
                    "confidence": block.confidence,
                }
                segments.append((text, meta))

        return segments

    @staticmethod
    def _segments_from_text(document: Any) -> List[Segment]:
        """
        Extract segments from a TextDocument.
        One segment per TextSection.
        """
        segments: List[Segment] = []

        for section in document.sections:
            text = (section.content or "").strip()
            if not text:
                continue
            # Skip auto-generated document statistics / summary sections —
            # they are parser artefacts and pollute topic modelling.
            if section.metadata.get("section_type") == "summary":
                continue
            meta: Dict[str, Any] = {
                "source": "text",
                "section_id": section.section_id,
                "format_type": section.format_type,
            }
            segments.append((text, meta))

        # If no sections, fall back to raw_text
        if not segments and hasattr(document, "raw_text") and document.raw_text:
            segments.append((document.raw_text.strip(), {"source": "raw_text"}))

        return segments
