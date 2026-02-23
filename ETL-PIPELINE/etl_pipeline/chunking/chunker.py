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
        granularity: str = "Auto",
    ) -> ChunkingResult:
        """
        Chunk a processed document.

        Parameters
        ----------
        document : TextDocument | BinaryDocument
            The result object produced by text_handler or binary_handler.
        method : 'line' | 'paragraph' | 'section' | 'context'
            Chunking strategy to apply.
        granularity : 'Very Broad' | 'Broad' | 'Auto' | 'Fine' | 'Very Fine'
            Context-mode only. Controls how aggressively topics are split:
            Very Broad → fewest groups; Very Fine → most groups.

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
        context_groups: List[ContextGroup] = []

        if method == "line":
            chunks = chunk_by_line(segments)

        elif method == "paragraph":
            chunks = chunk_by_paragraph(segments)

        elif method == "section":
            chunks = chunk_by_section(segments)

        elif method == "context":
            chunks, context_groups = chunk_by_context(segments, granularity=granularity)

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

    # ── Internal helpers ────────────────────────────────────────────────────

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
