"""
Marker handler — converts text-based PDFs to clean structured output.

Marker (https://github.com/VikParuchuri/marker) uses Surya internally to:
  - Detect layout (columns, headings, tables, figures)
  - Recognise text with high accuracy
  - Output clean Markdown preserving document structure

Use this for PDFs that contain EMBEDDED TEXT (not scanned).
For scanned PDFs / images → use DocLayout-YOLO + Chandra (binary_handler.py).

Singleton pattern: models (~4 GB total) are loaded once per process and
kept hot. They are much smaller than Chandra (~7 GB) so both can coexist
on the L4 (22.5 GB).

Output: BinaryDocument with one Page per logical section group,
blocks derived from Markdown heading structure.
"""

import io
import logging
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import List, Optional

from handlers.binary_schema import BinaryDocument, Page, Block, Region
from ingestion.schemas import DocumentObject

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton model dict (loaded once)
# ---------------------------------------------------------------------------
_marker_models = None


def _load_marker_models():
    global _marker_models
    if _marker_models is not None:
        return _marker_models

    logger.info("Loading Marker models (~4 GB, one-time download)...")
    try:
        from marker.models import create_model_dict
        _marker_models = create_model_dict()
        logger.info("Marker models loaded.")
        return _marker_models
    except Exception as exc:
        logger.error(f"Failed to load Marker models: {exc}")
        raise


def unload_marker_models() -> None:
    """Free Marker model memory."""
    global _marker_models
    if _marker_models is None:
        return
    try:
        import torch
        del _marker_models
        _marker_models = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Marker models unloaded.")
    except Exception as exc:
        logger.warning(f"Error unloading Marker models: {exc}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_pdf_with_marker(doc: DocumentObject) -> BinaryDocument:
    """
    Convert a text-based PDF using Marker.

    Marker handles:
      - Multi-column layouts
      - Tables → clean Markdown tables
      - Headings → preserved hierarchy
      - Mixed text + figure pages
      - Headers/footers → stripped
      - Mathematical formulas → LaTeX

    Returns:
        BinaryDocument with blocks derived from Markdown structure.
    """
    markdown_text = _run_marker(doc.raw_bytes, doc.document_id)
    pages = _markdown_to_pages(markdown_text, doc.document_id)

    logger.info(
        f"Marker converted {doc.document_id}: "
        f"{len(pages)} page(s), "
        f"{sum(len(p.blocks) for p in pages)} block(s)."
    )

    return BinaryDocument(
        document_id=doc.document_id,
        pages=pages,
        metadata={
            "source_format":      doc.detected_format,
            "mime_type":          doc.mime_type,
            "format_type":        "pdf",
            "pages_count":        len(pages),
            "file_size":          len(doc.raw_bytes),
            "extraction_method":  "marker",
            "vlm_layout_guided":  True,
        },
    )


# ---------------------------------------------------------------------------
# Marker invocation
# ---------------------------------------------------------------------------

def _run_marker(raw_bytes: bytes, document_id: str) -> str:
    """
    Write PDF to a temp file, run Marker, return Markdown string.
    Temp file is cleaned up regardless of success/failure.
    """
    tmp_path: Optional[Path] = None
    try:
        # Marker requires a file path, not raw bytes
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = Path(tmp.name)

        model_dict = _load_marker_models()

        from marker.converters.pdf import PdfConverter
        converter = PdfConverter(artifact_dict=model_dict)

        logger.info(f"Marker processing {document_id} ({len(raw_bytes) // 1024} KB)...")
        rendered = converter(str(tmp_path))

        markdown_text = rendered.markdown
        logger.info(
            f"Marker output for {document_id}: "
            f"{len(markdown_text)} chars of Markdown."
        )
        return markdown_text

    except Exception as exc:
        logger.error(f"Marker conversion failed for {document_id}: {exc}")
        raise
    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Markdown → Page/Block structure
# ---------------------------------------------------------------------------

# Matches a Markdown heading at the START of a line, e.g. "## Section Title"
_HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)


def _markdown_to_pages(markdown_text: str, document_id: str) -> List[Page]:
    """
    Parse Marker's Markdown output into our Page / Block schema.

    Strategy:
      - Split the Markdown on top-level headings (# and ##) to group into
        logical "pages" (chapters/sections).  Sub-headings (###–######) stay
        inside their parent section as nested blocks.
      - Within each section, further split on sub-headings to build blocks.
      - Tables, figures, and formula blocks get their own Block with the
        appropriate label.
      - Each block gets a single Region containing the full text.

    Returns:
        List[Page] — at least one page (whole-doc fallback).
    """
    if not markdown_text or not markdown_text.strip():
        return [_empty_page(1)]

    # --- Split into top-level sections on H1/H2 boundaries ---
    sections = re.split(r'\n(?=#{1,2} )', "\n" + markdown_text.strip())
    sections = [s.strip() for s in sections if s.strip()]

    if not sections:
        return [_empty_page(1)]

    pages: List[Page] = []
    for page_idx, section_text in enumerate(sections, start=1):
        blocks = _section_to_blocks(section_text, page_idx)
        pages.append(Page(
            page_id=str(uuid.uuid4()),
            page_number=page_idx,
            blocks=blocks,
            metadata={
                "extraction_method": "marker",
                "layout_guided": True,
            },
        ))

    return pages


def _section_to_blocks(section_text: str, page_number: int) -> List[Block]:
    """
    Convert a single top-level Markdown section into a list of Blocks.

    The first heading line becomes the section title block; subsequent
    sub-headings (###–######) and body paragraphs become their own blocks.
    """
    blocks: List[Block] = []
    lines  = section_text.split("\n")

    # The first line may be a heading — extract it as the section title block
    first_line = lines[0]
    heading_match = _HEADING_RE.match(first_line)
    if heading_match:
        level = len(heading_match.group(1))
        title = heading_match.group(2).strip()
        label = "title" if level == 1 else "heading"
        blocks.append(_make_block(title, label, title=title))
        remaining_lines = lines[1:]
    else:
        remaining_lines = lines

    # Split remaining content on sub-headings (###–######)
    chunks = _split_on_subheadings("\n".join(remaining_lines))
    for chunk_title, chunk_label, chunk_body in chunks:
        text = (f"{chunk_title}\n{chunk_body}".strip()
                if chunk_title else chunk_body.strip())
        if text:
            blocks.append(_make_block(text, chunk_label, title=chunk_title))

    return blocks or [_make_block(section_text.strip(), "body")]


def _split_on_subheadings(text: str):
    """
    Yield (title, label, body) tuples by splitting text at ### – ###### headings.
    Paragraphs between headings become ("", "body", paragraph_text) tuples.
    Tables are detected by pipe-table syntax and labelled accordingly.
    """
    parts: List[tuple] = []
    current_title = ""
    current_label = "body"
    current_body_lines: List[str] = []

    for line in text.split("\n"):
        sub_match = re.match(r'^(#{3,6})\s+(.+)$', line)
        if sub_match:
            # Flush current buffer
            body = "\n".join(current_body_lines).strip()
            if body:
                parts.append((current_title, current_label, body))
            current_title = sub_match.group(2).strip()
            current_label = "heading"
            current_body_lines = []
        else:
            current_body_lines.append(line)

    # Flush final buffer
    body = "\n".join(current_body_lines).strip()
    if body or current_title:
        # Detect tables: Markdown table has | chars and a separator row with ---
        if "|" in body and re.search(r'\|[-: ]+\|', body):
            current_label = "table"
        parts.append((current_title, current_label, body))

    return parts


def _make_block(
    text: str,
    label: str,
    title: str = "",
    bbox: Optional[List[int]] = None,
) -> Block:
    """Build a Block with a single Region."""
    return Block(
        block_id  = str(uuid.uuid4()),
        title     = title,
        label     = label,
        bbox      = bbox or [0, 0, 0, 0],  # Marker Markdown has no pixel coords
        regions   = [Region(
            region_id  = str(uuid.uuid4()),
            text       = text,
            bbox       = [0, 0, 0, 0],
            confidence = 1.0,
            metadata   = {"engine": "marker"},
        )],
        raw_text   = text,
        confidence = 1.0,
        metadata   = {"engine": "marker"},
    )


def _empty_page(page_number: int) -> Page:
    return Page(
        page_id    = str(uuid.uuid4()),
        page_number = page_number,
        blocks     = [],
        metadata   = {"extraction_method": "marker", "empty": True},
    )
