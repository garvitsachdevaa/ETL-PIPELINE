"""
chandra_adapter.py — Chandra OCR integration (Phase 2)

Chandra (datalab-to/chandra) is a VLM-based OCR model.
It requires GPU to run (--method hf loads the HuggingFace model locally).

The Chandra package is installed in the OLD venv:
    /home/dev-khera/Desktop/etl_pipeline/.venv

We import it by adding that venv's site-packages to sys.path at call time.
A PyPDF2 fallback is used when Chandra / GPU is not available.

Output per page (BatchOutputItem):
    .markdown  — full page text as Markdown
    .html      — full page text as HTML
    .chunks    — List[dict]  each: {"bbox": [...], "label": "...", "content": "<html>"}
    .page_box  — [0, 0, width, height]
    .error     — bool
"""

import io
import logging
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import PyPDF2

logger = logging.getLogger(__name__)

def _ensure_chandra_importable():
    """No-op: chandra-ocr is now installed in the project venv."""
    pass


# ---------------------------------------------------------------------------
# Singleton model (loaded once per process)
# ---------------------------------------------------------------------------
_chandra_model = None


def _load_chandra_model():
    """Load the Chandra InferenceManager singleton (GPU required)."""
    global _chandra_model
    if _chandra_model is not None:
        return _chandra_model

    _ensure_chandra_importable()
    from chandra.model import InferenceManager  # noqa: PLC0415

    logger.info("Loading Chandra model (method=hf) — GPU required …")
    _chandra_model = InferenceManager(method="hf")
    logger.info("Chandra model loaded.")
    return _chandra_model


def unload_chandra_model():
    """Free GPU memory used by the Chandra model."""
    global _chandra_model
    if _chandra_model is None:
        return
    try:
        import torch
        del _chandra_model
        _chandra_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Chandra model unloaded.")
    except Exception as exc:
        logger.warning(f"Error unloading Chandra model: {exc}")


# ---------------------------------------------------------------------------
# Public API — whole-document OCR (legacy / whole-page path)
# ---------------------------------------------------------------------------

def run_chandra_cli(raw_bytes: bytes, document_id: str) -> dict:
    """
    Run Chandra OCR on a full PDF or image.

    Returns a dict compatible with the legacy schema:
        {"pages": [{"page_number": N, "blocks": [{"text": ..., "bbox": ..., ...}]}]}
    for PDFs, or:
        {"blocks": [...]}
    for images.

    Falls back to PyPDF2 text extraction if Chandra / GPU is unavailable.
    """
    try:
        return _run_chandra_python(raw_bytes, document_id)
    except Exception as exc:
        logger.warning(f"Chandra OCR unavailable ({exc}), falling back to PyPDF2.")
        return _run_pypdf2_fallback(raw_bytes, document_id)


# ---------------------------------------------------------------------------
# Public API — block-level OCR (VLM layout-guided path)
# ---------------------------------------------------------------------------

def run_chandra_on_block(image, block_id: str, block_label: str) -> dict:
    """
    Run Chandra OCR on a single cropped PIL image (one LayoutBlock crop).

    Args:
        image:        PIL.Image.Image of the cropped region.
        block_id:     UUID of the LayoutBlock.
        block_label:  VLM semantic label (passed through).

    Returns:
        {
            "block_id":   str,
            "title":      str,   # Chandra-inferred label for this crop
            "label":      str,   # original VLM label (passed through)
            "text":       str,   # Markdown text from Chandra
            "words":      [],    # word-level not available from Chandra; empty list
            "confidence": float, # 0.0 on error, 1.0 on success
        }

    Falls back to an empty result if Chandra / GPU is not available.
    """
    try:
        return _run_chandra_on_pil(image, block_id, block_label)
    except Exception as exc:
        logger.warning(f"Chandra block OCR failed ({exc}), returning empty block.")
        return _chandra_block_fallback(block_id, block_label)


# ---------------------------------------------------------------------------
# Internal — Chandra Python API
# ---------------------------------------------------------------------------

def _run_chandra_python(raw_bytes: bytes, document_id: str) -> dict:
    """
    Run Chandra on raw bytes using its Python API.
    Writes bytes to a temp file, loads with chandra.input.load_file,
    then runs InferenceManager.generate().
    """
    _ensure_chandra_importable()
    from chandra.input import load_file           # noqa: PLC0415
    from chandra.model.schema import BatchInputItem  # noqa: PLC0415

    # Write bytes to temp file so load_file can detect type
    suffix = _guess_suffix(raw_bytes)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        images = load_file(tmp_path, {})
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    model = _load_chandra_model()
    batch = [BatchInputItem(image=img, prompt_type="ocr_layout") for img in images]
    results = model.generate(batch, include_images=False, include_headers_footers=False)

    # Map to legacy schema
    if len(images) == 1 and suffix != ".pdf":
        # Single image
        r = results[0]
        blocks = _chunks_to_blocks(r.chunks, r.markdown)
        return {"blocks": blocks}
    else:
        # PDF — one result per page
        pages = []
        for page_idx, r in enumerate(results, start=1):
            blocks = _chunks_to_blocks(r.chunks, r.markdown)
            pages.append({"page_number": page_idx, "blocks": blocks})
        return {"pages": pages}


def _run_chandra_on_pil(image, block_id: str, block_label: str) -> dict:
    """Run Chandra on a single PIL image (already cropped to a block)."""
    _ensure_chandra_importable()
    from chandra.model.schema import BatchInputItem  # noqa: PLC0415

    model = _load_chandra_model()
    batch = [BatchInputItem(image=image, prompt_type="ocr_layout")]
    results = model.generate(batch, include_images=False, include_headers_footers=False)
    r = results[0]

    # Use the first chunk label as the Chandra-assigned title (if available)
    title = ""
    if r.chunks:
        title = r.chunks[0].get("label", "")

    return {
        "block_id":   block_id,
        "title":      title,
        "label":      block_label,
        "text":       r.markdown,
        "words":      [],       # Chandra doesn't expose word-level tokens
        "confidence": 0.0 if r.error else 1.0,
    }


# ---------------------------------------------------------------------------
# Internal — helpers
# ---------------------------------------------------------------------------

def _chunks_to_blocks(chunks: list, full_markdown: str) -> list:
    """
    Convert Chandra's chunk list to our legacy block schema.

    Each chunk: {"bbox": [x0,y0,x1,y1], "label": str, "content": "<html>"}
    """
    if not chunks:
        # No chunk-level data — return the whole markdown as one block
        return [{"text": full_markdown, "bbox": [0, 0, 0, 0],
                 "label": "full_page", "title": "", "confidence": 1.0}]

    blocks = []
    for chunk in chunks:
        # Strip HTML tags from content to get plain text
        content_html = chunk.get("content", "")
        text = _strip_html(content_html)
        blocks.append({
            "text":       text,
            "bbox":       chunk.get("bbox", [0, 0, 0, 0]),
            "label":      chunk.get("label", "unknown"),
            "title":      chunk.get("label", ""),   # Chandra label as title
            "confidence": 1.0,
        })
    return blocks


def _strip_html(html: str) -> str:
    """Remove HTML tags and return plain text."""
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup(html, "html.parser").get_text(separator=" ").strip()
    except Exception:
        import re
        return re.sub(r"<[^>]+>", " ", html).strip()


def _guess_suffix(raw_bytes: bytes) -> str:
    """Guess file extension from magic bytes."""
    if raw_bytes[:4] == b"%PDF":
        return ".pdf"
    if raw_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if raw_bytes[:2] in (b"\xff\xd8", b"\xff\xe0", b"\xff\xe1"):
        return ".jpg"
    return ".bin"


def _chandra_block_fallback(block_id: str, block_label: str) -> dict:
    """Return an empty OCR result when Chandra is not available."""
    return {
        "block_id":   block_id,
        "title":      "",
        "label":      block_label,
        "text":       "",
        "words":      [],
        "confidence": 0.0,
    }


# ---------------------------------------------------------------------------
# Fallback — PyPDF2 text extraction (no GPU, text PDFs only)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fallback — PyPDF2 text extraction (no GPU, text PDFs only)
# ---------------------------------------------------------------------------

def _run_pypdf2_fallback(raw_bytes: bytes, document_id: str) -> dict:
    """
    Fallback PDF text extraction using PyPDF2.
    Only works for text-based PDFs (not scanned images).
    Returns the same schema as run_chandra_cli.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(raw_bytes))
        pages = []
        for page_num, page in enumerate(pdf_reader.pages, 1):
            try:
                text = page.extract_text() or ""
                pages.append({
                    "page_number": page_num,
                    "blocks": [{
                        "text":       text,
                        "bbox":       [0, 0, 0, 0],
                        "label":      "full_page",
                        "title":      "",
                        "confidence": 0.9,
                    }]
                })
            except Exception as exc:
                pages.append({
                    "page_number": page_num,
                    "blocks": [{
                        "text":       f"[Error extracting page {page_num}: {exc}]",
                        "bbox":       [0, 0, 0, 0],
                        "label":      "full_page",
                        "title":      "",
                        "confidence": 0.0,
                    }]
                })
        return {"pages": pages}
    except Exception as exc:
        raise RuntimeError(f"PyPDF2 fallback failed: {exc}") from exc
