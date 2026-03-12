"""
Layout detector — orchestration layer for VLM-based layout detection.

Responsibilities:
  - Rasterise PDF pages to PIL images via PyMuPDF (fitz)
  - Feed images to paligemma_adapter for block detection
  - Wrap raw block dicts into typed LayoutBlock dataclasses
  - Expose crop_image_to_block() for use by the OCR step
"""

import io
import logging
import uuid
from typing import List

from handlers.vlm.block_schema import LayoutBlock
from handlers.vlm.doclayout_yolo_adapter import detect_layout_blocks_yolo

logger = logging.getLogger(__name__)

# DPI used when rasterising PDF pages for YOLO layout detection.
# 150 DPI gives ~1240×1754 px for A4 — well within YOLO's 1024 inference size.
# YOLO internally resizes to 1024 px so higher DPI has diminishing returns.
RASTERISE_DPI = 150


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_layout_detection_on_image(
    image,                 # PIL.Image.Image
    page_number: int = 1,
    model_id: str | None = None,
) -> List[LayoutBlock]:
    """
    Detect layout blocks on a single PIL image (images or pre-rasterised PDF pages).

    Args:
        image:       A PIL.Image.Image of the full page.
        page_number: 1-based page index (used for logging and block metadata).
        model_id:    Override the default PaliGemma model ID.

    Returns:
        List[LayoutBlock] — at least one block (whole-page fallback).
    """
    kwargs = {"page_number": page_number}
    if model_id is not None:
        kwargs["model_id"] = model_id

    raw_blocks = detect_layout_blocks_yolo(image, **kwargs)
    return _build_layout_blocks(raw_blocks, page_number)


def run_layout_detection_on_pdf(
    raw_bytes: bytes,
    model_id: str | None = None,
    dpi: int = RASTERISE_DPI,
) -> List[List[LayoutBlock]]:
    """
    Rasterise every page of a PDF and detect layout blocks per page.

    Args:
        raw_bytes:  Raw PDF file bytes.
        model_id:   Override the default PaliGemma model ID.
        dpi:        Rasterisation DPI (default 150).

    Returns:
        List[List[LayoutBlock]] — one inner list per PDF page.

    Raises:
        ImportError  if PyMuPDF (fitz) is not installed.
        RuntimeError if the PDF cannot be opened.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required for PDF rasterisation. "
            "Install it with: pip install pymupdf"
        ) from exc

    try:
        pdf = fitz.open(stream=raw_bytes, filetype="pdf")
    except Exception as exc:
        raise RuntimeError(f"Failed to open PDF for layout detection: {exc}") from exc

    pages_blocks: List[List[LayoutBlock]] = []

    try:
        for page_idx, page in enumerate(pdf, start=1):
            try:
                pix = page.get_pixmap(dpi=dpi)
                from PIL import Image
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                blocks = run_layout_detection_on_image(img, page_number=page_idx, model_id=model_id)
                pages_blocks.append(blocks)
                logger.info(f"PDF page {page_idx}: detected {len(blocks)} block(s).")
            except Exception as exc:
                logger.warning(
                    f"Layout detection failed for PDF page {page_idx}: {exc}. "
                    "Using whole-page fallback."
                )
                pages_blocks.append(_fallback_full_page_block(page_idx))
    finally:
        pdf.close()

    return pages_blocks


def crop_image_to_block(image, block: LayoutBlock):
    """
    Crop a PIL image to the bounding box of a LayoutBlock.

    Coordinates are clamped to the image dimensions so an out-of-bounds
    bbox from the VLM never raises an exception.

    Args:
        image: PIL.Image.Image of the full page.
        block: LayoutBlock whose bbox defines the crop area.

    Returns:
        PIL.Image.Image — the cropped region.
    """
    w, h = image.size
    x0, y0, x1, y1 = block.bbox
    x0 = max(0, min(x0, w))
    y0 = max(0, min(y0, h))
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    return image.crop((x0, y0, x1, y1))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_layout_blocks(raw_blocks: List[dict], page_number: int) -> List[LayoutBlock]:
    """Convert raw VLM output dicts to typed LayoutBlock instances."""
    return [
        LayoutBlock(
            block_id=str(uuid.uuid4()),
            label=b.get("label", "unknown"),
            bbox=b["bbox"],
            confidence=float(b.get("confidence", 1.0)),
            page_number=page_number,
            metadata={"vlm_raw": b},
        )
        for b in raw_blocks
    ]


def _fallback_full_page_block(page_number: int) -> List[LayoutBlock]:
    """
    Return a single full-page LayoutBlock when detection fails entirely.
    The bbox uses a generic A4-ish pixel size at the default rasterisation DPI.
    """
    # A4 at 150 DPI ≈ 1240 × 1754 px
    return [
        LayoutBlock(
            block_id=str(uuid.uuid4()),
            label="full_page",
            bbox=[0, 0, 1240, 1754],
            confidence=0.0,
            page_number=page_number,
            metadata={"fallback": True},
        )
    ]
