import io
import logging
from typing import List, Optional

from PIL import Image

from ingestion.schemas import DocumentObject
from handlers.binary_schema import BinaryDocument, Page, Block
from handlers.ocr.dispatcher import run_ocr
from handlers.docx_handler import handle_docx
from handlers.xlsx_handler import handle_xlsx
from handlers.vlm import run_layout_detection_on_pdf, run_layout_detection_on_image

logger = logging.getLogger(__name__)

# Supported binary formats
SUPPORTED_FORMATS = {
    'application/pdf': 'pdf',
    'image/jpeg': 'image',
    'image/png': 'image',
    'image/tiff': 'image',
    'image/bmp': 'image',
    'image/gif': 'image',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
    'application/vnd.ms-excel': 'xlsx'
}

def handle_binary(doc: DocumentObject) -> BinaryDocument:
    """Handle binary documents with format-specific processing"""
    try:
        if doc.mime_type not in SUPPORTED_FORMATS:
            logger.warning(f"Unsupported binary format: {doc.mime_type}")
            return _create_empty_binary_doc(doc, f"Unsupported format: {doc.mime_type}")

        format_type = SUPPORTED_FORMATS[doc.mime_type]

        if format_type == 'docx':
            return handle_docx(doc)

        if format_type == 'xlsx':
            return handle_xlsx(doc)

        if format_type in ('pdf', 'image'):
            # ── Step 1: PaliGemma visual layout detection ────────────────────
            # Rasterises the page(s) to pixels and detects column / header /
            # table regions visually — works regardless of how the PDF was
            # authored (no dependence on internal text-stream structure).
            layout_blocks = _run_vlm_layout(doc, format_type)

            # ── Free PaliGemma VRAM before Chandra loads ─────────────────────
            # L4 GPU has 22 GB. PaliGemma (~6 GB) + Chandra (~7 GB) fit, but
            # fragmentation pushes active usage to ~21 GB causing Chandra OOM.
            # Unloading PaliGemma weights after layout detection frees ~6 GB
            # before Chandra needs it, keeping peak usage under ~8 GB.
            try:
                from handlers.vlm import unload_model
                unload_model()
                logger.info("PaliGemma unloaded from VRAM before Chandra OCR.")
            except Exception as _oom_e:
                logger.warning(f"Could not unload PaliGemma ({_oom_e}); continuing.")

            # ── Step 2: Chandra OCR per detected block ───────────────────────
            # If layout_blocks is None (PaliGemma failed) Chandra falls back
            # to whole-page OCR automatically via run_ocr.
            pages = run_ocr(doc, layout_blocks=layout_blocks)

            vlm_guided = layout_blocks is not None
            extraction_method = "vlm_layout_ocr" if vlm_guided else "chandra_ocr"
            logger.info(
                f"handle_binary {doc.document_id}: format={format_type} "
                f"vlm_guided={vlm_guided} pages={len(pages)}"
            )
            return BinaryDocument(
                document_id=doc.document_id,
                pages=pages,
                metadata={
                    "source_format": doc.detected_format,
                    "mime_type": doc.mime_type,
                    "format_type": format_type,
                    "pages_count": len(pages),
                    "file_size": len(doc.raw_bytes),
                    "extraction_method": extraction_method,
                    "vlm_layout_guided": vlm_guided,
                },
            )

        return _create_empty_binary_doc(doc, f"No handler for format: {format_type}")

    except Exception as e:
        logger.error(f"Failed to handle binary document {doc.document_id}: {e}")
        return _create_empty_binary_doc(doc, f"Processing failed: {str(e)}")


# ---------------------------------------------------------------------------
# VLM layout detection — PaliGemma rasterises + detects blocks, no GPU bypass
# ---------------------------------------------------------------------------

def _run_vlm_layout(doc: DocumentObject, format_type: str):
    """
    Run PaliGemma visual layout detection on the document.

    PaliGemma rasterises the page(s) to pixel images and detects visually
    distinct content blocks (columns, headers, tables, figures) by reading
    the rendered pixels — not the PDF's internal text stream.  This means
    multi-column PDFs, scanned PDFs, and image documents are all handled
    identically and correctly.

    Returns:
        List[List[LayoutBlock]] for PDFs  (one inner list per page)
        List[LayoutBlock]       for images
        None                    on any error (run_ocr falls back to whole-page)
    """
    try:
        if format_type == 'pdf':
            return run_layout_detection_on_pdf(doc.raw_bytes)
        else:
            img = Image.open(io.BytesIO(doc.raw_bytes))
            return run_layout_detection_on_image(img)
    except Exception as exc:
        logger.warning(
            f"PaliGemma layout detection failed for {doc.document_id} "
            f"({exc}); falling back to whole-page Chandra OCR."
        )
        return None


def _extract_pdf_with_pymupdf(doc: DocumentObject) -> Optional[BinaryDocument]:
    """
    Extract text blocks from a text-based PDF using PyMuPDF (fitz).

    Uses PaliGemma for visual layout detection (see _run_vlm_layout above).
    This function is kept for reference only and is no longer called.
    """
    pass  # superseded by _run_vlm_layout + run_ocr


def _create_empty_binary_doc(doc: DocumentObject, error_message: str) -> BinaryDocument:
    """Create an empty binary document with error information"""
    return BinaryDocument(
        document_id=doc.document_id,
        pages=[],
        metadata={
            "source_format": doc.detected_format,
            "mime_type": doc.mime_type,
            "error": error_message,
            "file_size": len(doc.raw_bytes) if doc.raw_bytes else 0
        }
    )

def get_supported_formats() -> List[str]:
    """Get list of supported binary formats"""
    return list(SUPPORTED_FORMATS.keys())
