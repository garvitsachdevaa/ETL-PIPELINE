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

# Minimum embedded-text characters across all pages to classify a PDF as
# "text-based" (not scanned).  A scanned page may have stray OCR artifacts
# of 10-50 chars; a real text page typically has hundreds.
_TEXT_PDF_CHAR_THRESHOLD = 150


def handle_binary(doc: DocumentObject) -> BinaryDocument:
    """
    Route binary documents to the correct extraction pipeline.

    ┌──────────────────┬──────────────────┬───────────────────────┐
    │   Text PDF       │  Scanned PDF /   │   DOCX / XLSX         │
    │  (embedded text) │  Image           │                       │
    ├──────────────────┼──────────────────┼───────────────────────┤
    │  Marker          │  DocLayout-YOLO  │  Existing handlers    │
    │  (layout+OCR     │  (layout blocks) │                       │
    │  in one pass)    │  + Chandra OCR   │                       │
    └──────────────────┴──────────────────┴───────────────────────┘
    """
    try:
        if doc.mime_type not in SUPPORTED_FORMATS:
            logger.warning(f"Unsupported binary format: {doc.mime_type}")
            return _create_empty_binary_doc(doc, f"Unsupported format: {doc.mime_type}")

        format_type = SUPPORTED_FORMATS[doc.mime_type]

        if format_type == 'docx':
            return handle_docx(doc)

        if format_type == 'xlsx':
            return handle_xlsx(doc)

        if format_type == 'pdf':
            if _is_text_pdf(doc.raw_bytes):
                # ── Path A: Text PDF → Marker ────────────────────────────────
                # Marker runs layout detection + OCR in a single pass using
                # Surya. Outputs clean Markdown preserving columns, tables,
                # headings. No need for a separate layout step.
                logger.info(
                    f"{doc.document_id}: text-based PDF detected → Marker pipeline."
                )
                return _handle_with_marker(doc)
            else:
                # ── Path B: Scanned PDF → DocLayout-YOLO + Chandra ──────────
                logger.info(
                    f"{doc.document_id}: scanned PDF detected → YOLO+Chandra pipeline."
                )
                return _handle_with_yolo_chandra(doc, 'pdf')

        if format_type == 'image':
            # ── Path C: Image → DocLayout-YOLO + Chandra ────────────────────
            return _handle_with_yolo_chandra(doc, 'image')

        return _create_empty_binary_doc(doc, f"No handler for format: {format_type}")

    except Exception as e:
        logger.error(f"Failed to handle binary document {doc.document_id}: {e}")
        return _create_empty_binary_doc(doc, f"Processing failed: {str(e)}")


# ---------------------------------------------------------------------------
# Path A — Marker (text PDFs)
# ---------------------------------------------------------------------------

def _handle_with_marker(doc: DocumentObject) -> BinaryDocument:
    """
    Convert a text-based PDF with Marker.
    Falls back to YOLO+Chandra if Marker fails.
    """
    try:
        from handlers.marker_handler import convert_pdf_with_marker
        return convert_pdf_with_marker(doc)
    except Exception as exc:
        logger.warning(
            f"Marker failed for {doc.document_id} ({exc}); "
            "falling back to YOLO+Chandra."
        )
        return _handle_with_yolo_chandra(doc, 'pdf')


# ---------------------------------------------------------------------------
# Path B/C — DocLayout-YOLO + Chandra (scanned PDFs and images)
# ---------------------------------------------------------------------------

def _handle_with_yolo_chandra(doc: DocumentObject, format_type: str) -> BinaryDocument:
    """
    Layout detection with DocLayout-YOLO then OCR with Chandra.

    YOLO (~0.5 GB) detects block regions visually.
    Chandra (~7 GB) reads text from each cropped region.
    Total peak VRAM: ~7.5 GB — well within L4's 22.5 GB.
    """
    # Step 1: DocLayout-YOLO visual layout detection
    layout_blocks = _run_vlm_layout(doc, format_type)

    # Step 2: Chandra OCR per detected block
    # layout_blocks=None triggers whole-page OCR fallback in run_ocr
    pages = run_ocr(doc, layout_blocks=layout_blocks)

    vlm_guided        = layout_blocks is not None
    extraction_method = "yolo_layout_ocr" if vlm_guided else "chandra_ocr"
    logger.info(
        f"handle_binary {doc.document_id}: format={format_type} "
        f"vlm_guided={vlm_guided} pages={len(pages)}"
    )
    return BinaryDocument(
        document_id=doc.document_id,
        pages=pages,
        metadata={
            "source_format":     doc.detected_format,
            "mime_type":         doc.mime_type,
            "format_type":       format_type,
            "pages_count":       len(pages),
            "file_size":         len(doc.raw_bytes),
            "extraction_method": extraction_method,
            "vlm_layout_guided": vlm_guided,
        },
    )


# ---------------------------------------------------------------------------
# Layout detection helper (DocLayout-YOLO)
# ---------------------------------------------------------------------------

def _run_vlm_layout(doc: DocumentObject, format_type: str):
    """
    Run DocLayout-YOLO layout detection on the document.

    Rasterises PDF pages to pixel images and detects visually distinct
    content blocks (columns, headers, tables, figures) — works on any PDF
    regardless of how it was authored.

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
            f"DocLayout-YOLO layout detection failed for {doc.document_id} "
            f"({exc}); falling back to whole-page Chandra OCR."
        )
        return None


# ---------------------------------------------------------------------------
# PDF type detection
# ---------------------------------------------------------------------------

def _is_text_pdf(raw_bytes: bytes) -> bool:
    """
    Return True if the PDF has sufficient embedded text to be processed by
    Marker (text-based PDF), False if it appears to be scanned.

    Uses PyMuPDF to count total extracted characters across all pages.
    Scanned pages may have stray OCR artifacts (~10-50 chars); a real text
    page typically has hundreds.
    """
    try:
        import fitz
        pdf = fitz.open(stream=raw_bytes, filetype="pdf")
        total_chars = 0
        for page in pdf:
            total_chars += len(page.get_text().strip())
            if total_chars > _TEXT_PDF_CHAR_THRESHOLD:
                pdf.close()
                return True
        pdf.close()
        return total_chars > _TEXT_PDF_CHAR_THRESHOLD
    except Exception:
        # If we can't open it as a PDF, assume scanned (safe default)
        return False


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
