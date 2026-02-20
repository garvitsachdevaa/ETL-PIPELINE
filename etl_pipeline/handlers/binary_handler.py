import io
import logging
import uuid
from typing import List, Optional

from ingestion.schemas import DocumentObject
from handlers.binary_schema import BinaryDocument, Page, Block
from handlers.ocr.dispatcher import run_ocr
from handlers.docx_handler import handle_docx
from handlers.xlsx_handler import handle_xlsx

logger = logging.getLogger(__name__)

# Minimum characters extracted by PyMuPDF to consider a PDF "text-based"
_MIN_TEXT_CHARS = 50

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

        if format_type == 'pdf':
            # Primary path: PyMuPDF block detection.
            # Reads the PDF's internal vector structure — handles multi-column
            # layouts correctly, instant, no GPU required.
            result = _extract_pdf_with_pymupdf(doc)
            if result is not None:
                logger.info(f"PyMuPDF block extraction used for {doc.document_id}")
                return result

            # Fallback: scanned/image PDF — run Chandra whole-page OCR
            logger.info(f"No selectable text in PDF {doc.document_id}; using Chandra OCR.")
            pages = run_ocr(doc, layout_blocks=None)
            return BinaryDocument(
                document_id=doc.document_id,
                pages=pages,
                metadata={
                    "source_format": doc.detected_format,
                    "mime_type": doc.mime_type,
                    "format_type": "pdf",
                    "pages_count": len(pages),
                    "file_size": len(doc.raw_bytes),
                    "extraction_method": "chandra_ocr_scanned",
                    "vlm_layout_guided": False,
                }
            )

        if format_type == 'image':
            pages = run_ocr(doc, layout_blocks=None)
            return BinaryDocument(
                document_id=doc.document_id,
                pages=pages,
                metadata={
                    "source_format": doc.detected_format,
                    "mime_type": doc.mime_type,
                    "format_type": "image",
                    "pages_count": len(pages),
                    "file_size": len(doc.raw_bytes),
                    "extraction_method": "chandra_ocr",
                    "vlm_layout_guided": False,
                }
            )

        return _create_empty_binary_doc(doc, f"No handler for format: {format_type}")

    except Exception as e:
        logger.error(f"Failed to handle binary document {doc.document_id}: {e}")
        return _create_empty_binary_doc(doc, f"Processing failed: {str(e)}")


# ---------------------------------------------------------------------------
# PyMuPDF block-level extraction (text PDFs, multi-column aware)
# ---------------------------------------------------------------------------

def _extract_pdf_with_pymupdf(doc: DocumentObject) -> Optional[BinaryDocument]:
    """
    Extract text blocks from a text-based PDF using PyMuPDF (fitz).

    Uses fitz.Page.get_text("blocks") which reads the PDF's internal vector
    text structure — columns, tables, headings are preserved as separate
    blocks each with their own bounding box.

    Blocks are sorted into correct reading order for multi-column layouts.

    Returns None if the PDF has no selectable text (scanned/image PDF),
    so the caller can fall through to Chandra OCR.
    """
    try:
        import fitz
    except ImportError:
        logger.warning("PyMuPDF not available; skipping block extraction.")
        return None

    try:
        pdf = fitz.open(stream=doc.raw_bytes, filetype="pdf")
    except Exception as exc:
        logger.warning(f"PyMuPDF could not open PDF ({exc}); falling back to Chandra.")
        return None

    pages = []
    total_chars = 0

    try:
        for page_num, fitz_page in enumerate(pdf, 1):
            page_width  = fitz_page.rect.width
            page_height = fitz_page.rect.height

            # get_text("blocks") → list of (x0,y0,x1,y1, text, block_no, block_type)
            # block_type 0 = text block, 1 = image block
            raw_blocks = fitz_page.get_text("blocks")
            text_blocks = [
                (float(b[0]), float(b[1]), float(b[2]), float(b[3]), b[4])
                for b in raw_blocks
                if b[6] == 0 and b[4].strip()
            ]

            if not text_blocks:
                continue

            ordered = _sort_into_reading_order(text_blocks, page_width)

            blocks = []
            for i, (x0, y0, x1, y1, text) in enumerate(ordered):
                text = text.strip()
                if not text:
                    continue
                total_chars += len(text)
                blocks.append(Block(
                    block_id=str(uuid.uuid4()),
                    title="",
                    label=_guess_label(text, y0, page_height),
                    bbox=[int(x0), int(y0), int(x1), int(y1)],
                    raw_text=text,
                    confidence=1.0,
                    metadata={"engine": "pymupdf", "page": page_num},
                ))

            if blocks:
                pages.append(Page(
                    page_id=str(uuid.uuid4()),
                    page_number=page_num,
                    blocks=blocks,
                    metadata={"pdf": True, "extraction_method": "pymupdf_blocks",
                              "layout_guided": True},
                ))
    finally:
        pdf.close()

    if total_chars < _MIN_TEXT_CHARS:
        return None  # Scanned PDF — fall through to Chandra

    return BinaryDocument(
        document_id=doc.document_id,
        pages=pages,
        metadata={
            "source_format": doc.detected_format,
            "mime_type": doc.mime_type,
            "format_type": "pdf",
            "pages_count": len(pages),
            "file_size": len(doc.raw_bytes),
            "extraction_method": "pymupdf_blocks",
            "vlm_layout_guided": False,
        },
    )


def _sort_into_reading_order(
    blocks: List[tuple], page_width: float
) -> List[tuple]:
    """
    Sort text blocks into natural reading order, handling multi-column layouts.

    Strategy:
    1. Detect column boundaries by finding x-position clusters among block starts.
    2. Assign each block to its column.
    3. Sort by column (left → right), then by y0 (top → bottom) within each column.
    """
    if not blocks:
        return blocks

    x0_positions = sorted(set(round(b[0] / 5) * 5 for b in blocks))  # bucket to 5px grid

    # Find significant horizontal gaps to identify column starts
    gap_threshold = page_width * 0.08
    col_starts = [x0_positions[0]]
    for i in range(1, len(x0_positions)):
        if x0_positions[i] - x0_positions[i - 1] > gap_threshold:
            col_starts.append(x0_positions[i])

    # Merge very close column starts (within 10% of page width)
    merged: List[float] = [col_starts[0]]
    for cs in col_starts[1:]:
        if cs - merged[-1] > page_width * 0.10:
            merged.append(cs)

    if len(merged) == 1:
        # Single column — sort top to bottom only
        return sorted(blocks, key=lambda b: b[1])

    # Multi-column: assign block to the rightmost column_start <= block.x0
    def col_index(block: tuple) -> int:
        idx = 0
        for i, cs in enumerate(merged):
            if block[0] >= cs - page_width * 0.05:
                idx = i
        return idx

    return sorted(blocks, key=lambda b: (col_index(b), b[1]))


def _guess_label(text: str, y0: float, page_height: float) -> str:
    """Assign a semantic label based on block position and content length."""
    lines = [l for l in text.split("\n") if l.strip()]
    first_line = lines[0].strip() if lines else ""

    if y0 > page_height * 0.88:
        return "footer"
    if y0 < page_height * 0.12 and len(first_line) < 120:
        return "header"
    if len(lines) == 1 and len(first_line) < 80:
        return "heading"
    return "body"


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
