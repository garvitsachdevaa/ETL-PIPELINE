import io
import logging
from typing import List, Optional

from PIL import Image
from ingestion.schemas import DocumentObject
from handlers.binary_schema import BinaryDocument, Page
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
        # Check if format is supported
        if doc.mime_type not in SUPPORTED_FORMATS:
            logger.warning(f"Unsupported binary format: {doc.mime_type}")
            return _create_empty_binary_doc(doc, f"Unsupported format: {doc.mime_type}")
        
        format_type = SUPPORTED_FORMATS[doc.mime_type]
        
        # Handle DOCX files with python-docx
        if format_type == 'docx':
            return handle_docx(doc)
        
        # Handle XLSX files with openpyxl
        if format_type == 'xlsx':
            return handle_xlsx(doc)
        
        # Handle PDF and image formats: VLM layout detection → block-aware OCR
        if format_type in ['pdf', 'image']:
            layout_blocks = _run_vlm_layout(doc, format_type)
            pages = run_ocr(doc, layout_blocks=layout_blocks)

            return BinaryDocument(
                document_id=doc.document_id,
                pages=pages,
                metadata={
                    "source_format": doc.detected_format,
                    "mime_type": doc.mime_type,
                    "format_type": format_type,
                    "pages_count": len(pages),
                    "file_size": len(doc.raw_bytes),
                    "extraction_method": "vlm_layout_ocr" if layout_blocks else "ocr_fallback",
                    "vlm_layout_guided": layout_blocks is not None,
                }
            )
        
        return _create_empty_binary_doc(doc, f"No handler for format: {format_type}")
        
    except Exception as e:
        logger.error(f"Failed to handle binary document {doc.document_id}: {e}")
        return _create_empty_binary_doc(doc, f"Processing failed: {str(e)}")

def _run_vlm_layout(doc: DocumentObject, format_type: str) -> Optional[List]:
    """
    Run VLM layout detection to get bounding boxes per content block.

    Returns:
        - PDFs:   List[List[LayoutBlock]] — one inner list per page
        - images: List[LayoutBlock]
        - None   — if VLM is unavailable or fails (triggers whole-page OCR fallback)
    """
    try:
        if format_type == 'pdf':
            return run_layout_detection_on_pdf(doc.raw_bytes)
        else:
            img = Image.open(io.BytesIO(doc.raw_bytes))
            return run_layout_detection_on_image(img)
    except Exception as exc:
        logger.warning(
            f"VLM layout detection failed for {doc.document_id} ({exc}); "
            "falling back to whole-page OCR."
        )
        return None


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
