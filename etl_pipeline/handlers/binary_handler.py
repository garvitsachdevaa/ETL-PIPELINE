import logging
from typing import List

from ingestion.schemas import DocumentObject
from handlers.binary_schema import BinaryDocument
from handlers.ocr.dispatcher import run_ocr
from handlers.docx_handler import handle_docx
from handlers.xlsx_handler import handle_xlsx

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
        
        # Handle PDF and image formats via Chandra OCR.
        # Chandra is a VLM-based OCR model that performs layout detection
        # internally and returns labelled, positioned blocks via its chunk
        # output — no separate PaliGemma pass needed.
        if format_type in ['pdf', 'image']:
            pages = run_ocr(doc, layout_blocks=None)

            return BinaryDocument(
                document_id=doc.document_id,
                pages=pages,
                metadata={
                    "source_format": doc.detected_format,
                    "mime_type": doc.mime_type,
                    "format_type": format_type,
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
