from typing import Optional

from handlers.ocr.image import ocr_image
from handlers.ocr.pdf import ocr_pdf


def run_ocr(doc, layout_blocks=None):
    """
    Route a document to the correct OCR handler.

    Args:
        doc:           DocumentObject (detected_format, mime_type, raw_bytes).
        layout_blocks: Optional VLM layout blocks.
                       - images: List[LayoutBlock]
                       - PDFs:   List[List[LayoutBlock]] (one list per page)
                       Pass None to fall back to whole-page OCR.
    Returns:
        List[Page]
    """
    if doc.detected_format == "image":
        return ocr_image(doc, layout_blocks=layout_blocks)

    if doc.detected_format == "document":
        if doc.mime_type == "application/pdf":
            return ocr_pdf(doc, layout_blocks=layout_blocks)
        raise ValueError(
            f"Document type {doc.mime_type} should be handled by a specific handler"
        )

    raise ValueError(f"Unsupported format for OCR: {doc.detected_format}")
