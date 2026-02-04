from handlers.ocr.image import ocr_image
from handlers.ocr.pdf import ocr_pdf

def run_ocr(doc):
    if doc.detected_format == "image":
        return ocr_image(doc)

    if doc.detected_format == "document":
        # Check if it's a PDF
        if doc.mime_type == "application/pdf":
            return ocr_pdf(doc)
        else:
            # For other document types (like DOCX), this should not be called
            # as they should be handled by their specific handlers
            raise ValueError(f"Document type {doc.mime_type} should be handled by specific handler")

    raise ValueError("Unsupported binary format")
