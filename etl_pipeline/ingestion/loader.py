import uuid
from ingestion.detector import detect_format
from ingestion.validator import validate_input
from ingestion.metadata import extract_metadata
from ingestion.router import route
from ingestion.schemas import DocumentObject

def load_input(file=None, text=None):
    if file:
        return file.name, file.read(), None
    if text:
        return "direct_text_input", None, text
    raise ValueError("No input provided")

def ingest(file=None, text=None):
    source_name, raw_bytes, raw_text = load_input(file, text)

    detected_format, mime_type, encoding = detect_format(
        source_name, raw_bytes, raw_text
    )

    validate_input(detected_format, raw_bytes, raw_text)

    metadata = extract_metadata(source_name, raw_bytes)

    return DocumentObject(
        document_id=str(uuid.uuid4()),
        source_name=source_name,
        raw_bytes=raw_bytes,
        raw_text=raw_text,
        detected_format=detected_format,
        mime_type=mime_type,
        encoding=encoding,
        language=None,
        metadata=metadata,
        routing_target=route(detected_format)
    )
