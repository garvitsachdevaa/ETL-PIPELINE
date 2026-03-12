import logging
import chardet
from typing import Optional
from ingestion.schemas import DocumentObject
from handlers.text_schema import TextDocument
from handlers.parsers.dispatcher import parse_text

logger = logging.getLogger(__name__)

def handle_text(doc: DocumentObject) -> TextDocument:
    """Handle text-based documents with robust extraction and parsing"""
    try:
        raw_text = _extract_text(doc)
        
        # Detect language if not provided
        language = doc.language or _detect_language(raw_text)
        
        # Parse text into sections based on format
        sections = parse_text(raw_text, doc.mime_type)
        
        return TextDocument(
            document_id=doc.document_id,
            language=language,
            sections=sections,
            raw_text=raw_text,
            metadata={
                'mime_type': doc.mime_type,
                'encoding': doc.encoding,
                'file_size': len(doc.raw_bytes) if doc.raw_bytes else len(raw_text or ''),
                'sections_count': len(sections),
                'extraction_method': 'raw_text' if doc.raw_text else 'byte_decode'
            }
        )
    except Exception as e:
        logger.error(f"Failed to handle text document {doc.document_id}: {e}")
        raise RuntimeError(f"Text handler failed: {e}")

def _extract_text(doc: DocumentObject) -> str:
    """Extract text with robust encoding detection"""
    if doc.raw_text is not None:
        return doc.raw_text

    if not doc.raw_bytes:
        return ""
    
    # Use provided encoding or detect it
    encoding = doc.encoding
    if not encoding:
        encoding = _detect_encoding(doc.raw_bytes)
    
    try:
        return doc.raw_bytes.decode(encoding, errors="replace")
    except (UnicodeDecodeError, LookupError):
        # Fallback to UTF-8 with ignore
        logger.warning(f"Failed to decode with {encoding}, falling back to UTF-8")
        return doc.raw_bytes.decode("utf-8", errors="ignore")

def _detect_encoding(raw_bytes: bytes) -> str:
    """Detect text encoding using chardet"""
    try:
        result = chardet.detect(raw_bytes)
        confidence = result.get('confidence', 0)
        encoding = result.get('encoding', 'utf-8')
        
        # Only use detected encoding if confidence is reasonable
        if confidence > 0.7 and encoding:
            return encoding
    except Exception:
        pass
    
    return 'utf-8'  # Safe fallback

def _detect_language(text: str) -> Optional[str]:
    """Basic language detection (placeholder for more sophisticated detection)"""
    if not text or len(text.strip()) < 10:
        return None
    
    # Simple heuristic - can be enhanced with langdetect or similar
    # For now, assume English for most content
    return 'en'
