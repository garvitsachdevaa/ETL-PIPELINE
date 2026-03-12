print("detector loaded")

import mimetypes
import chardet
import os
import re

TEXT_MIME_PREFIXES = [
    "text/"
]

STRUCTURED_MIME_TYPES = [
    "application/json",
    "text/csv",
    "text/html",
    "application/xml",
    "text/markdown",
    "application/vnd.ms-excel"  # Added for CSV files on Windows
]

DOCUMENT_MIME_TYPES = [
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel"
]

IMAGE_MIME_PREFIXES = [
    "image/"
]

# Extension-based mapping for reliable detection
EXTENSION_MAPPINGS = {
    '.csv': ('structured', 'text/csv'),
    '.json': ('structured', 'application/json'),
    '.html': ('structured', 'text/html'),
    '.htm': ('structured', 'text/html'),
    '.xml': ('structured', 'application/xml'),
    '.md': ('structured', 'text/markdown'),
    '.markdown': ('structured', 'text/markdown'),
    '.txt': ('text', 'text/plain'),
    '.pdf': ('document', 'application/pdf'),
    '.docx': ('document', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'),
    '.xlsx': ('document', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
    '.xls': ('document', 'application/vnd.ms-excel'),
    '.jpg': ('image', 'image/jpeg'),
    '.jpeg': ('image', 'image/jpeg'),
    '.png': ('image', 'image/png'),
    '.gif': ('image', 'image/gif'),
    '.bmp': ('image', 'image/bmp'),
    '.tiff': ('image', 'image/tiff')
}

def detect_format(source_name, raw_bytes, raw_text):
    # If raw_text is provided, treat as plain text initially
    if raw_text is not None:
        # Check if the text contains multiple formats
        if _has_mixed_formats(raw_text):
            return "mixed", "text/mixed", "utf-8"
        return "text", "text/plain", "utf-8"

    # Get encoding for byte content
    encoding = chardet.detect(raw_bytes)["encoding"] if raw_bytes else "utf-8"
    
    # Try to decode bytes to check for mixed content
    if raw_bytes:
        try:
            decoded_text = raw_bytes.decode(encoding or 'utf-8')
            if _has_mixed_formats(decoded_text):
                return "mixed", "text/mixed", encoding
        except (UnicodeDecodeError, AttributeError):
            pass
    
    # Step 1: Check file extension first (most reliable)
    if source_name:
        ext = os.path.splitext(source_name.lower())[1]
        if ext in EXTENSION_MAPPINGS:
            format_type, mime_type = EXTENSION_MAPPINGS[ext]
            return format_type, mime_type, encoding
    
    # Step 2: Fall back to mimetypes.guess_type
    mime_type, _ = mimetypes.guess_type(source_name) if source_name else (None, None)
    
    if mime_type:
        # Check if it's an image
        if any(mime_type.startswith(p) for p in IMAGE_MIME_PREFIXES):
            return "image", mime_type, encoding
        
        # Check if it's a document
        if mime_type in DOCUMENT_MIME_TYPES:
            return "document", mime_type, encoding
        
        # Check if it's structured data
        if mime_type in STRUCTURED_MIME_TYPES:
            return "structured", mime_type, encoding
        
        # Check if it's text
        if any(mime_type.startswith(p) for p in TEXT_MIME_PREFIXES):
            return "text", mime_type, encoding
    
    # Step 3: Content-based detection for unknown types
    if raw_bytes:
        try:
            # Try to decode as text
            decoded_text = raw_bytes.decode(encoding or 'utf-8')
            # If successful and looks like text content, classify as text
            if len(decoded_text.strip()) > 0 and decoded_text.isprintable():
                return "text", "text/plain", encoding
        except (UnicodeDecodeError, AttributeError):
            pass
    
    # Final fallback: unknown format (will trigger error in routing)
    return "unknown", mime_type, encoding

def _has_mixed_formats(text: str) -> bool:
    """Detect if text contains multiple embedded formats"""
    if not text or len(text) < 100:  # Too short to be mixed format
        return False
    
    # FIRST: Check if entire content is a pure format - if so, NOT mixed
    if _is_pure_json(text.strip()):
        return False
    
    if _is_pure_html(text.strip()):
        return False
        
    if _is_pure_csv(text.strip()):
        return False
    
    # THEN: Look for multiple embedded formats
    format_indicators = 0
    
    # Check for code blocks
    if re.search(r'```\w*\n.*?\n```', text, re.DOTALL):
        format_indicators += 1
    
    # Check for JSON-like content (but not if entire file is JSON)
    json_matches = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_matches:
        # Only count as mixed if there's surrounding non-JSON content
        total_json_length = sum(len(match) for match in json_matches)
        if total_json_length < len(text.strip()) * 0.8:  # JSON is less than 80% of content
            format_indicators += 1
    
    # Check for HTML tags (but not XML)
    html_matches = re.findall(r'<[^>]+>.*?</[^>]+>', text, re.DOTALL | re.IGNORECASE)
    has_html = False
    for match in html_matches:
        # Skip XML-style content
        if not _is_xml_like_content(match):
            has_html = True
            break
    
    if has_html:
        # Only count if there's substantial non-HTML content
        html_content = re.sub(r'<[^>]*>', '', text)
        if len(html_content.strip()) > len(text) * 0.3:  # At least 30% non-HTML content
            format_indicators += 1
    
    # Check for CSV-like tabular data (multiple lines with separators)
    csv_lines = 0
    for line in text.split('\n'):
        line = line.strip()
        if line and _is_csv_like(line):
            csv_lines += 1
    if csv_lines >= 3:  # At least 3 lines that look like CSV
        # Only count if there's non-CSV content too
        total_lines = len([l for l in text.split('\n') if l.strip()])
        if csv_lines < total_lines * 0.8:  # CSV is less than 80% of lines
            format_indicators += 1
    
    # Check for XML content (separate from HTML)
    xml_patterns = [
        r'<\?xml.*?\?>',  # XML declaration
        r'<[a-zA-Z_][a-zA-Z0-9_]*>[^<]*</[a-zA-Z_][a-zA-Z0-9_]*>'  # XML-style tags
    ]
    
    has_xml = False
    for pattern in xml_patterns:
        xml_matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        for match in xml_matches:
            if _is_xml_like_content(match):
                has_xml = True
                break
        if has_xml:
            break
    
    if has_xml:
        # XML in mixed content should always count as a format indicator
        format_indicators += 1
    
    # If we found 2 or more different format indicators, consider it mixed
    return format_indicators >= 2

def _is_pure_json(text: str) -> bool:
    """Check if entire text is valid JSON"""
    try:
        import json
        json.loads(text)
        return True
    except:
        return False

def _is_pure_html(text: str) -> bool:
    """Check if entire text is HTML document"""
    text = text.strip()
    
    # Must be a complete HTML document, not mixed content
    html_document_indicators = [
        text.lower().startswith('<!doctype html'),
        text.lower().startswith('<html'),
        '<head>' in text.lower() and '<body>' in text.lower()
    ]
    
    # If it has clear HTML document structure, check further
    if any(html_document_indicators):
        return True
    
    # For other cases, be more restrictive
    # Don't consider it pure HTML if it has:
    # - XML declarations
    # - JSON objects
    # - Substantial plain text mixed with tags
    if '<?xml' in text.lower():
        return False
    
    if re.search(r'\{[^{}]*"[^"]*"[^{}]*\}', text):
        return False
    
    # If it's mostly HTML tags with minimal content, it could be pure HTML
    tag_content = re.sub(r'<[^>]*>', '', text)
    if len(tag_content.strip()) < len(text) * 0.2:  # Less than 20% actual content
        return True
    
    return False

def _is_pure_csv(text: str) -> bool:
    """Check if entire text is CSV format"""
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if len(lines) < 2:  # Need at least header + 1 data row
        return False
    
    # Check if most lines look like CSV
    csv_lines = sum(1 for line in lines if _is_csv_like(line))
    return csv_lines >= len(lines) * 0.8  # 80% of lines are CSV-like

def _is_likely_json(content: str) -> bool:
    """Check if content is likely JSON"""
    try:
        import json
        json.loads(content)
        return True
    except:
        # Check for JSON-like patterns
        json_patterns = [r'"[^"]+"\s*:', r'\[\s*\{', r'\}\s*,\s*\{']
        return any(re.search(pattern, content) for pattern in json_patterns)

def _is_csv_like(line: str) -> bool:
    """Check if line looks like CSV"""
    separators = [',', '\t', '|', ';']
    return any(sep in line and line.count(sep) >= 2 for sep in separators)

def _is_xml_like_content(content: str) -> bool:
    """Helper to distinguish XML from HTML content"""
    content_lower = content.lower()
    
    # XML indicators
    xml_indicators = [
        content_lower.startswith('<?xml'),
        'xmlns:' in content_lower,
        '/>' in content,  # Self-closing XML tags
        # Common XML tag names that aren't HTML
        any(tag in content_lower for tag in ['<database>', '<configuration>', '<connection>', '<host>', '<port>', '<name>'])
    ]
    
    # HTML indicators (if these are present, it's likely HTML not XML)
    html_indicators = [
        any(tag in content_lower for tag in ['<div>', '<span>', '<p>', '<h1>', '<h2>', '<h3>', '<form>', '<input>', '<button>', '<table>']),
        'class=' in content_lower,
        'id=' in content_lower,
        'onclick=' in content_lower
    ]
    
    return any(xml_indicators) and not any(html_indicators)
