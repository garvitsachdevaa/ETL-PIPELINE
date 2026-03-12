import uuid
import re
import logging
from typing import List, Dict, Any, Tuple
from handlers.text_schema import TextSection, TextDocument
from handlers.parsers.plain import parse_plain
from handlers.parsers.html import parse_html
from handlers.parsers.csv import parse_csv
from handlers.parsers.json import parse_json
from handlers.parsers.markdown import parse_markdown

logger = logging.getLogger(__name__)

def handle_mixed(doc) -> TextDocument:
    """Handle files with multiple embedded formats"""
    try:
        raw_text = _extract_text(doc)
        
        # Detect language if not provided
        language = doc.language or _detect_language(raw_text)
        
        # Analyze and extract multiple formats from the content
        mixed_sections = _parse_mixed_content(raw_text, doc.mime_type)
        
        return TextDocument(
            document_id=doc.document_id,
            language=language,
            sections=mixed_sections,
            raw_text=raw_text,
            metadata={
                'mime_type': doc.mime_type,
                'encoding': doc.encoding,
                'file_size': len(doc.raw_bytes) if doc.raw_bytes else len(raw_text or ''),
                'sections_count': len(mixed_sections),
                'extraction_method': 'multi_format_analysis',
                'detected_formats': _get_format_summary(mixed_sections)
            }
        )
    except Exception as e:
        logger.error(f"Failed to handle mixed document {doc.document_id}: {e}")
        raise RuntimeError(f"Mixed handler failed: {e}")

def _extract_text(doc) -> str:
    """Extract text with robust encoding detection"""
    if doc.raw_text is not None:
        return doc.raw_text
    
    if doc.raw_bytes:
        try:
            import chardet
            encoding = chardet.detect(doc.raw_bytes)["encoding"] or "utf-8"
            return doc.raw_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            return doc.raw_bytes.decode("utf-8", errors="replace")
    
    return ""

def _detect_language(text: str) -> str:
    """Simple language detection"""
    return "en"  # Placeholder implementation

def _parse_mixed_content(raw_text: str, mime_type: str = None) -> List[TextSection]:
    """Parse content that may contain multiple embedded formats"""
    if not raw_text or not raw_text.strip():
        return [TextSection(
            section_id=str(uuid.uuid4()),
            format_type="mixed",
            content="Empty document",
            metadata={"empty": True, "section_type": "empty"}
        )]
    
    # Normalize line endings
    normalized_text = raw_text.replace('\r\n', '\n')
    
    # Detect and extract different format regions
    format_regions = _detect_format_regions(normalized_text)
    
    sections = []
    
    # Add document summary first
    if len(format_regions) > 1:
        summary = _create_mixed_summary(normalized_text, format_regions)
        sections.append(summary)
    
    # Process each format region
    for region in format_regions:
        region_sections = _process_format_region(region)
        sections.extend(region_sections)
    
    return sections

def _detect_format_regions(text: str) -> List[Dict[str, Any]]:
    """Detect different format regions within the text"""
    regions = []
    
    # XML detection (before HTML to avoid confusion)
    xml_regions = _find_xml_regions(text)
    regions.extend(xml_regions)
    
    # HTML detection (excluding XML)
    html_regions = _find_html_regions(text)
    regions.extend(html_regions)
    
    # Code block detection (markdown style)
    code_regions = _find_code_block_regions(text)
    regions.extend(code_regions)
    
    # JSON detection
    json_regions = _find_json_regions(text)
    regions.extend(json_regions)
    
    # CSV detection (tabular data)
    csv_regions = _find_csv_regions(text)
    regions.extend(csv_regions)
    
    # Sort regions by start position
    regions.sort(key=lambda x: x['start'])
    
    # Remove overlapping regions (keep higher confidence ones)
    regions = _remove_overlapping_regions(regions)
    
    # Fill gaps with plain text regions
    filled_regions = _fill_text_gaps(text, regions)
    
    return filled_regions

def _remove_overlapping_regions(regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove overlapping regions, keeping higher confidence ones"""
    if not regions:
        return regions
        
    non_overlapping = []
    
    for region in regions:
        # Check if this region overlaps with any existing non-overlapping region
        overlaps = False
        for existing in non_overlapping:
            if (region['start'] < existing['end'] and region['end'] > existing['start']):
                # There's overlap - keep the one with higher confidence
                if region['confidence'] > existing['confidence']:
                    non_overlapping.remove(existing)
                    non_overlapping.append(region)
                overlaps = True
                break
        
        if not overlaps:
            non_overlapping.append(region)
    
    return sorted(non_overlapping, key=lambda x: x['start'])

def _find_html_regions(text: str) -> List[Dict[str, Any]]:
    """Find HTML content regions (excluding XML)"""
    regions = []
    
    # Look for HTML tags, but exclude XML
    html_pattern = r'<[^>]+>.*?</[^>]+>'
    matches = re.finditer(html_pattern, text, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        content = match.group()
        # Skip if this is XML content
        if _is_xml_content(content):
            continue
            
        regions.append({
            'format': 'html',
            'start': match.start(),
            'end': match.end(),
            'content': content,
            'confidence': _calculate_html_confidence(content)
        })
    
    return regions

def _find_xml_regions(text: str) -> List[Dict[str, Any]]:
    """Find XML content regions"""
    regions = []
    
    # Strategy 1: Find XML declaration and everything until next format
    xml_decl_pattern = r'<\?xml.*?\?>.*?(?=\n\s*(?:<form|\{|"|\w+:)|\n\s*$)'
    matches = re.finditer(xml_decl_pattern, text, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        content = match.group().strip()
        if _is_xml_content(content):
            regions.append({
                'format': 'xml',
                'start': match.start(),
                'end': match.end(),
                'content': content,
                'confidence': 0.95
            })
    
    # Strategy 2: Find complete XML blocks with proper nesting
    # Look for XML tags that have proper structure
    xml_block_pattern = r'<([a-zA-Z_][a-zA-Z0-9_-]*)\b[^>]*>.*?</\1>'
    matches = re.finditer(xml_block_pattern, text, re.DOTALL | re.IGNORECASE)
    
    for match in matches:
        content = match.group().strip()
        if _is_xml_content(content) and len(content) > 50:  # Only substantial XML blocks
            # Check if already covered by XML declaration region
            overlap = any(r['start'] <= match.start() < r['end'] for r in regions)
            if not overlap:
                regions.append({
                    'format': 'xml',
                    'start': match.start(),
                    'end': match.end(),
                    'content': content,
                    'confidence': 0.8
                })
    
    return regions

def _is_xml_content(content: str) -> bool:
    """Check if content is XML rather than HTML"""
    content_lower = content.lower()
    
    # XML indicators
    xml_indicators = [
        content_lower.startswith('<?xml'),
        'xmlns:' in content_lower,
        '/>' in content,  # Self-closing XML tags
        # Common XML tag names that aren't HTML
        any(tag in content_lower for tag in ['<configuration>', '<database>', '<connection>', '<pool_size>', '<timeout>', '<cache>', '<type>', '<host>', '<port>'])
    ]
    
    # HTML indicators (if these are present, it's likely HTML not XML)
    html_indicators = [
        any(tag in content_lower for tag in ['<div>', '<span>', '<p>', '<h1>', '<h2>', '<h3>', '<form>', '<input>', '<button>', '<table>']),
        'class=' in content_lower,
        'id=' in content_lower,
        'onclick=' in content_lower
    ]
    
    return any(xml_indicators) and not any(html_indicators)

def _find_code_block_regions(text: str) -> List[Dict[str, Any]]:
    """Find code block regions (```language ... ```)"""
    regions = []
    
    # Markdown-style code blocks
    code_pattern = r'```(\w*)\n(.*?)\n```'
    matches = re.finditer(code_pattern, text, re.DOTALL)
    
    for match in matches:
        language = match.group(1) or 'text'
        code_content = match.group(2)
        
        regions.append({
            'format': 'code',
            'start': match.start(),
            'end': match.end(),
            'content': match.group(),
            'language': language,
            'code_content': code_content,
            'confidence': 0.9
        })
    
    return regions

def _find_json_regions(text: str) -> List[Dict[str, Any]]:
    """Find JSON content regions"""
    regions = []
    
    # Look for JSON-like structures
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.finditer(json_pattern, text, re.DOTALL)
    
    for match in matches:
        content = match.group().strip()
        if _is_likely_json(content):
            regions.append({
                'format': 'json',
                'start': match.start(),
                'end': match.end(),
                'content': content,
                'confidence': _calculate_json_confidence(content)
            })
    
    return regions

def _find_csv_regions(text: str) -> List[Dict[str, Any]]:
    """Find CSV/tabular data regions"""
    regions = []
    
    lines = text.split('\n')
    current_region = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            if current_region:
                current_region['end_line'] = i - 1
                regions.append(current_region)
                current_region = None
            continue
            
        # Check if line looks like CSV (has separators)
        if _is_csv_like(line):
            if not current_region:
                current_region = {
                    'format': 'csv',
                    'start_line': i,
                    'lines': [],
                    'confidence': 0.0
                }
            current_region['lines'].append(line)
            current_region['confidence'] += 0.1
        else:
            if current_region:
                current_region['end_line'] = i - 1
                regions.append(current_region)
                current_region = None
    
    # Handle final region
    if current_region:
        current_region['end_line'] = len(lines) - 1
        regions.append(current_region)
    
    # Convert line-based regions to character-based
    char_regions = []
    for region in regions:
        if region['confidence'] > 0.3:  # Only include likely CSV regions
            start_pos = sum(len(lines[i]) + 1 for i in range(region['start_line']))
            end_pos = sum(len(lines[i]) + 1 for i in range(region['end_line'] + 1))
            
            char_regions.append({
                'format': 'csv',
                'start': start_pos,
                'end': end_pos,
                'content': '\n'.join(region['lines']),
                'confidence': min(region['confidence'], 1.0)
            })
    
    return char_regions

def _fill_text_gaps(text: str, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fill gaps between format regions with plain text"""
    if not regions:
        return [{
            'format': 'text',
            'start': 0,
            'end': len(text),
            'content': text,
            'confidence': 1.0
        }]
    
    filled_regions = []
    last_end = 0
    
    for region in regions:
        # Add text region before this format region
        if region['start'] > last_end:
            gap_content = text[last_end:region['start']].strip()
            if gap_content:
                filled_regions.append({
                    'format': 'text',
                    'start': last_end,
                    'end': region['start'],
                    'content': gap_content,
                    'confidence': 1.0
                })
        
        # Add the format region
        filled_regions.append(region)
        last_end = region['end']
    
    # Add final text region if needed
    if last_end < len(text):
        final_content = text[last_end:].strip()
        if final_content:
            filled_regions.append({
                'format': 'text',
                'start': last_end,
                'end': len(text),
                'content': final_content,
                'confidence': 1.0
            })
    
    return filled_regions

def _process_format_region(region: Dict[str, Any]) -> List[TextSection]:
    """Process a detected format region"""
    format_type = region['format']
    content = region['content']
    
    try:
        if format_type == 'html':
            # For HTML regions in mixed docs, preserve structure instead of parsing
            # This keeps the HTML intact rather than extracting just text
            sections = [_create_html_block_section(region)]
        elif format_type == 'xml':
            # Treat XML as structured code block, not HTML
            sections = [_create_xml_section(region)]
        elif format_type == 'json':
            sections = parse_json(content)
        elif format_type == 'csv':
            sections = parse_csv(content)
        elif format_type == 'code':
            sections = [_create_code_section(region)]
        else:  # text
            sections = parse_plain(content)
            
        # Add region metadata to all sections
        for section in sections:
            section.metadata.update({
                'region_format': format_type,
                'region_confidence': region.get('confidence', 1.0),
                'region_start': region.get('start', 0),
                'region_end': region.get('end', 0)
            })
            
        return sections
        
    except Exception as e:
        logger.warning(f"Failed to parse {format_type} region: {e}")
        # Fallback to plain text
        return [TextSection(
            section_id=str(uuid.uuid4()),
            format_type="mixed",
            content=content,
            metadata={
                'section_type': 'unparsed_region',
                'original_format': format_type,
                'parse_error': str(e),
                'region_confidence': region.get('confidence', 1.0)
            }
        )]

def _create_xml_section(region: Dict[str, Any]) -> TextSection:
    """Create a section for XML content"""
    return TextSection(
        section_id=str(uuid.uuid4()),
        format_type="mixed",
        content=region['content'],
        metadata={
            'section_type': 'xml_block',
            'region_confidence': region.get('confidence', 1.0),
            'char_count': len(region['content']),
            'preserved_structure': True
        }
    )

def _create_html_block_section(region: Dict[str, Any]) -> TextSection:
    """Create a section for HTML content, preserving structure"""
    return TextSection(
        section_id=str(uuid.uuid4()),
        format_type="mixed", 
        content=region['content'],
        metadata={
            'section_type': 'html_block',
            'region_confidence': region.get('confidence', 1.0),
            'char_count': len(region['content']),
            'preserved_structure': True
        }
    )

def _create_grouped_html_section(html_sections: List[TextSection], region: Dict[str, Any]) -> TextSection:
    """Group multiple HTML sections into one cohesive section"""
    combined_content = region['content']  # Use original HTML content
    combined_metadata = {
        'section_type': 'html_block',
        'region_confidence': region.get('confidence', 1.0),
        'char_count': len(combined_content),
        'sub_sections': len(html_sections),
        'preserved_structure': True
    }
    
    return TextSection(
        section_id=str(uuid.uuid4()),
        format_type="mixed",
        content=combined_content,
        metadata=combined_metadata
    )

def _create_code_section(region: Dict[str, Any]) -> TextSection:
    """Create a section for code blocks"""
    return TextSection(
        section_id=str(uuid.uuid4()),
        format_type="mixed",
        content=region.get('code_content', region['content']),
        metadata={
            'section_type': 'code_block',
            'language': region.get('language', 'text'),
            'region_confidence': region.get('confidence', 1.0),
            'char_count': len(region['content'])
        }
    )

def _create_mixed_summary(text: str, regions: List[Dict[str, Any]]) -> TextSection:
    """Create summary section for mixed format document"""
    format_counts = {}
    for region in regions:
        fmt = region['format']
        format_counts[fmt] = format_counts.get(fmt, 0) + 1
    
    total_regions = len(regions)
    word_count = len(text.split())
    
    summary_parts = [
        f"Mixed format document contains {total_regions} regions",
        f"Total: {word_count} words, {len(text)} characters"
    ]
    
    format_summary = ", ".join([f"{count} {fmt}" for fmt, count in format_counts.items()])
    summary_parts.append(f"Formats detected: {format_summary}")
    
    return TextSection(
        section_id=str(uuid.uuid4()),
        format_type="mixed",
        content="\n".join(summary_parts),
        metadata={
            'section_type': 'summary',
            'total_regions': total_regions,
            'total_words': word_count,
            'total_chars': len(text),
            'format_counts': format_counts,
            'extraction_method': 'multi_format_analysis'
        }
    )

def _get_format_summary(sections: List[TextSection]) -> Dict[str, int]:
    """Get summary of detected formats"""
    format_counts = {}
    for section in sections:
        region_format = section.metadata.get('region_format', 'unknown')
        format_counts[region_format] = format_counts.get(region_format, 0) + 1
    return format_counts

# Helper functions for format detection confidence

def _calculate_html_confidence(content: str) -> float:
    """Calculate confidence that content is HTML"""
    tag_count = len(re.findall(r'<[^>]+>', content))
    if tag_count > 2:
        return min(tag_count * 0.2, 1.0)
    return 0.3

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

def _calculate_json_confidence(content: str) -> float:
    """Calculate confidence that content is JSON"""
    try:
        import json
        json.loads(content)
        return 1.0
    except:
        # Partial JSON patterns
        score = 0.0
        if re.search(r'"[^"]+"\s*:', content): score += 0.3
        if re.search(r'\[\s*\{', content): score += 0.2
        if re.search(r'\}\s*,\s*\{', content): score += 0.2
        return min(score, 0.8)

def _is_csv_like(line: str) -> bool:
    """Check if line looks like CSV"""
    separators = [',', '\t', '|', ';']
    return any(sep in line and line.count(sep) >= 2 for sep in separators)
