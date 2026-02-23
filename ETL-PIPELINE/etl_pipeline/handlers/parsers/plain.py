import uuid
import re
import logging
from typing import List
from handlers.text_schema import TextSection

logger = logging.getLogger(__name__)

def parse_plain(raw_text: str) -> List[TextSection]:
    """Parse plain text with intelligent paragraph and section detection"""
    if not raw_text or not raw_text.strip():
        return [TextSection(
            section_id=str(uuid.uuid4()),
            format_type="text",
            content="Empty document",
            metadata={"empty": True}
        )]

    sections = []
    
    # Try to detect document structure
    structure_type = _detect_structure(raw_text)
    
    if structure_type == "markdown_like":
        sections = _parse_structured_text(raw_text)
    elif structure_type == "list_like":
        sections = _parse_list_content(raw_text)
    else:
        sections = _parse_paragraphs(raw_text)
    
    # Add document summary
    if len(sections) > 1:
        summary = _create_document_summary(raw_text, sections)
        sections.insert(0, summary)
    
    return sections

def _detect_structure(text: str) -> str:
    """Detect the likely structure type of the text"""
    # Normalize line endings for consistent processing
    normalized_text = text.replace('\r\n', '\n')
    lines = normalized_text.split('\n')
    
    # Check for markdown-like headers
    header_pattern = re.compile(r'^#{1,6}\\s+.*$')
    header_count = sum(1 for line in lines if header_pattern.match(line.strip()))
    
    if header_count > 0:
        return "markdown_like"
    
    # Check for list-like structure
    list_pattern = re.compile(r'^\\s*[-*•]\\s+.*$|^\\s*\\d+\\.\\s+.*$')
    list_count = sum(1 for line in lines if list_pattern.match(line))
    
    if list_count > len(lines) * 0.3:  # More than 30% are list items
        return "list_like"
    
    return "plain"

def _parse_structured_text(text: str) -> List[TextSection]:
    """Parse text with header-like structure"""
    sections = []
    lines = text.split('\\n')
    
    current_section = []
    current_header = None
    header_pattern = re.compile(r'^(#{1,6})\\s+(.*)$')
    
    for line in lines:
        line = line.strip()
        header_match = header_pattern.match(line)
        
        if header_match:
            # Save previous section
            if current_section:
                content = '\n'.join(current_section).strip()
                if content:
                    sections.append(TextSection(
                        section_id=str(uuid.uuid4()),
                        format_type="text",
                        content=content,
                        metadata={
                            "section_type": "content",
                            "header": current_header,
                            "word_count": len(content.split())
                        }
                    ))
            
            # Start new section
            header_level = len(header_match.group(1))
            header_text = header_match.group(2)
            current_header = header_text
            current_section = []
            
            # Add header as separate section
            sections.append(TextSection(
                section_id=str(uuid.uuid4()),
                format_type="text",
                content=header_text,
                metadata={
                    "section_type": "header",
                    "level": header_level,
                    "word_count": len(header_text.split())
                }
            ))
        
        elif line:  # Non-empty line
            current_section.append(line)
    
    # Add final section
    if current_section:
        content = '\n'.join(current_section).strip()
        if content:
            sections.append(TextSection(
                section_id=str(uuid.uuid4()),
                format_type="text",
                content=content,
                metadata={
                    "section_type": "content",
                    "header": current_header,
                    "word_count": len(content.split())
                }
            ))
    
    return sections

def _parse_list_content(text: str) -> List[TextSection]:
    """Parse text that appears to be primarily lists"""
    sections = []
    lines = text.split('\\n')
    
    current_list = []
    current_type = None
    list_patterns = {
        'bullet': re.compile(r'^\\s*[-*•]\\s+(.*)$'),
        'numbered': re.compile(r'^\\s*(\\d+)\\.\\s+(.*)$')
    }
    
    for line in lines:
        line_stripped = line.strip()
        
        matched_type = None
        matched_content = None
        
        # Check which pattern matches
        for pattern_type, pattern in list_patterns.items():
            match = pattern.match(line_stripped)
            if match:
                matched_type = pattern_type
                matched_content = match.group(-1)  # Last group is the content
                break
        
        if matched_type:
            if current_type and current_type != matched_type:
                # Different list type, save current list
                if current_list:
                    _add_list_section(sections, current_list, current_type)
                    current_list = []
            
            current_type = matched_type
            current_list.append(matched_content)
        
        elif line_stripped:  # Non-list content
            # Save current list if exists
            if current_list:
                _add_list_section(sections, current_list, current_type)
                current_list = []
                current_type = None
            
            # Add as regular content
            sections.append(TextSection(
                section_id=str(uuid.uuid4()),
                format_type="text",
                content=line_stripped,
                metadata={
                    "section_type": "content",
                    "word_count": len(line_stripped.split())
                }
            ))
    
    # Add final list
    if current_list:
        _add_list_section(sections, current_list, current_type)
    
    return sections

def _add_list_section(sections: List[TextSection], items: List[str], list_type: str):
    """Add a list section to the sections list"""
    if not items:
        return
        
    content = '\n'.join(f"• {item}" for item in items)
    
    sections.append(TextSection(
        section_id=str(uuid.uuid4()),
        format_type="text",
        content=content,
        metadata={
            "section_type": "list",
            "list_type": list_type,
            "item_count": len(items),
            "word_count": sum(len(item.split()) for item in items)
        }
    ))

def _parse_paragraphs(text: str) -> List[TextSection]:
    """Parse text into paragraph sections"""
    # Normalize line endings to handle Windows CRLF before splitting
    normalized_text = text.replace('\r\n', '\n')
    
    # Split by double newlines (paragraph breaks)
    paragraphs = [p.strip() for p in normalized_text.split("\n\n") if p.strip()]
    
    if not paragraphs:
        # Fallback: split by single newlines
        paragraphs = [p.strip() for p in normalized_text.split("\n") if p.strip()]
    
    sections = []
    
    for para in paragraphs:
        if not para:  # Only skip completely empty paragraphs
            continue
        sections.append(
            TextSection(
                section_id=str(uuid.uuid4()),
                format_type="text",
                content=para,
                metadata={
                    "section_type": "paragraph",
                    "word_count": len(para.split()),
                    "char_count": len(para)
                }
            )
        )

    return sections

def _create_document_summary(text: str, sections: List[TextSection]) -> TextSection:
    """Create a summary section for the document"""
    word_count = len(text.split())
    char_count = len(text)
    line_count = len(text.split('\n'))
    
    section_types = {}
    for section in sections:
        section_type = section.metadata.get('section_type', 'unknown')
        section_types[section_type] = section_types.get(section_type, 0) + 1
    
    summary_lines = [
        f"Document contains {len(sections)} sections",
        f"Total: {word_count} words, {char_count} characters, {line_count} lines"
    ]
    
    if section_types:
        type_summary = ", ".join(f"{count} {stype}" for stype, count in section_types.items())
        summary_lines.append(f"Sections: {type_summary}")
    
    return TextSection(
        section_id=str(uuid.uuid4()),
        format_type="text",
        content="\n".join(summary_lines),
        metadata={
            "section_type": "summary",
            "total_sections": len(sections),
            "total_words": word_count,
            "total_chars": char_count,
            "section_types": section_types
        }
    )
