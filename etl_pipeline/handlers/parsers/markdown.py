import uuid
import re
from typing import List
from handlers.text_schema import TextSection

def parse_markdown(raw_text: str) -> List[TextSection]:
    """Parse markdown with proper structure preservation"""
    if not raw_text or not raw_text.strip():
        return [TextSection(
            section_id=str(uuid.uuid4()),
            format_type="markdown",
            content="Empty document",
            metadata={"empty": True}
        )]

    # Normalize line endings
    normalized_text = raw_text.replace('\r\n', '\n')
    sections = []
    
    # Split content by headers and other markdown structures
    parsed_sections = _parse_markdown_structure(normalized_text)
    
    # Add document summary
    if len(parsed_sections) > 1:
        summary = _create_markdown_summary(normalized_text, parsed_sections)
        sections.append(summary)
    
    sections.extend(parsed_sections)
    return sections

def _parse_markdown_structure(text: str) -> List[TextSection]:
    """Parse markdown into structured sections"""
    lines = text.split('\n')
    sections = []
    current_section = []
    current_header = None
    current_level = 0
    in_code_block = False
    code_lang = None
    
    i = 0
    while i < len(lines):
        line = lines[i]
        
        # Handle code blocks
        if line.strip().startswith('```'):
            if not in_code_block:
                # Start of code block
                in_code_block = True
                code_lang = line.strip()[3:].strip() or 'text'
                if current_section:
                    sections.append(_create_section(current_section, current_header, current_level))
                    current_section = []
                current_section.append(line)
            else:
                # End of code block
                current_section.append(line)
                sections.append(_create_code_section(current_section, code_lang))
                current_section = []
                in_code_block = False
                code_lang = None
            i += 1
            continue
            
        if in_code_block:
            current_section.append(line)
            i += 1
            continue
            
        # Handle headers
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        if header_match:
            # Save previous section
            if current_section:
                sections.append(_create_section(current_section, current_header, current_level))
                current_section = []
                
            # Start new section
            level = len(header_match.group(1))
            header_text = header_match.group(2)
            current_header = header_text
            current_level = level
            current_section.append(line)
            i += 1
            continue
            
        # Handle tables
        if _is_table_row(line):
            if current_section and not any(_is_table_row(l) for l in current_section[-3:]):
                # Start of table, save previous section
                if current_section:
                    sections.append(_create_section(current_section, current_header, current_level))
                    current_section = []
                current_header = None
                current_level = 0
            current_section.append(line)
            i += 1
            continue
            
        # Handle horizontal rules
        if re.match(r'^-{3,}$|^\*{3,}$|^_{3,}$', line.strip()):
            if current_section:
                sections.append(_create_section(current_section, current_header, current_level))
                current_section = []
            sections.append(_create_rule_section(line))
            i += 1
            continue
            
        # Regular content
        current_section.append(line)
        i += 1
    
    # Handle final section
    if current_section:
        sections.append(_create_section(current_section, current_header, current_level))
    
    return [s for s in sections if s]

def _create_section(lines: List[str], header: str = None, level: int = 0) -> TextSection:
    """Create a text section from lines"""
    content = '\n'.join(lines).strip()
    if not content:
        return None
        
    word_count = len(content.split())
    
    if header:
        section_type = f"header_h{level}"
        title = header
    elif _is_table_content(content):
        section_type = "table"
        title = "Table"
    elif content.strip().startswith(('- ', '* ', '+ ')) or re.match(r'^\d+\.\s', content.strip()):
        section_type = "list"
        title = "List"
    else:
        section_type = "paragraph"
        title = "Content"
    
    metadata = {
        "section_type": section_type,
        "word_count": word_count,
        "char_count": len(content),
        "line_count": len([l for l in lines if l.strip()])
    }
    
    if header:
        metadata["header_text"] = header
        metadata["header_level"] = level
        
    if section_type == "table":
        metadata.update(_parse_table_metadata(content))
        
    return TextSection(
        section_id=str(uuid.uuid4()),
        format_type="markdown",
        content=content,
        metadata=metadata
    )

def _create_code_section(lines: List[str], language: str) -> TextSection:
    """Create a code block section"""
    content = '\n'.join(lines).strip()
    
    return TextSection(
        section_id=str(uuid.uuid4()),
        format_type="markdown",
        content=content,
        metadata={
            "section_type": "code_block",
            "language": language,
            "line_count": len(lines),
            "char_count": len(content)
        }
    )

def _create_rule_section(line: str) -> TextSection:
    """Create a horizontal rule section"""
    return TextSection(
        section_id=str(uuid.uuid4()),
        format_type="markdown", 
        content=line.strip(),
        metadata={
            "section_type": "horizontal_rule",
            "char_count": len(line.strip())
        }
    )

def _is_table_row(line: str) -> bool:
    """Check if line is part of a markdown table"""
    stripped = line.strip()
    return bool(stripped and '|' in stripped and stripped.count('|') >= 2)

def _is_table_content(content: str) -> bool:
    """Check if content contains table structure"""
    lines = [l.strip() for l in content.split('\n') if l.strip()]
    return len(lines) >= 2 and all(_is_table_row(l) for l in lines)

def _parse_table_metadata(content: str) -> dict:
    """Extract table structure metadata"""
    lines = [l.strip() for l in content.split('\n') if l.strip()]
    if len(lines) < 2:
        return {}
        
    header_row = lines[0]
    headers = [h.strip() for h in header_row.split('|')[1:-1]]  # Remove empty first/last
    
    data_rows = []
    for line in lines[2:]:  # Skip header and separator
        if _is_table_row(line):
            row_data = [cell.strip() for cell in line.split('|')[1:-1]]
            data_rows.append(row_data)
    
    return {
        "table_headers": headers,
        "table_rows": len(data_rows),
        "table_columns": len(headers),
        "table_data": data_rows
    }

def _create_markdown_summary(text: str, sections: List[TextSection]) -> TextSection:
    """Create document summary section"""
    lines = text.split('\n')
    word_count = len(text.split())
    
    # Count different section types
    section_counts = {}
    for section in sections:
        stype = section.metadata.get('section_type', 'unknown')
        section_counts[stype] = section_counts.get(stype, 0) + 1
    
    # Count headers by level
    header_counts = {}
    for section in sections:
        if section.metadata.get('section_type', '').startswith('header_'):
            level = section.metadata.get('header_level', 0)
            header_counts[f'h{level}'] = header_counts.get(f'h{level}', 0) + 1
    
    summary_parts = [
        f"Markdown document contains {len(sections)} sections",
        f"Total: {word_count} words, {len(text)} characters, {len(lines)} lines"
    ]
    
    if section_counts:
        type_summary = ", ".join([f"{count} {stype}" for stype, count in section_counts.items()])
        summary_parts.append(f"Sections: {type_summary}")
        
    if header_counts:
        header_summary = ", ".join([f"{count} {level}" for level, count in header_counts.items()])
        summary_parts.append(f"Headers: {header_summary}")
    
    summary_content = "\n".join(summary_parts)
    
    return TextSection(
        section_id=str(uuid.uuid4()),
        format_type="markdown",
        content=summary_content,
        metadata={
            "section_type": "summary",
            "total_sections": len(sections),
            "total_words": word_count,
            "total_chars": len(text),
            "total_lines": len(lines),
            "section_types": section_counts,
            "header_levels": header_counts
        }
    )