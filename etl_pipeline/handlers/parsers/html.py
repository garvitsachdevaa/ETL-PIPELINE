import uuid
import logging
from bs4 import BeautifulSoup
from handlers.text_schema import TextSection

logger = logging.getLogger(__name__)

# HTML elements to extract as separate sections
HEADER_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6"]
CONTENT_TAGS = ["p", "div", "section", "article", "blockquote"]
LIST_TAGS = ["ul", "ol", "li"]
TABLE_TAGS = ["table", "tr", "td", "th"]

def parse_html(raw_text: str):
    """Parse HTML content with structured extraction"""
    try:
        soup = BeautifulSoup(raw_text, "lxml")

        # Remove script, style, and other non-content elements
        for tag in soup(["script", "style", "noscript", "meta", "link"]):
            tag.decompose()

        sections = []

        # Extract title if present
        title_tag = soup.find('title')
        if title_tag and title_tag.get_text(strip=True):
            sections.append(TextSection(
                section_id=str(uuid.uuid4()),
                format_type="html",
                content=title_tag.get_text(strip=True),
                metadata={
                    "tag": "title",
                    "section_type": "title"
                }
            ))

        # Extract headers with hierarchy
        for header_tag in soup.find_all(HEADER_TAGS):
            text = header_tag.get_text(strip=True)
            if text:
                sections.append(TextSection(
                    section_id=str(uuid.uuid4()),
                    format_type="html",
                    content=text,
                    metadata={
                        "tag": header_tag.name,
                        "section_type": "header",
                        "hierarchy_level": int(header_tag.name[1])  # Extract number from h1, h2, etc.
                    }
                ))

        # Extract content paragraphs
        for content_tag in soup.find_all(CONTENT_TAGS):
            text = content_tag.get_text(strip=True)
            if text and len(text) > 10:  # Ignore very short content
                sections.append(TextSection(
                    section_id=str(uuid.uuid4()),
                    format_type="html",
                    content=text,
                    metadata={
                        "tag": content_tag.name,
                        "section_type": "content",
                        "word_count": len(text.split())
                    }
                ))

        # Extract lists
        for list_tag in soup.find_all(['ul', 'ol']):
            list_items = list_tag.find_all('li')
            if list_items:
                list_text = '\n'.join(f"• {item.get_text(strip=True)}" for item in list_items if item.get_text(strip=True))
                if list_text:
                    sections.append(TextSection(
                        section_id=str(uuid.uuid4()),
                        format_type="html",
                        content=list_text,
                        metadata={
                            "tag": list_tag.name,
                            "section_type": "list",
                            "item_count": len(list_items),
                            "list_type": "ordered" if list_tag.name == "ol" else "unordered"
                        }
                    ))

        # Extract tables
        for table in soup.find_all('table'):
            table_text = _extract_table_text(table)
            if table_text:
                sections.append(TextSection(
                    section_id=str(uuid.uuid4()),
                    format_type="html",
                    content=table_text,
                    metadata={
                        "tag": "table",
                        "section_type": "table",
                        "has_headers": bool(table.find('th'))
                    }
                ))

        # Fallback: if no structured content found, extract all text
        if not sections:
            body_text = soup.get_text(separator="\n", strip=True)
            if body_text:
                sections.append(TextSection(
                    section_id=str(uuid.uuid4()),
                    format_type="html",
                    content=body_text,
                    metadata={
                        "section_type": "fallback",
                        "extraction_method": "get_text",
                        "word_count": len(body_text.split())
                    }
                ))

        return sections

    except Exception as e:
        logger.error(f"Failed to parse HTML: {e}")
        # Fallback to plain text extraction
        try:
            soup = BeautifulSoup(raw_text, "html.parser")
            plain_text = soup.get_text(strip=True)
            return [TextSection(
                section_id=str(uuid.uuid4()),
                format_type="html",
                content=plain_text[:1000] + ("..." if len(plain_text) > 1000 else ""),
                metadata={
                    "error": "parse_failed",
                    "error_message": str(e),
                    "fallback": True
                }
            )]
        except Exception:
            return [TextSection(
                section_id=str(uuid.uuid4()),
                format_type="html",
                content="Failed to parse HTML content",
                metadata={"error": "complete_failure"}
            )]

def _extract_table_text(table):
    """Extract text from HTML table with structure preservation"""
    try:
        rows = []
        
        # Extract headers
        headers = []
        header_row = table.find('tr')
        if header_row:
            header_cells = header_row.find_all(['th', 'td'])
            headers = [cell.get_text(strip=True) for cell in header_cells]
            if headers and any(headers):
                rows.append(" | ".join(headers))
                rows.append("-" * 50)  # Separator
        
        # Extract data rows
        for row in table.find_all('tr')[1:]:  # Skip first row if it was headers
            cells = row.find_all(['td', 'th'])
            cell_texts = [cell.get_text(strip=True) for cell in cells]
            if any(cell_texts):  # Only add non-empty rows
                rows.append(" | ".join(cell_texts))
        
        return "\n".join(rows) if rows else None
        
    except Exception:
        return None
