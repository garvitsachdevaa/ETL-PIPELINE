import uuid
import csv
from io import StringIO
import logging
from handlers.text_schema import TextSection

logger = logging.getLogger(__name__)

def parse_csv(raw_text: str):
    """Parse CSV content with robust error handling and multiple delimiter detection"""
    try:
        # Try to detect dialect
        sample_size = min(1024, len(raw_text))
        sample = raw_text[:sample_size]
        
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=',;\t|')
        except csv.Error:
            # Fallback to comma delimiter
            dialect = 'excel'
        
        reader = csv.reader(StringIO(raw_text), dialect=dialect)
        rows = list(reader)

        if not rows:
            return [TextSection(
                section_id=str(uuid.uuid4()),
                format_type="csv",
                content="Empty CSV file",
                metadata={"error": "no_data", "rows": 0}
            )]

        # Filter out completely empty rows
        non_empty_rows = [row for row in rows if any(cell.strip() for cell in row)]
        
        if not non_empty_rows:
            return [TextSection(
                section_id=str(uuid.uuid4()),
                format_type="csv",
                content="CSV contains no data",
                metadata={"error": "empty_rows", "total_rows": len(rows)}
            )]

        header = non_empty_rows[0]
        data_rows = non_empty_rows[1:] if len(non_empty_rows) > 1 else []

        sections = []

        # Add header section
        if header:
            sections.append(TextSection(
                section_id=str(uuid.uuid4()),
                format_type="csv",
                content=f"CSV Headers: {' | '.join(header)}",
                metadata={
                    "section_type": "header",
                    "column_count": len(header),
                    "headers": header
                }
            ))

        # Add data sections
        for idx, row in enumerate(data_rows):
            # Pad row to match header length
            padded_row = row + [''] * (len(header) - len(row))
            
            row_dict = {
                header[i]: padded_row[i] if i < len(padded_row) else ""
                for i in range(len(header))
            }

            content = " | ".join(
                f"{key}: {value}" for key, value in row_dict.items() if value.strip()
            )

            sections.append(
                TextSection(
                    section_id=str(uuid.uuid4()),
                    format_type="csv",
                    content=content,
                    metadata={
                        "section_type": "data_row",
                        "row_index": idx,
                        "row_data": row_dict,
                        "column_count": len([v for v in row_dict.values() if v.strip()])
                    }
                )
            )

        # Add summary section
        if sections:
            sections.append(TextSection(
                section_id=str(uuid.uuid4()),
                format_type="csv",
                content=f"CSV Summary: {len(data_rows)} data rows with {len(header)} columns",
                metadata={
                    "section_type": "summary",
                    "total_data_rows": len(data_rows),
                    "total_columns": len(header),
                    "dialect": str(dialect)
                }
            ))

        return sections

    except Exception as e:
        logger.error(f"Failed to parse CSV: {e}")
        return [TextSection(
            section_id=str(uuid.uuid4()),
            format_type="csv",
            content=raw_text[:500] + ("..." if len(raw_text) > 500 else ""),
            metadata={
                "error": "parse_failed",
                "error_message": str(e),
                "fallback": True
            }
        )]
