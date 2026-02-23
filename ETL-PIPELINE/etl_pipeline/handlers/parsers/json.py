import uuid
import json
import logging
from typing import Any, List, Tuple
from handlers.text_schema import TextSection

logger = logging.getLogger(__name__)

def _flatten(obj: Any, parent_key: str = "", sep: str = ".") -> List[Tuple[str, Any]]:
    """Flatten nested JSON structure with path tracking"""
    items = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(_flatten(v, new_key, sep))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{parent_key}[{i}]"
            items.extend(_flatten(v, new_key, sep))
    else:
        items.append((parent_key, obj))

    return items

def parse_json(raw_text: str):
    """Parse JSON content with structured extraction and error handling"""
    try:
        data = json.loads(raw_text)
        
        sections = []
        
        # Add summary section
        summary = _get_json_summary(data)
        sections.append(TextSection(
            section_id=str(uuid.uuid4()),
            format_type="json",
            content=f"JSON Structure: {summary}",
            metadata={
                "section_type": "summary",
                "json_type": type(data).__name__,
                **_get_structure_stats(data)
            }
        ))

        # Flatten and create sections for each value
        flattened = _flatten(data)

        # Group similar paths for better organization
        grouped_items = _group_json_items(flattened)
        
        for group_name, items in grouped_items.items():
            if len(items) == 1:
                # Single item
                key, value = items[0]
                content = f"{key}: {_format_value(value)}"
            else:
                # Multiple items in group
                content_lines = [f"{key}: {_format_value(value)}" for key, value in items[:10]]  # Limit to first 10
                if len(items) > 10:
                    content_lines.append(f"... and {len(items) - 10} more items")
                content = "\n".join(content_lines)
            
            sections.append(TextSection(
                section_id=str(uuid.uuid4()),
                format_type="json",
                content=content,
                metadata={
                    "section_type": "json_group",
                    "group_name": group_name,
                    "item_count": len(items),
                    "sample_paths": [item[0] for item in items[:3]]  # First 3 paths as sample
                }
            ))

        return sections

    except json.JSONDecodeError as e:
        logger.warning(f"Invalid JSON content: {e}")
        return [TextSection(
            section_id=str(uuid.uuid4()),
            format_type="json",
            content=raw_text[:500] + ("..." if len(raw_text) > 500 else ""),
            metadata={
                "error": "invalid_json",
                "error_message": str(e),
                "error_line": getattr(e, 'lineno', None),
                "fallback": True
            }
        )]
    
    except Exception as e:
        logger.error(f"Failed to parse JSON: {e}")
        return [TextSection(
            section_id=str(uuid.uuid4()),
            format_type="json",
            content=raw_text[:500] + ("..." if len(raw_text) > 500 else ""),
            metadata={
                "error": "parse_failed",
                "error_message": str(e),
                "fallback": True
            }
        )]

def _get_json_summary(data: Any) -> str:
    """Generate a human-readable summary of JSON structure"""
    if isinstance(data, dict):
        key_count = len(data.keys())
        sample_keys = list(data.keys())[:3]
        keys_preview = ", ".join(sample_keys)
        if key_count > 3:
            keys_preview += f", ... ({key_count - 3} more)"
        return f"Object with {key_count} keys ({keys_preview})"
    
    elif isinstance(data, list):
        length = len(data)
        if length == 0:
            return "Empty array"
        item_types = set(type(item).__name__ for item in data[:5])  # Check first 5 items
        type_summary = ", ".join(item_types)
        return f"Array with {length} items (types: {type_summary})"
    
    else:
        return f"Single {type(data).__name__} value"

def _get_structure_stats(data: Any) -> dict:
    """Get statistics about JSON structure"""
    stats = {
        "total_keys": 0,
        "total_arrays": 0,
        "total_objects": 0,
        "max_depth": 0
    }
    
    def count_recursive(obj, depth=0):
        stats["max_depth"] = max(stats["max_depth"], depth)
        
        if isinstance(obj, dict):
            stats["total_objects"] += 1
            stats["total_keys"] += len(obj)
            for value in obj.values():
                count_recursive(value, depth + 1)
        elif isinstance(obj, list):
            stats["total_arrays"] += 1
            for item in obj:
                count_recursive(item, depth + 1)
    
    count_recursive(data)
    return stats

def _group_json_items(flattened: List[Tuple[str, Any]]) -> dict:
    """Group flattened JSON items by common path patterns"""
    groups = {}
    
    for key, value in flattened:
        # Extract base path (remove array indices for grouping)
        base_path = _extract_base_path(key)
        
        if base_path not in groups:
            groups[base_path] = []
        groups[base_path].append((key, value))
    
    return groups

def _extract_base_path(path: str) -> str:
    """Extract base path by removing array indices"""
    import re
    # Replace [number] with [*] to group array items
    return re.sub(r'\\[\\d+\\]', '[*]', path)

def _format_value(value: Any) -> str:
    """Format a JSON value for display"""
    if isinstance(value, str):
        if len(value) > 100:
            return f'"{value[:97]}..."'
        return f'"{value}"'
    elif isinstance(value, (dict, list)):
        return f"{type(value).__name__} ({len(value)} items)"
    elif value is None:
        return "null"
    else:
        return str(value)

    if not sections:
        sections.append(
            TextSection(
                section_id=str(uuid.uuid4()),
                format_type="json",
                content=json.dumps(data, indent=2),
                metadata={"raw": True}
            )
        )

    return sections
