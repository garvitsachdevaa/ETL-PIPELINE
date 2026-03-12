# ETL Pipeline Documentation Test

## Overview

This **Markdown document** tests the ETL pipeline's ability to process structured text content with various formatting elements.

## Key Features

### Text Formatting Support
- **Bold text** processing
- *Italic text* handling  
- `Code snippets` extraction
- ~~Strikethrough~~ text recognition

### List Processing
1. Ordered list item one
2. Ordered list item two
   - Nested unordered item
   - Another nested item
3. Ordered list item three

### Code Blocks

```python
# Python code example
def process_document(content, format_type):
    """Process document content based on format type"""
    if format_type == "markdown":
        return extract_markdown_structure(content)
    return default_processing(content)
```

```json
{
  "test_case": "markdown_processing",
  "expected_sections": 8,
  "features_tested": [
    "heading_extraction",
    "list_processing", 
    "code_block_handling",
    "table_parsing"
  ]
}
```

### Table Processing

| Component | Status | Performance | Notes |
|-----------|--------|-------------|--------|
| Markdown Parser | Active | Excellent | Full CommonMark support |
| Code Highlighting | Enhanced | Very Good | Multiple language support |
| Table Extraction | Optimized | Good | Complex table handling |

### Links and References

- [External Link](https://example.com)
- [Internal Reference](#overview)
- ![Image Reference](test_image.png "Test Image")

### Blockquotes

> This blockquote contains important information about the ETL pipeline's
> markdown processing capabilities. It should be extracted as a distinct
> content section with appropriate metadata.

### Special Characters and Encoding

Testing special characters: áéíóú, ñ, ü, ç, €, £, ¥, ©, ®, ™

Mathematical expressions: α + β = γ, ∑(x²), √(a² + b²)

## Processing Validation

This markdown document tests:

1. **Structure preservation** - Maintaining heading hierarchy
2. **Content extraction** - Extracting text while removing formatting markup
3. **Metadata generation** - Creating appropriate section metadata
4. **Link handling** - Processing internal and external references
5. **Code processing** - Handling inline code and code blocks
6. **Table parsing** - Extracting tabular data structure

---

**Expected Output**: Multiple structured sections with preserved hierarchy and rich metadata indicating content types (heading, paragraph, list, code, table, etc.).