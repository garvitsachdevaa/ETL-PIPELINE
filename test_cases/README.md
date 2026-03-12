# ETL Pipeline Test Cases

This directory contains comprehensive test files for validating the ETL pipeline's document processing capabilities across various formats and content types.

## Test Files Overview

### 📄 Text Format Tests

#### `sample_text.txt`
- **Purpose**: Plain text processing validation
- **Content**: Multi-paragraph document with various text structures
- **Tests**: Text normalization, section detection, content extraction

#### `employee_data.csv`
- **Purpose**: CSV parsing and structured data handling
- **Content**: Employee records with headers, special characters, and mixed data types
- **Tests**: Delimiter detection, data type inference, structured output generation

#### `inventory.csv`  
- **Purpose**: Complex CSV processing with quoted fields
- **Content**: Product inventory with embedded commas and special characters
- **Tests**: Quoted field handling, price formatting, supplier data with LLC notation

#### `project_config.json`
- **Purpose**: JSON structure parsing and validation
- **Content**: Nested objects, arrays, and complex data structures
- **Tests**: JSON validation, nested object extraction, array processing

### 🌐 Web Content Tests

#### `sample_html.html`
- **Purpose**: HTML content extraction and structure preservation
- **Content**: Complete HTML document with styles, tables, lists, and embedded content
- **Tests**: Tag removal, structure preservation, table extraction, link handling

#### `sample_xml.xml`
- **Purpose**: XML processing with namespace handling
- **Content**: Multi-namespace XML with nested elements, attributes, and CDATA sections
- **Tests**: Namespace resolution, attribute extraction, nested element processing

### 📝 Rich Content Tests

#### `documentation.md`
- **Purpose**: Markdown processing and formatting recognition
- **Content**: Full Markdown features including headers, lists, code blocks, tables
- **Tests**: Markdown parsing, structure preservation, code block extraction

#### `mixed_content_sample.txt`
- **Purpose**: Mixed content detection and routing
- **Content**: Text file containing embedded JSON, XML, HTML, and CSV sections
- **Tests**: Content type detection, format routing, mixed content processing

### 📊 Binary Format Tests

#### `test_spreadsheet.xlsx`
- **Purpose**: Excel spreadsheet processing with multiple worksheets
- **Content**: 
  - **Employee_Data**: Personnel records with formulas
  - **Sales_Data**: Quarterly sales with totals and calculations
  - **Analytics**: Cross-sheet references and complex formulas
  - **Formatted_Data**: Various number formats and conditional formatting
- **Tests**: Multi-worksheet support, formula preservation, cell formatting, cross-sheet references

#### `test_document.docx`
- **Purpose**: Word document processing with structure preservation
- **Content**: 
  - Hierarchical headings (H1, H2, H3)
  - Formatted text (bold, italic, underline)
  - Bullet and numbered lists
  - Complex tables with headers
  - Mixed content elements
- **Tests**: Structure preservation, formatting recognition, table extraction, heading hierarchy

## Test Execution

### Using the ETL Pipeline CLI

```bash
# Test individual files
cd etl_pipeline
python main.py --batch ../test_cases/sample_text.txt output/

# Test entire directory
python main.py --batch ../test_cases/ output/

# Verbose output for debugging
python main.py --batch ../test_cases/ output/ --verbose
```

### Using the Web Interface

```bash
python main.py --ui
```

Then upload test files individually through the Streamlit interface.

## Expected Processing Results

### Text Files
- **Sections**: Multiple content regions based on paragraph structure
- **Metadata**: Format type, confidence scores, content statistics
- **Output**: Structured JSON with preserved content organization

### CSV Files
- **Structure**: Tabular data with proper column/row identification
- **Data Types**: Automatic inference for numbers, dates, strings
- **Metadata**: Column headers, row counts, data type information

### HTML/XML Files
- **Content**: Clean text extraction with tag removal
- **Structure**: Preserved heading hierarchy and list organization
- **Tables**: Extracted as structured data with proper headers

### Binary Files (XLSX/DOCX)
- **XLSX**: Multiple worksheet processing with formula preservation
- **DOCX**: Hierarchical content with formatting metadata
- **Structure**: Rich metadata including element types, formatting, and statistics

## Validation Checklist

For each test file, verify:

- ✅ **Content Extraction**: All text content properly extracted
- ✅ **Structure Preservation**: Document hierarchy maintained  
- ✅ **Metadata Generation**: Rich metadata with confidence scores
- ✅ **Format Detection**: Correct MIME type and format identification
- ✅ **Error Handling**: Graceful handling of malformed content
- ✅ **Performance**: Processing completed within acceptable time limits

## Test Coverage

| Format Type | File Count | Features Tested |
|-------------|------------|-----------------|
| Plain Text | 2 | Multi-paragraph, encoding, normalization |
| CSV | 2 | Headers, special chars, quoted fields |
| JSON | 1 | Nested objects, arrays, validation |
| HTML | 1 | Tags, tables, structure, links |
| XML | 1 | Namespaces, attributes, CDATA |
| Markdown | 1 | Headers, lists, code, tables |
| Mixed Content | 1 | Format detection, routing |
| XLSX | 1 | Multi-sheet, formulas, formatting |
| DOCX | 1 | Structure, tables, formatting |

## Adding New Test Cases

To add new test cases:

1. **Create test file** in appropriate format
2. **Document purpose** and expected results
3. **Update this README** with test details
4. **Validate processing** with ETL pipeline
5. **Add to version control**

## Notes

- Test files contain realistic data patterns and edge cases
- Binary files include complex formatting and cross-references
- Mixed content tests validate format detection accuracy
- All test cases designed for automated validation and CI/CD integration