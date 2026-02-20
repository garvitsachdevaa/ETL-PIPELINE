---
title: ETL Pipeline
emoji: 🔄
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.54.0"
app_file: etl_pipeline/ui/streamlit_app_unified.py
pinned: false
license: apache-2.0
suggested_hardware: a10g-small
models:
  - google/paligemma2-3b-pt-224
  - datalab-to/chandra
---

# ETL Pipeline

A comprehensive Extract, Transform, Load (ETL) pipeline for processing various document formats including text, binary documents (XLSX, DOCX, PDF), and mixed content with advanced format detection and structure preservation.

## Features

### 🔍 Smart Format Detection
- Automatic MIME type detection
- Pure vs. mixed content classification  
- Support for text, binary, and mixed document types

### 📄 Document Processing
- **Text Documents**: Plain text, CSV, JSON, HTML, XML
- **Binary Documents**: 
  - Excel spreadsheets (XLSX) with multi-worksheet support
  - Word documents (DOCX) with structure preservation
  - PDF documents via OCR integration
  - Images (JPEG, PNG, TIFF, BMP, GIF)

### ⚡ Advanced Capabilities
- Structure-preserving document extraction
- Hierarchical content organization
- Rich metadata extraction
- Confidence scoring and validation
- Batch processing with parallel execution
- Web-based user interface

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ETL-PIPELINE
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS/Linux  
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   cd etl_pipeline
   pip install -r requirements.txt
   ```

## Usage

### Command Line Interface

```bash
# Launch web interface
python main.py --ui

# Batch process documents
python main.py --batch input_directory/ output_directory/

# Show help
python main.py --help

# Enable verbose logging
python main.py --batch input/ output/ --verbose
```

### Web Interface

Launch the Streamlit web interface for interactive document processing:

```bash
python main.py --ui
```

The web interface provides:
- File upload and processing
- Real-time processing status
- Results visualization
- Batch processing capabilities

### Programmatic Usage

```python
from ingestion.loader import DocumentLoader
from ingestion.router import DocumentRouter

# Load and process a document
loader = DocumentLoader()
doc = loader.load_file("document.pdf")

router = DocumentRouter()
results = router.route_document(doc)

for region in results:
    print(f"Content: {region.content}")
    print(f"Metadata: {region.metadata}")
```

## Architecture

```
etl_pipeline/
├── main.py                 # Main entry point
├── requirements.txt        # Dependencies
├── common/                 # Shared utilities
│   ├── errors.py          # Exception classes
│   └── utils.py           # Utility functions
├── ingestion/             # Data ingestion layer
│   ├── loader.py          # File loading
│   ├── detector.py        # Format detection
│   ├── validator.py       # Content validation
│   ├── router.py          # Document routing
│   ├── batch_processor.py # Batch processing
│   └── schemas.py         # Data schemas
├── handlers/              # Format-specific processors
│   ├── text_handler.py    # Text document processing
│   ├── binary_handler.py  # Binary document routing
│   ├── mixed_handler.py   # Mixed content processing
│   ├── docx_handler.py    # Word document processing
│   ├── xlsx_handler.py    # Excel spreadsheet processing
│   ├── parsers/           # Text format parsers
│   └── ocr/              # OCR integration
└── ui/                    # User interface
    ├── streamlit_app.py   # Single file processing
    └── streamlit_app_batch.py # Batch processing UI
```

## Supported Formats

### Text Documents
- **Plain Text**: `.txt` files with automatic encoding detection
- **CSV**: Structured data with delimiter detection
- **JSON**: Structured data with validation
- **HTML**: Web content with tag preservation
- **XML**: Structured markup with namespace handling

### Binary Documents  
- **Excel**: `.xlsx` files with multi-worksheet support, formulas, and formatting
- **Word**: `.docx` files with structure preservation (headings, lists, tables)
- **PDF**: Document extraction via OCR integration
- **Images**: `.jpg`, `.png`, `.tiff`, `.bmp`, `.gif` via OCR

### Mixed Content
- Documents containing both text and binary elements
- HTML with embedded content
- XML with mixed namespaces

## Output Structure

All processed documents generate structured output with:

```json
{
  "document_id": "unique-identifier",
  "sections": [
    {
      "section_id": "section-1", 
      "content": "extracted text content",
      "format_type": "heading|paragraph|table|list",
      "metadata": {
        "confidence": 0.95,
        "source_format": "docx",
        "extraction_method": "python-docx"
      }
    }
  ],
  "metadata": {
    "total_sections": 5,
    "processing_time": 1.23,
    "source_file": "document.docx"
  }
}
```

## Configuration

### Environment Variables
- `CHANDRA_OCR_ENDPOINT`: OCR service endpoint
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Processing Settings
Processing parameters can be adjusted in the respective handler modules.

## Development

### Project Structure
The codebase follows a modular architecture with clear separation of concerns:

- **Ingestion Layer**: Handles file loading, format detection, and routing
- **Processing Layer**: Format-specific document processing 
- **Output Layer**: Structured data generation and export

### Adding New Format Support

1. Create a new handler in `handlers/`
2. Register the format in `handlers/binary_handler.py` or text routing
3. Add MIME type detection in `ingestion/detector.py`  
4. Update format mappings in routing logic

### Testing

The pipeline includes comprehensive validation:
- Format detection accuracy
- Content extraction quality  
- Structure preservation
- Performance benchmarks

## Dependencies

### Core Dependencies
- **streamlit**: Web interface framework
- **pandas**: Data processing and analysis
- **chardet**: Character encoding detection
- **beautifulsoup4**: HTML/XML parsing
- **lxml**: Fast XML/HTML processing

### Document Processing
- **python-docx**: Word document processing
- **openpyxl**: Excel spreadsheet processing  
- **PyPDF2**: PDF text extraction
- **PyMuPDF**: Advanced PDF processing
- **Pillow**: Image processing

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Support

[Add support contact information here]