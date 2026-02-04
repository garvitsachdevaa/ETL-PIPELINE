import json
import subprocess
from pathlib import Path
import tempfile
import PyPDF2
import io

def run_chandra_cli(raw_bytes: bytes, document_id: str):
    """OCR adapter with fallback to PyPDF2 for text extraction if Chandra is not available"""
    
    # Try Chandra OCR first
    try:
        return _run_chandra_ocr(raw_bytes, document_id)
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Chandra OCR not available ({e}), falling back to PyPDF2...")
        return _run_pypdf2_fallback(raw_bytes, document_id)

def _run_chandra_ocr(raw_bytes: bytes, document_id: str):
    """Original Chandra OCR implementation"""
    base_dir = Path("data/ocr_outputs") / document_id
    base_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as tmp:
        tmp.write(raw_bytes)
        tmp_path = tmp.name

    try:
        cmd = [
        "chandra",
        tmp_path,
        str(base_dir)
    ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Chandra OCR failed: {e.stderr}"
        )

    finally:
        Path(tmp_path).unlink(missing_ok=True)

    json_path = base_dir / "ocr.json"

    if not json_path.exists():
        raise RuntimeError("Chandra OCR did not produce ocr.json")

    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def _run_pypdf2_fallback(raw_bytes: bytes, document_id: str):
    """Fallback PDF text extraction using PyPDF2"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(raw_bytes))
        pages = []
        
        for page_num, page in enumerate(pdf_reader.pages, 1):
            try:
                text = page.extract_text()
                pages.append({
                    "page_number": page_num,
                    "blocks": [{
                        "text": text,
                        "bbox": [0, 0, 100, 100],  # Placeholder bbox
                        "confidence": 0.9  # Assume good confidence for direct text extraction
                    }]
                })
            except Exception as e:
                print(f"Error extracting page {page_num}: {e}")
                pages.append({
                    "page_number": page_num,
                    "blocks": [{
                        "text": f"[Error extracting page {page_num}: {str(e)}]",
                        "bbox": [0, 0, 100, 100],
                        "confidence": 0.0
                    }]
                })
        
        return {"pages": pages}
        
    except Exception as e:
        raise RuntimeError(f"PDF text extraction failed: {str(e)}")
