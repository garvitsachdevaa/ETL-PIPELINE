import uuid
from handlers.binary_schema import Page, Region
from handlers.ocr.chandra_adapter import run_chandra_cli

def ocr_pdf(doc):
    ocr_data = run_chandra_cli(doc.raw_bytes, doc.document_id)

    pages = []

    for page_idx, page in enumerate(ocr_data.get("pages", []), start=1):
        regions = []

        for block in page.get("blocks", []):
            regions.append(
                Region(
                    region_id=str(uuid.uuid4()),
                    text=block.get("text", ""),
                    bbox=block.get("bbox", [0, 0, 0, 0]),
                    confidence=block.get("confidence", 1.0),
                    metadata={
                        "engine": "chandra",
                        "page": page_idx
                    }
                )
            )

        pages.append(
            Page(
                page_id=str(uuid.uuid4()),
                page_number=page_idx,
                regions=regions,
                metadata={"pdf": True}
            )
        )

    return pages
