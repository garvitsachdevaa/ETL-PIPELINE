import time

def extract_metadata(source_name, raw_bytes):
    return {
        "source_name": source_name,
        "file_size": len(raw_bytes) if raw_bytes else None,
        "ingested_at": time.time()
    }
