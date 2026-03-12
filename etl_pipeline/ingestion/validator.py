def validate_input(detected_format, raw_bytes, raw_text):
    # For text formats, we can have either raw_text OR raw_bytes (which will be decoded later)
    if detected_format == "text" and not raw_text and not raw_bytes:
        raise ValueError("Empty text input")

    if detected_format != "text" and raw_bytes is None:
        raise ValueError("Binary input missing bytes")

    return True
