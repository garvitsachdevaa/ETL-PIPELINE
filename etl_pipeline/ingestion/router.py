def route(detected_format):
    if detected_format == "text":
        return "text_handler"

    if detected_format == "structured":
        return "text_handler"

    if detected_format in ["image", "document"]:
        return "binary_handler"
    
    if detected_format == "unknown":
        return "unknown_handler"  # Will trigger explicit error in batch_processor

    return "mixed_handler"
