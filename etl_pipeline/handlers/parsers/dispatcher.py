from handlers.parsers.plain import parse_plain
from handlers.parsers.html import parse_html
from handlers.parsers.csv import parse_csv
from handlers.parsers.json import parse_json
from handlers.parsers.markdown import parse_markdown

def parse_text(raw_text: str, mime_type: str):
    if mime_type == "text/html":
        return parse_html(raw_text)

    if mime_type == "text/csv":
        return parse_csv(raw_text)

    if mime_type == "application/json":
        return parse_json(raw_text)

    if mime_type == "text/markdown":
        return parse_markdown(raw_text)

    return parse_plain(raw_text)
