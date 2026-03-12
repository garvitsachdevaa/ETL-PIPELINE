"""
conftest.py — ensures etl_pipeline/ root is on sys.path so that
`import handlers`, `import ingestion`, etc. resolve correctly when
pytest is run from the etl_pipeline/ directory.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
