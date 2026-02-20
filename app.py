"""
HuggingFace Spaces entry point.

NOT used directly — HF Spaces runs the app_file set in README.md frontmatter:
    app_file: etl_pipeline/ui/streamlit_app_unified.py

This file is kept only as a fallback if HF Spaces needs a root app.py.
In that case, launch the unified app via the streamlit CLI.
"""
import sys
import subprocess
from pathlib import Path

if __name__ == "__main__":
    app = Path(__file__).parent / "etl_pipeline" / "ui" / "streamlit_app_unified.py"
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app)]))
