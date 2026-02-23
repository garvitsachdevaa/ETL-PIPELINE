#!/usr/bin/env python3
"""
ETL Pipeline - Main Entry Point

A comprehensive ETL pipeline for processing various document formats
including text, binary (XLSX, DOCX, PDF), and mixed content.

Usage:
    python main.py --help
    python main.py --batch <input_directory> <output_directory>
    python main.py --ui                    # Launch Streamlit interface

Author: ETL Pipeline Team
Version: 2.0.0
"""

import argparse
import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def launch_streamlit():
    """Launch the Streamlit UI"""
    import subprocess
    import os
    
    ui_path = Path(__file__).parent / "ui" / "streamlit_app_unified.py"
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(ui_path)
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to launch Streamlit: {e}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error("Streamlit not found. Install with: pip install streamlit")
        sys.exit(1)

def process_single_file(input_file: Path, output_file: Path = None):
    """Process a single file and generate output"""
    try:
        from ingestion.loader import ingest
        from handlers.text_handler import handle_text
        from handlers.binary_handler import handle_binary
        from handlers.mixed_handler import handle_mixed
        import json
        
        if not input_file.exists():
            logger.error(f"Input file does not exist: {input_file}")
            sys.exit(1)
        
        logger.info(f"Processing single file: {input_file}")
        
        # Step 1: Ingest the file
        with open(input_file, 'rb') as f:
            # Create a mock file object for ingest function
            class MockFile:
                def __init__(self, name, content):
                    self.name = name
                    self._content = content
                def read(self):
                    return self._content
            
            mock_file = MockFile(input_file.name, f.read())
            doc = ingest(file=mock_file)
        
        logger.info(f"Document ingested: format={doc.detected_format}, mime={doc.mime_type}, target={doc.routing_target}")
        
        # Step 2: Route to appropriate handler
        if doc.routing_target == "text_handler":
            result = handle_text(doc)
        elif doc.routing_target == "binary_handler":
            result = handle_binary(doc)
        elif doc.routing_target == "mixed_handler":
            result = handle_mixed(doc)
        else:
            logger.error(f"Unknown routing target: {doc.routing_target}")
            sys.exit(1)
        
        # Step 3: Generate output
        output_data = {
            "document_id": result.document_id,
            "processing_metadata": {
                "source_file": str(input_file),
                "detected_format": doc.detected_format,
                "mime_type": doc.mime_type,
                "routing_target": doc.routing_target,
                "encoding": doc.encoding
            },
            "result": result.to_dict() if hasattr(result, 'to_dict') else str(result)
        }
        
        # Step 4: Save or display output
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Output saved to: {output_file}")
        else:
            print("\n" + "="*50)
            print("PROCESSING RESULT")
            print("="*50)
            print(json.dumps(output_data, indent=2, ensure_ascii=False))
            print("="*50)
        
        return output_data
        
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.error("Install requirements with: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)

def run_batch_processing(input_dir: Path, output_dir: Path):
    """Run batch processing on a directory"""
    try:
        from ingestion.batch_processor import BatchProcessor
        
        if not input_dir.exists():
            logger.error(f"Input directory does not exist: {input_dir}")
            sys.exit(1)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processor = BatchProcessor()
        logger.info(f"Processing files from {input_dir} to {output_dir}")
        
        results = processor.process_directory(input_dir, output_dir)
        
        logger.info(f"Processing complete. Processed {len(results)} files.")
        return results
        
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.error("Install requirements with: pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="ETL Pipeline - Document Processing System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --ui                                    # Launch unified web interface (recommended)
  %(prog)s --file document.txt                     # Process single file (output to console)
  %(prog)s --file document.pdf result.json        # Process file with output
  %(prog)s --batch input/ output/                  # Process directory
  %(prog)s --batch documents/ processed/           # Batch process documents

Web Interface Features:
  - Auto-detects single file vs batch processing
  - Interactive upload and text input
  - Real-time progress tracking
  - Export options (JSON, text)
  - Error handling and debugging
        """
    )
    
    parser.add_argument(
        "--ui", 
        action="store_true", 
        help="Launch Streamlit web interface"
    )
    
    parser.add_argument(
        "--file", 
        nargs="+", 
        metavar=("INPUT_FILE", "[OUTPUT_FILE]"),
        help="Process a single file (optional output file path)"
    )
    
    parser.add_argument(
        "--batch", 
        nargs=2, 
        metavar=("INPUT_DIR", "OUTPUT_DIR"),
        help="Run batch processing on input directory"
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version="ETL Pipeline 2.0.0"
    )
    
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.ui:
        logger.info("Launching Streamlit interface...")
        launch_streamlit()
    
    elif args.file:
        input_file = Path(args.file[0])
        output_file = Path(args.file[1]) if len(args.file) > 1 else None
        process_single_file(input_file, output_file)
    
    elif args.batch:
        input_dir = Path(args.batch[0])
        output_dir = Path(args.batch[1])
        run_batch_processing(input_dir, output_dir)
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
