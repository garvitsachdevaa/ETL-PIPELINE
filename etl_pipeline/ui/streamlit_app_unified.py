import sys
from pathlib import Path
import os
import time
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# Add project root (etl_pipeline/) to sys.path so all imports work
sys.path.append(str(Path(__file__).resolve().parents[1]))

# ── HuggingFace authentication ──────────────────────────────────────────────
# On HF Spaces the HF_TOKEN secret is injected as an env var automatically.
# This lets transformers download gated models (PaliGemma, Chandra).
_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    try:
        from huggingface_hub import login as _hf_login
        _hf_login(token=_hf_token, add_to_git_credential=False)
    except Exception:
        pass  # huggingface_hub not available or token invalid — model load will fail later

import streamlit as st
import pandas as pd

# Import from our modules
from ingestion.batch_processor import BatchProcessor, _process_single_file_task
from ingestion.loader import ingest

st.set_page_config(
    page_title="ETL Pipeline - Document Processing",
    page_icon="🔄",
    layout="wide"
)

def main():
    """Main unified interface for all ETL pipeline processing"""
    
    # Header
    st.title("🔄 ETL Pipeline - Document Processing")
    st.markdown("**Comprehensive document processing system** - Upload single files, multiple files, or paste text directly")
    
    # Main upload area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "📁 Upload Document(s)",
            type=["txt", "pdf", "png", "jpg", "jpeg", "html", "csv", "json", "md", "docx", "webp", "xlsx"],
            accept_multiple_files=True,
            help="Supports text, images, PDFs, Office docs, and structured data formats"
        )
    
    with col2:
        text_input = st.text_area(
            "📝 Or Paste Text", 
            height=150, 
            placeholder="Paste your content here...",
            help="For quick text processing without file upload"
        )

    # Auto-detect processing mode and show appropriate interface
    if uploaded_files and len(uploaded_files) > 1:
        show_batch_interface(uploaded_files)
    elif uploaded_files and len(uploaded_files) == 1:
        show_single_file_interface(uploaded_files[0])
    elif text_input.strip():
        show_text_interface(text_input)
    else:
        show_welcome_interface()

def show_welcome_interface():
    """Show welcome screen when no input is provided"""
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📄 Single File
        - Upload one document
        - Instant processing
        - Full content extraction
        - Export options
        """)
    
    with col2:
        st.markdown("""
        ### 📁 Batch Processing
        - Upload multiple files
        - Parallel processing
        - Progress tracking
        - Bulk export
        """)
    
    with col3:
        st.markdown("""
        ### 📝 Text Input
        - Paste content directly
        - No file required
        - Quick analysis
        - Immediate results
        """)
    
    st.info("👆 **Get started** by uploading file(s) or pasting text content above")

def show_single_file_interface(uploaded_file):
    """Interface for single file processing"""
    st.markdown("---")
    st.info(f"📄 **Single File Mode** - Processing: `{uploaded_file.name}`")
    
    # File details
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("File Name", uploaded_file.name)
    with col2:
        st.metric("File Size", f"{uploaded_file.size/1024:.1f} KB")
    with col3:
        st.metric("File Type", uploaded_file.type or "Unknown")
    
    # Process button
    if st.button("🔄 Process Document", type="primary", use_container_width=True):
        process_single_file(uploaded_file)

def show_text_interface(text_input):
    """Interface for text input processing"""
    st.markdown("---")
    st.info(f"📝 **Text Input Mode** - Processing {len(text_input)} characters")
    
    # Text preview
    with st.expander("📖 Text Preview", expanded=False):
        st.text_area("", value=text_input[:500] + "..." if len(text_input) > 500 else text_input, 
                    height=100, disabled=True)
    
    # Process button
    if st.button("🔄 Process Text", type="primary", use_container_width=True):
        process_single_text(text_input)

def show_batch_interface(uploaded_files):
    """Interface for batch processing"""
    st.markdown("---")
    st.info(f"📁 **Batch Mode** - {len(uploaded_files)} files ready for processing")
    
    # Batch configuration
    with st.expander("⚙️ Batch Processing Configuration", expanded=True):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            max_workers = st.slider("Parallel Workers", 1, 16, 4, 
                                   help="Number of concurrent processes")
        with col_b:
            use_processes = st.checkbox("Multi-processing", value=True, 
                                       help="Use processes (CPU-bound) vs threads (I/O-bound)")
        with col_c:
            save_outputs = st.checkbox("Auto-save", value=True, 
                                      help="Automatically save results to disk")
    
    # File queue preview
    st.subheader(f"📋 Processing Queue ({len(uploaded_files)})")
    file_details = []
    for f in uploaded_files:
        file_details.append({
            "Filename": f.name,
            "Size (KB)": round(f.size/1024, 2),
            "Type": f.type or "Unknown"
        })
    st.dataframe(pd.DataFrame(file_details), use_container_width=True, height=200)
    
    # Process button
    if st.button(f"🔄 Process {len(uploaded_files)} Files", type="primary", use_container_width=True):
        process_batch_files(uploaded_files, max_workers, use_processes, save_outputs)

def process_single_text(text_input):
    """Process text input with full pipeline processing"""
    try:
        with st.status("Processing text…", expanded=True) as _status:
            st.write("📥 Ingesting text input…")
            # Step 1: Ingest
            doc = ingest(file=None, text=text_input)
            st.write(f"✅ Detected: **{doc.detected_format}** · routed to `{doc.routing_target}`")

            # Step 2: Route to appropriate handler
            from handlers.text_handler import handle_text
            from handlers.binary_handler import handle_binary
            from handlers.mixed_handler import handle_mixed

            st.write("⚙️ Running handler…")
            if doc.routing_target == "text_handler":
                result = handle_text(doc)
            elif doc.routing_target == "binary_handler":
                result = handle_binary(doc)
            elif doc.routing_target == "mixed_handler":
                result = handle_mixed(doc)
            else:
                st.error(f"Unknown routing target: {doc.routing_target}")
                return

            method = getattr(result, "metadata", {}).get("extraction_method", "")
            st.write(f"✅ Done" + (f" · method: `{method}`" if method else ""))

            # Step 3: Generate full output
            output_data = {
                "document_id": result.document_id,
                "processing_metadata": {
                    "source_type": "text_input",
                    "detected_format": doc.detected_format,
                    "mime_type": doc.mime_type,
                    "routing_target": doc.routing_target,
                    "encoding": doc.encoding
                },
                "result": result.to_dict()
            }
            _status.update(label="✅ Text processed!", state="complete", expanded=False)

        show_processing_results(output_data, "text_input", result)

    except Exception as e:
        st.error(f"❌ Error during text processing: {str(e)}")
        with st.expander("🐛 Error Details"):
            st.exception(e)

def process_single_file(uploaded_file):
    """Process single uploaded file with full pipeline processing"""
    try:
        with st.status(f"Processing **{uploaded_file.name}** …", expanded=True) as _status:
            # Step 1: Ingest
            st.write("📥 Loading and detecting format…")
            doc = ingest(file=uploaded_file, text=None)
            st.write(f"✅ Detected: **{doc.detected_format}** · `{doc.mime_type}` · routed to `{doc.routing_target}`")

            # Hint about what happens next for binary docs
            if doc.routing_target == "binary_handler":
                mime = doc.mime_type or ""
                if "pdf" in mime:
                    st.write("🔍 Checking for embedded text (instant) — only loads GPU models if scanned…")
                elif "image" in mime:
                    st.write("🧠 Image detected — VLM + Chandra OCR (first run downloads ~13 GB of models)…")

            # Step 2: Route to appropriate handler
            from handlers.text_handler import handle_text
            from handlers.binary_handler import handle_binary
            from handlers.mixed_handler import handle_mixed

            st.write("⚙️ Running document handler…")
            if doc.routing_target == "text_handler":
                result = handle_text(doc)
            elif doc.routing_target == "binary_handler":
                result = handle_binary(doc)
            elif doc.routing_target == "mixed_handler":
                result = handle_mixed(doc)
            else:
                st.error(f"Unknown routing target: {doc.routing_target}")
                return

            method = getattr(result, "metadata", {}).get("extraction_method", "")
            st.write(f"✅ Extraction complete" + (f" · method: `{method}`" if method else ""))

            # Step 3: Generate full output
            output_data = {
                "document_id": result.document_id,
                "processing_metadata": {
                    "source_file": uploaded_file.name,
                    "file_size": uploaded_file.size,
                    "detected_format": doc.detected_format,
                    "mime_type": doc.mime_type,
                    "routing_target": doc.routing_target,
                    "encoding": doc.encoding
                },
                "result": result.to_dict()
            }
            _status.update(label=f"✅ **{uploaded_file.name}** processed!",
                           state="complete", expanded=False)

        show_processing_results(output_data, uploaded_file.name, result)

    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        with st.expander("🐛 Error Details"):
            st.exception(e)

def show_processing_results(output_data, source_name, result):
    """Show processing results in organized tabs"""
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Document ID", output_data["document_id"][:8] + "...")
    with col2:
        st.metric("Format", output_data["processing_metadata"]["detected_format"])
    with col3:
        st.metric("MIME Type", output_data["processing_metadata"]["mime_type"])
    with col4:
        st.metric("Handler", output_data["processing_metadata"]["routing_target"].replace("_handler", "").title())

    # Processing results with tabs
    tab1, tab2, tab3 = st.tabs(["📋 Extracted Content", "🔍 Technical Details", "📦 Export & Download"])
    
    with tab1:
        show_extracted_content(result)
    
    with tab2:
        show_technical_details(output_data)
    
    with tab3:
        show_export_options(output_data, source_name, result)

def show_extracted_content(result):
    """Display the extracted and processed content"""

    # ── BinaryDocument (PDF / image) ────────────────────────────────────────
    if hasattr(result, 'pages') and result.pages:
        _show_binary_content(result)
        return

    # ── TextDocument (HTML, CSV, TXT, DOCX, …) ──────────────────────────────
    if hasattr(result, 'sections') and result.sections:
        _show_text_content(result)
        return

    st.info("No structured content extracted")


def _show_binary_content(result):
    """Render a BinaryDocument: pages → blocks with VLM labels, titles and OCR text."""
    total_blocks  = sum(len(p.blocks)  for p in result.pages)
    total_regions = sum(len(p.regions) for p in result.pages)

    # ── Top metrics ─────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Pages",   len(result.pages))
    c2.metric("Blocks",  total_blocks,  help="VLM-detected content blocks")
    c3.metric("Regions", total_regions, help="Word-level OCR outputs")

    # ── Global search ────────────────────────────────────────────────────────
    search_term = st.text_input("🔍 Search OCR text:",
                                placeholder="Filter blocks by keyword...")

    st.markdown("---")

    for page in result.pages:
        blocks = page.blocks
        if not blocks:
            continue

        # Filter by search term
        if search_term:
            blocks = [b for b in blocks
                      if search_term.lower() in (b.raw_text or "").lower()
                      or search_term.lower() in (b.title or "").lower()]
            if not blocks:
                continue

        # ── Page header ──────────────────────────────────────────────────────
        layout_guided = page.metadata.get("layout_guided", False)
        badge = "🧠 VLM-guided" if layout_guided else "📄 Whole-page"
        st.markdown(f"### 📄 Page {page.page_number}  `{badge}`")

        for i, block in enumerate(blocks):
            raw_text   = (block.raw_text or "").strip()
            corr_text  = (block.corrected_text or "").strip()
            label      = block.label or "unknown"
            title      = block.title or ""
            confidence = block.confidence
            n_regions  = len(block.regions)

            # Colour-code the label badge
            LABEL_COLOURS = {
                "header": "🔵", "footer": "⚫", "table": "🟠",
                "figure": "🟣", "full_page": "⬜", "body": "🟢",
            }
            icon = LABEL_COLOURS.get(label, "🔷")

            # Build expander title
            conf_str  = f"{confidence:.0%}"
            title_str = f" — {title}" if title else ""
            expander_label = (
                f"{icon} Block {i+1}  `{label}`{title_str}  "
                f"· conf {conf_str}  · {n_regions} word(s)"
            )

            with st.expander(expander_label, expanded=(i == 0 and page.page_number == 1)):

                # ── Block metadata strip ─────────────────────────────────────
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.markdown(f"**Label**  `{label}`")
                mc2.markdown(f"**Title**  {title or '_(none)_'}")
                mc3.markdown(f"**Confidence**  {conf_str}")
                mc4.markdown(f"**BBox**  `{block.bbox}`")

                st.markdown("---")

                # ── OCR text ─────────────────────────────────────────────────
                if raw_text:
                    if corr_text and corr_text != raw_text:
                        t1, t2 = st.tabs(["✏️ Corrected text", "📝 Raw OCR text"])
                        with t1:
                            st.text_area("", value=corr_text, height=150,
                                         disabled=True,
                                         key=f"corr_p{page.page_number}_b{i}")
                        with t2:
                            st.text_area("", value=raw_text, height=150,
                                         disabled=True,
                                         key=f"raw_p{page.page_number}_b{i}")
                    else:
                        st.text_area("📝 OCR text", value=raw_text, height=150,
                                     disabled=True,
                                     key=f"raw_p{page.page_number}_b{i}")
                else:
                    st.caption("_(no text extracted for this block)_")

                # ── Word-level regions (collapsed) ───────────────────────────
                if block.regions:
                    with st.expander(f"🔤 Word-level regions ({n_regions})",
                                     expanded=False):
                        rows = [
                            {
                                "word":       r.text,
                                "confidence": f"{r.confidence:.2f}",
                                "bbox":       str(r.bbox),
                            }
                            for r in block.regions
                        ]
                        st.dataframe(pd.DataFrame(rows),
                                     use_container_width=True,
                                     hide_index=True)

        if search_term:
            st.caption(f"Showing {len(blocks)} matching block(s) on page {page.page_number}")

        st.markdown("---")


def _show_text_content(result):
    """Render a TextDocument: sections with search and filtering."""
    sections_to_process = result.sections
    st.subheader(f"📄 Extracted Sections ({len(sections_to_process)})")

    if len(sections_to_process) > 5:
        search_term = st.text_input("🔍 Search sections:",
                                    placeholder="Enter keywords to filter sections...")
        if search_term:
            filtered = [s for s in sections_to_process
                        if search_term.lower() in s.content.lower()]
            st.info(f"Found {len(filtered)} sections matching '{search_term}'")
            sections_to_show = filtered[:10]
        else:
            sections_to_show = sections_to_process[:10]
    else:
        sections_to_show = sections_to_process

    for i, section in enumerate(sections_to_show):
        section_id  = getattr(section, 'section_id',  f'section_{i}')
        format_type = getattr(section, 'format_type', 'extracted_content')

        with st.expander(f"Section {i+1}: {format_type} ({section_id[:8]}...)",
                         expanded=(i < 3)):
            st.write("**Content:**")
            preview = (section.content[:500] + "..."
                       if len(section.content) > 500 else section.content)
            st.text_area("", value=preview, height=100, disabled=True,
                         key=f"sec_content_{i}")
            if hasattr(section, 'metadata') and section.metadata:
                st.write("**Metadata:**")
                st.json(section.metadata, expanded=False)

    if len(sections_to_process) > len(sections_to_show):
        st.info(f"📝 Showing {len(sections_to_show)} of {len(sections_to_process)} "
                f"sections. Download full results to see all content.")

def show_technical_details(output_data):
    """Show technical processing details"""
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("🔍 Processing Metadata")
        st.json(output_data["processing_metadata"], expanded=False)
    
    with col_b:
        st.subheader("🛠️ Processing Result")
        st.json(output_data["result"], expanded=False)

def show_export_options(output_data, source_name, result):
    """Show export and download options"""
    st.markdown("### 📥 Download Options")
    
    # Create downloadable JSON
    json_str = json.dumps(output_data, indent=2, ensure_ascii=False)
    
    col_down1, col_down2 = st.columns(2)
    with col_down1:
        st.download_button(
            label="📥 Download Full Results (JSON)",
            data=json_str,
            file_name=f"etl_result_{source_name}_{output_data['document_id'][:8]}.json",
            mime="application/json",
            use_container_width=True
        )
    
    with col_down2:
        # Simple text export of just the content
        if hasattr(result, 'sections'):
            text_content = "\n\n".join(
                [f"=== {section.format_type.upper()} SECTION ===\n{section.content}"
                 for section in result.sections]
            )
        elif hasattr(result, 'pages'):
            page_parts = []
            for page in result.pages:
                block_texts = []
                for block in page.blocks:
                    text = (block.corrected_text or block.raw_text or "").strip()
                    if text:
                        header = f"[{block.label.upper()}]"
                        if block.title:
                            header += f" {block.title}"
                        block_texts.append(f"{header}\n{text}")
                if block_texts:
                    page_parts.append(
                        f"=== PAGE {page.page_number} ===\n" + "\n\n".join(block_texts)
                    )
            text_content = "\n\n".join(page_parts) if page_parts else "No extractable text found"
        else:
            text_content = "No extractable text content found"
        
        st.download_button(
            label="📄 Download Text Only",
            data=text_content,
            file_name=f"etl_text_{source_name}_{output_data['document_id'][:8]}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    st.markdown("### 👁️ Full JSON Preview")
    with st.expander("Show Complete Processing Results", expanded=False):
        st.json(output_data)

def process_batch_files(uploaded_files, max_workers, use_processes, save_outputs):
    """Process multiple files using batch logic"""
    # Prepare files for the processor
    files_data = []
    for f in uploaded_files:
        content = f.read()
        files_data.append({
            "name": f.name,
            "bytes": content
        })
        f.seek(0)  # Reset file pointer
    
    processor = BatchProcessor(
        max_workers=max_workers, 
        use_processes=use_processes,
        save_outputs=save_outputs
    )
    
    job_id = processor.create_batch_job(files_data)
    job = processor.jobs[job_id]
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    start_time = time.time()
    Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    # Process files with real-time UI updates
    with Executor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(_process_single_file_task, f): f for f in files_data}
        
        for i, future in enumerate(as_completed(future_to_file)):
            try:
                res_dict = future.result()
                if res_dict['status'] == 'success':
                    job.results.append(res_dict)
                    job.processed_files += 1
                else:
                    job.errors.append(res_dict)
                    job.failed_files += 1
            except Exception as exc:
                file_data = future_to_file[future]
                job.errors.append({
                    "file_name": file_data['name'],
                    "error_message": f"Worker error: {str(exc)}",
                    "status": "failed",
                    "processing_time": 0
                })
                job.failed_files += 1
            
            # Update Progress UI
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processed {i+1} of {len(uploaded_files)} files...")

    job.status = 'completed'
    job.completed_at = time.time()
    total_time = job.completed_at - start_time
    
    if save_outputs:
        processor._save_job_results(job)
    
    st.success(f"🎉 Batch processing completed in {total_time:.2f} seconds!")
    
    # Show batch results
    show_batch_results(job, total_time)

def show_batch_results(job, total_time):
    """Display batch processing results"""
    st.markdown("---")
    st.subheader("📊 Batch Processing Summary")
    
    # Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Files", job.total_files)
    with m2:
        st.metric("Success", job.processed_files, f"{job.processed_files/job.total_files*100:.1f}%")
    with m3:
        st.metric("Failures", job.failed_files, f"-{job.failed_files/job.total_files*100:.1f}%", delta_color="inverse")
    with m4:
        st.metric("Total Time", f"{total_time:.2f}s")
    
    # Detailed Results
    tab1, tab2, tab3 = st.tabs(["✅ Successful Results", "❌ Errors", "📦 Export All"])
    
    with tab1:
        if job.results:
            table_data = []
            for r in job.results:
                table_data.append({
                    "File Name": r['file_name'],
                    "Doc ID": r['document_id'][:8] + "...",
                    "Format": r['detected_format'],
                    "Target": r['routing_target'],
                    "Time (s)": round(r['processing_time'], 3)
                })
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)
        else:
            st.info("No files were successfully processed.")
            
    with tab2:
        if job.errors:
            err_data = []
            for e in job.errors:
                err_data.append({
                    "File Name": e['file_name'],
                    "Error Message": e['error_message'],
                    "Time (s)": round(e.get('processing_time', 0), 3)
                })
            st.dataframe(pd.DataFrame(err_data), use_container_width=True)
        else:
            st.success("No errors occurred during this batch!")
            
    with tab3:
        full_results = {
            "job_id": job.job_id,
            "summary": {
                "total_files": job.total_files,
                "processed_files": job.processed_files,
                "failed_files": job.failed_files,
                "processing_time": total_time,
                "timestamp": time.time()
            },
            "results": job.results,
            "errors": job.errors
        }
        
        st.markdown("Download complete batch processing results including all extracted data.")
        json_str = json.dumps(full_results, indent=2)
        st.download_button(
            label="📥 Download Complete Batch Results (JSON)",
            data=json_str,
            file_name=f"batch_results_{job.job_id}.json",
            mime="application/json",
            use_container_width=True
        )
        
        with st.expander("Preview Batch Results JSON"):
            st.json(full_results)

if __name__ == "__main__":
    main()