import sys
from pathlib import Path
import os
import time
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project root (etl_pipeline/) to sys.path so all imports work
sys.path.append(str(Path(__file__).resolve().parents[1]))

# ── Critical env vars — set before ANY other import ─────────────────────────
# Prevents tokenizer deadlocks in Streamlit's multi-threaded environment.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

# ── HuggingFace authentication ──────────────────────────────────────────────
print("[startup] authenticating with HuggingFace...", flush=True)
_hf_token = (
    os.environ.get("HF_TOKEN")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    or os.environ.get("HUGGINGFACE_TOKEN")
    or os.environ.get("HF_API_TOKEN")
)
if _hf_token:
    os.environ["HUGGING_FACE_HUB_TOKEN"] = _hf_token
    os.environ["HF_TOKEN"] = _hf_token
    try:
        from huggingface_hub import login as _hf_login
        _hf_login(token=_hf_token, add_to_git_credential=False)
        print("[startup] HF login succeeded.", flush=True)
    except Exception as _e:
        print(f"[startup] HF login failed: {_e}", flush=True)
else:
    print("[startup] WARNING: no HF_TOKEN found — gated models will fail.", flush=True)

print("[startup] importing streamlit...", flush=True)
import streamlit as st
import pandas as pd

print("[startup] importing ingestion.loader...", flush=True)
try:
    from ingestion.loader import ingest
    print("[startup] ingest ok", flush=True)
except Exception as _e:
    print(f"[startup] FATAL: could not import ingest: {_e}", flush=True)
    raise

# ── st.set_page_config MUST be the first st.* call ──────────────────────────
# No @st.cache_resource or other st.* calls above this line.
st.set_page_config(
    page_title="ETL Pipeline - Document Processing",
    page_icon="🔄",
    layout="wide"
)

print("[startup] app ready.", flush=True)


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

    # Clear cached results when a different file is loaded
    if st.session_state.get("_last_file") != uploaded_file.name:
        st.session_state.pop("_result", None)
        st.session_state.pop("_output_data", None)
        st.session_state.pop("_source_name", None)
        st.session_state["_last_file"] = uploaded_file.name

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

    # Re-render results if they are already in session state (survives tab/button reruns)
    elif "_result" in st.session_state and st.session_state.get("_last_file") == uploaded_file.name:
        show_processing_results(
            st.session_state["_output_data"],
            st.session_state["_source_name"],
            st.session_state["_result"],
        )

def show_text_interface(text_input):
    """Interface for text input processing"""
    st.markdown("---")
    st.info(f"📝 **Text Input Mode** - Processing {len(text_input)} characters")

    # Clear cached results when text content changes
    if st.session_state.get("_last_text") != text_input:
        st.session_state.pop("_result", None)
        st.session_state.pop("_output_data", None)
        st.session_state.pop("_source_name", None)
        st.session_state["_last_text"] = text_input

    # Text preview
    with st.expander("📖 Text Preview", expanded=False):
        st.text_area("", value=text_input[:500] + "..." if len(text_input) > 500 else text_input,
                    height=100, disabled=True)

    # Process button
    if st.button("🔄 Process Text", type="primary", use_container_width=True):
        process_single_text(text_input)

    # Re-render results if already in session state
    elif "_result" in st.session_state and st.session_state.get("_last_text") == text_input:
        show_processing_results(
            st.session_state["_output_data"],
            st.session_state["_source_name"],
            st.session_state["_result"],
        )

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

        # Persist so chunking tab survives reruns
        st.session_state["_result"] = result
        st.session_state["_output_data"] = output_data
        st.session_state["_source_name"] = "text_input"
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
                if "pdf" in mime or "image" in mime:
                    st.write(
                        "🧠 Step 1: PaliGemma detecting layout blocks (columns, headers, tables)…  "
                        "\n⚡ Step 2: Chandra OCR on each detected block — "
                        "first run loads models (~13 GB total, one-time), subsequent files are fast."
                    )

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

        # Persist so chunking tab survives reruns
        st.session_state["_result"] = result
        st.session_state["_output_data"] = output_data
        st.session_state["_source_name"] = uploaded_file.name
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
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Extracted Content", "🔍 Technical Details", "📦 Export & Download", "🗂️ Chunk & Context"])
    
    with tab1:
        show_extracted_content(result)
    
    with tab2:
        show_technical_details(output_data)
    
    with tab3:
        show_export_options(output_data, source_name, result)

    with tab4:
        show_chunking_tab(result)

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

def show_chunking_tab(result):
    """
    Chunking & Context tab — lets the user pick a chunking strategy,
    runs it against the already-extracted document, and displays the
    output ready for the downstream SLM extractor.
    """
    st.subheader("🗂️ Chunk & Context")
    st.markdown(
        "Choose how to split the extracted text into chunks. "
        "The result is the input your fine-tuned SLM will consume for "
        "entity and relation extraction."
    )

    # ── Method selector ─────────────────────────────────────────────────────
    METHOD_INFO = {
        "line":      ("📄 Line by Line",          "Each non-empty line becomes one chunk."),
        "paragraph": ("📝 Paragraph by Paragraph", "Blank-line boundaries define paragraphs."),
        "section":   ("📑 Section by Section",     "Detects headings, chapter markers, horizontal rules."),
        "context":   ("🧠 By Context (BERTopic)",  
                      "Paragraph-level base split → sentence-transformers embeddings → "
                      "BERTopic topic assignment. Semantically related paragraphs "
                      "(e.g. all mentions of 'John') are grouped into the same "
                      "ContextGroup, even if they appear on different pages."),
    }

    method = st.radio(
        "Select chunking method:",
        options=list(METHOD_INFO.keys()),
        format_func=lambda k: METHOD_INFO[k][0],
        horizontal=True,
    )

    # Description beneath the selector
    st.caption(f"ℹ️ {METHOD_INFO[method][1]}")

    # ── Context-mode grouping controls ──────────────────────────────────
    nr_topics_input: int | None = None  # default for non-context modes
    if method == "context":
        st.info(
            "**BERTopic mode** will:\n"
            "1. Split text into paragraphs first\n"
            "2. Generate sentence-transformer embeddings\n"
            "3. Cluster semantically similar paragraphs → ContextGroups\n\n"
            "First run loads the embedding model (~90 MB). "
            "Requires ≥5 paragraphs — short documents fall back to paragraph mode."
        )

        st.markdown("**Number of Context Groups**")
        col_mode, col_input = st.columns([1, 1])
        with col_mode:
            grouping_mode = st.radio(
                "Mode",
                options=["Auto", "Manual"],
                horizontal=True,
                label_visibility="collapsed",
                help="Auto: BERTopic finds the natural number of topics. Manual: you decide.",
            )
        with col_input:
            if grouping_mode == "Manual":
                nr_topics_input = st.number_input(
                    "Number of groups",
                    min_value=2,
                    max_value=50,
                    value=4,
                    step=1,
                    help=(
                        "How many context groups to create. "
                        "BERTopic first finds all natural topics, then merges "
                        "down to this number. If you ask for more than naturally "
                        "exist, the natural count is kept."
                    ),
                )
            else:
                st.caption(
                    "🤖 BERTopic will find the optimal number of natural topic groups automatically."
                )

    st.markdown("---")

    # ── Run chunking ────────────────────────────────────────────────────────
    if st.button("▶️ Run Chunking", type="primary", use_container_width=True):
        try:
            from chunking import Chunker
            with st.spinner(f"Chunking document ({METHOD_INFO[method][0]})…"):
                chunk_result = Chunker.chunk(result, method=method, nr_topics=nr_topics_input)

            st.success(
                f"✅ **{chunk_result.total_chunks}** chunk(s) produced "
                f"using **{METHOD_INFO[method][0]}**"
            )

            # ── Context mode: warn if manual nr_topics was capped ─────────
            if method == "context" and nr_topics_input is not None:
                actual_groups = len([g for g in chunk_result.context_groups if g.topic_id != -1])
                if nr_topics_input > actual_groups:
                    st.warning(
                        f"⚠️ You requested **{nr_topics_input} groups**, but BERTopic found only "
                        f"**{actual_groups} natural topic(s)** in this document — "
                        f"showing {actual_groups}. "
                        f"You cannot create more groups than there are distinct semantic topics. "
                        f"Try a document with more varied content, or switch to **Auto** mode."
                    )

            # ── Context mode: show grouped view + flat view ────────────────
            if method == "context" and chunk_result.context_groups:
                _show_context_groups(chunk_result)
            else:
                _show_flat_chunks(chunk_result)

            # ── SLM payload download ───────────────────────────────────────
            st.markdown("---")
            st.markdown("### 📤 Send to SLM Extractor")
            payload_json = json.dumps(chunk_result.slm_payload, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Download SLM Payload (JSON)",
                data=payload_json,
                file_name=f"slm_payload_{method}_{chunk_result.total_chunks}chunks.json",
                mime="application/json",
                use_container_width=True,
            )
            with st.expander("Preview SLM Payload", expanded=False):
                st.json(chunk_result.slm_payload)

        except ImportError as ie:
            st.error(f"❌ Missing dependency: {ie}")
        except Exception as exc:
            st.error(f"❌ Chunking failed: {exc}")
            with st.expander("🐛 Error Details"):
                st.exception(exc)


def _show_flat_chunks(chunk_result):
    """Display flat list of chunks (line / paragraph / section modes)."""
    st.subheader(f"📦 {chunk_result.total_chunks} Chunks")

    # Search / filter
    search = st.text_input("🔍 Filter chunks:", placeholder="Keyword to search…")
    chunks = chunk_result.chunks
    if search:
        chunks = [c for c in chunks if search.lower() in c.text.lower()]
        st.caption(f"{len(chunks)} chunk(s) match '{search}'")

    # Show first 50 to avoid overwhelming the UI
    display_chunks = chunks[:50]
    for chunk in display_chunks:
        page_info = ""
        if "page_number" in chunk.metadata:
            page_info = f" · Page {chunk.metadata['page_number']}"
        label_info = ""
        if "block_label" in chunk.metadata:
            label_info = f" · `{chunk.metadata['block_label']}`"
        if "format_type" in chunk.metadata:
            label_info = f" · `{chunk.metadata['format_type']}`"

        with st.expander(
            f"Chunk {chunk.chunk_index + 1}{page_info}{label_info} — "
            f"{len(chunk.text)} chars",
            expanded=False,
        ):
            st.text_area("", value=chunk.text, height=120, disabled=True,
                         key=f"flat_chunk_{chunk.chunk_id}")
            st.json(chunk.metadata, expanded=False)

    if len(chunks) > 50:
        st.info(f"Showing 50 of {len(chunks)} chunks. Download the SLM payload to see all.")


def _show_context_groups(chunk_result):
    """Display BERTopic context groups (Scenario C).

    Each ContextGroup exposes:
      - merged_chunk  : ONE combined chunk the SLM receives
      - source_chunks : individual paragraphs with cosine similarity scores
    """
    groups        = chunk_result.context_groups
    named_groups  = [g for g in groups if g.topic_id != -1]
    outlier_groups = [g for g in groups if g.topic_id == -1]
    individual    = chunk_result.chunks   # all paragraphs in doc order

    # ── Header ────────────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    col_a.metric("Context Groups (merged)", len(named_groups))
    col_b.metric("Individual Paragraphs", len(individual))

    st.caption(
        "Each group's **merged chunk** is what the SLM receives. "
        "Expand a group to also inspect every source paragraph with its "
        "cosine-similarity score to the topic centroid."
    )
    if outlier_groups:
        n_out = len(outlier_groups[0].source_chunks) if outlier_groups[0].source_chunks else 0
        st.caption(f"+ 1 outlier group — **{n_out}** paragraph(s) with no clear topic (topic_id = -1)")

    st.markdown("---")

    # ── Per-group display ──────────────────────────────────────────────────
    for group in groups:
        is_outlier   = group.topic_id == -1
        icon         = "🔘" if is_outlier else "🏷️"
        label        = group.topic_label
        kw_str       = "  ·  ".join(f"`{w}`" for w, _ in group.topic_words[:5]) \
                       if group.topic_words else "_(no keywords)_"
        merged       = group.merged_chunk            # new field
        srcs         = group.source_chunks           # new field — list of Chunk with sim scores
        para_count   = len(srcs)
        merged_len   = len(merged.text) if merged else 0
        avg_sim      = merged.metadata.get("avg_similarity_score", 0.0) if merged else 0.0

        with st.expander(
            f"{icon} **{label}**  ·  {para_count} paragraph(s) merged  ·  "
            f"{merged_len} chars  ·  avg sim {avg_sim:.2f}",
            expanded=(not is_outlier),
        ):
            # Keywords row
            st.markdown(f"**Top keywords:** {kw_str}")
            if merged:
                src_indices = merged.metadata.get("source_paragraph_indices", [])
                if src_indices:
                    st.caption(f"Source paragraph positions (0-based in document): {src_indices}")
            st.markdown("---")

            # ── TAB 1: Merged chunk for the SLM ──────────────────────────
            tab_merged, tab_individual = st.tabs(
                ["📦 Merged — SLM Input", "🔍 Individual Paragraphs + Similarity"]
            )

            with tab_merged:
                if merged:
                    st.text_area(
                        "Merged text sent to SLM",
                        value=merged.text,
                        height=250,
                        disabled=True,
                        key=f"merged_{group.topic_id}",
                    )
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Characters", len(merged.text))
                    c2.metric("Paragraphs merged", para_count)
                    c3.metric("Avg similarity", f"{avg_sim:.3f}")
                else:
                    st.info("No merged chunk (outlier group).")

            # ── TAB 2: Individual paragraphs with similarity scores ───────
            with tab_individual:
                if not srcs:
                    st.info("No source paragraphs.")
                for i, para in enumerate(srcs):
                    sim   = para.similarity_score
                    n_rel = len(para.related_chunk_ids)

                    # Colour-code similarity badge
                    if sim >= 0.75:
                        badge_colour = "🟢"
                    elif sim >= 0.50:
                        badge_colour = "🟡"
                    else:
                        badge_colour = "🔴"

                    with st.expander(
                        f"Para {para.chunk_index + 1}  "
                        f"{badge_colour} sim={sim:.3f}  ·  {len(para.text)} chars  "
                        f"·  related to {n_rel} other para(s)",
                        expanded=False,
                    ):
                        st.text_area(
                            "",
                            value=para.text,
                            height=100,
                            disabled=True,
                            key=f"src_para_{group.topic_id}_{i}",
                            label_visibility="collapsed",
                        )
                        meta_display = {
                            "similarity_score": sim,
                            "related_chunk_ids": para.related_chunk_ids,
                            "topic_id": para.metadata.get("topic_id"),
                            "chunk_id": para.chunk_id,
                        }
                        st.json(meta_display, expanded=False)

    # ── Flat view of ALL individual paragraphs in document order ──────────
    if individual:
        st.markdown("---")
        with st.expander(
            f"📄 All {len(individual)} individual paragraph(s) in document order",
            expanded=False,
        ):
            st.caption(
                "Every paragraph in its original position, tagged with topic label "
                "and similarity score. This mirrors what is sent in the `individual_chunks` "
                "field of the SLM payload."
            )
            for para in individual:
                sim   = para.similarity_score
                badge = "🟢" if sim >= 0.75 else ("🟡" if sim >= 0.50 else "🔴")
                topic = para.metadata.get("topic_label", "outlier")
                st.markdown(
                    f"**Para {para.chunk_index + 1}** — `{topic}`  "
                    f"{badge} sim={sim:.3f}"
                )
                st.text_area(
                    "",
                    value=para.text,
                    height=80,
                    disabled=True,
                    key=f"all_para_{para.chunk_id}",
                    label_visibility="collapsed",
                )


def process_batch_files(uploaded_files, max_workers, use_processes, save_outputs):
    """Process multiple files using batch logic"""
    from ingestion.batch_processor import BatchProcessor, _process_single_file_task
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