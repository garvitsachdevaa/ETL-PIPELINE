# ETL Pipeline UI

## Streamlit App

### 🔄 `streamlit_app_unified.py`
**The unified Streamlit interface** for all ETL pipeline processing.

**Features:**
- ✅ **Auto-detection**: Single file, batch, or text input modes
- ✅ **Full processing**: Complete pipeline from ingestion to export
- ✅ **Smart UI**: Adapts interface based on input type
- ✅ **Export options**: JSON and text downloads
- ✅ **Batch processing**: Parallel processing with progress tracking
- ✅ **Error handling**: Comprehensive error reporting

**Usage:**
```bash
python main.py --ui
# or directly:
streamlit run ui/streamlit_app_unified.py
```

---

## Legacy Apps (For Reference)

### 📄 `streamlit_app.py` (LEGACY)
**Original single-file processing app** - Now superseded by unified app.

**Historical Purpose:**
- Simple single-file ingestion and processing
- Clean interface for individual documents

**Status:** ⚠️ **DEPRECATED** - Use `streamlit_app_unified.py` instead

### 📁 `streamlit_app_batch.py` (LEGACY)
**Original batch processing app** - Now superseded by unified app.

**Historical Purpose:**
- Multi-file batch processing
- Parallel processing capabilities
- Advanced configuration options

**Status:** ⚠️ **DEPRECATED** - Use `streamlit_app_unified.py` instead

---

## Why We Consolidated

**Previous Issues:**
1. **Duplicate functionality**: Both apps could process single files
2. **User confusion**: Users didn't know which app to use
3. **Split features**: Some features only in one app
4. **Maintenance burden**: Two UIs to update and maintain

**Solution:**
- **Single unified app** that auto-detects processing mode
- **All features in one place**: Single file, batch, text input
- **Consistent interface**: Same UI patterns throughout
- **Easier maintenance**: One codebase to update

---

## Migration Notes

If you were using the old apps:

### From `streamlit_app.py`:
```bash
# Old way:
streamlit run ui/streamlit_app.py

# New way:
python main.py --ui
# The unified app will automatically detect single file mode
```

### From `streamlit_app_batch.py`:
```bash
# Old way:
streamlit run ui/streamlit_app_batch.py

# New way:
python main.py --ui
# The unified app will automatically detect batch mode when multiple files are uploaded
```

**All functionality is preserved** - just in a cleaner, unified interface!