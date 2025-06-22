
import os
import hashlib
import streamlit as st
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_page_config():
    """Setup Streamlit page configuration"""
    st.set_page_config(
        page_title="🤖 Agentic RAG Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def create_custom_css():
    """Create custom CSS for the application"""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #007acc;
        margin: 1rem 0;
    }
    
    .query-section {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .response-section {
        background-color: #f9f9f9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    
    .source-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 3px solid #007acc;
    }
    
    .status-success {
        color: #28a745;
    }
    
    .status-error {
        color: #dc3545;
    }
    
    .status-warning {
        color: #ffc107;
    }
    </style>
    """, unsafe_allow_html=True)

def get_file_hash(file_content: bytes) -> str:
    """Generate hash for file content"""
    return hashlib.md5(file_content).hexdigest()

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024
        i += 1
    return f"{size_bytes:.1f} {size_names[i]}"

def display_success_message(message: str):
    """Display success message"""
    st.markdown(f'<div class="status-success">✅ {message}</div>', unsafe_allow_html=True)

def display_error_message(message: str):
    """Display error message"""
    st.markdown(f'<div class="status-error">❌ {message}</div>', unsafe_allow_html=True)

def display_warning_message(message: str):
    """Display warning message"""
    st.markdown(f'<div class="status-warning">⚠️ {message}</div>', unsafe_allow_html=True)

def create_progress_bar(progress: float, text: str = ""):
    """Create a progress bar with text"""
    progress_bar = st.progress(progress)
    if text:
        st.text(text)
    return progress_bar

def format_sources(sources: List[Dict[str, Any]]) -> str:
    """Format source information for display"""
    if not sources:
        return "No sources available"
    
    formatted_sources = []
    for i, source in enumerate(sources, 1):
        source_text = f"**Source {i}:**\n"
        if 'filename' in source:
            source_text += f"📄 File: {source['filename']}\n"
        if 'page' in source:
            source_text += f"📖 Page: {source['page']}\n"
        if 'content' in source:
            content_preview = source['content'][:200] + "..." if len(source['content']) > 200 else source['content']
            source_text += f"📝 Content: {content_preview}\n"
        formatted_sources.append(source_text)
    
    return "\n---\n".join(formatted_sources)

def validate_file_upload(uploaded_file, max_size: int, supported_formats: List[str]) -> tuple:
    """Validate uploaded file"""
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Check file size
    if uploaded_file.size > max_size:
        return False, f"File size exceeds maximum limit of {format_file_size(max_size)}"
    
    # Check file format
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension not in supported_formats:
        return False, f"Unsupported file format. Supported formats: {', '.join(supported_formats)}"
    
    return True, "File validation successful"

def create_sidebar_info():
    """Create sidebar with application information"""
    with st.sidebar:
        st.markdown("## 🤖 Agentic RAG Assistant")
        st.markdown("---")
        
        st.markdown("### 📋 Features")
        st.markdown("""
        - 🔄 **Agentic Workflows**: Multi-step reasoning
        - 📚 **Multi-format Support**: PDF, Word, Excel, PPT, etc.
        - 🧠 **Smart Chunking**: Intelligent document processing
        - 🔍 **Vector Search**: FAISS-powered similarity search
        - 📖 **Source Attribution**: Detailed citations
        - 🎯 **Query Planning**: Strategic question decomposition
        """)
        
        st.markdown("### 📁 Supported Formats")
        st.markdown("""
        - 📄 **Documents**: PDF, DOCX, TXT
        - 📊 **Spreadsheets**: XLSX, CSV
        - 🎨 **Presentations**: PPTX
        - 🌐 **Web**: HTML, XML
        - 💾 **Data**: JSON
        """)
        
        st.markdown("### ⚙️ Configuration")
        st.markdown("""
        - **Model**: GPT-4 Turbo
        - **Embeddings**: text-embedding-3-small
        - **Vector DB**: FAISS (Local)
        - **Chunk Size**: 1000 tokens
        """)
