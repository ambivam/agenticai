
import streamlit as st
import os
import time
from datetime import datetime
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage

# Set page config first
st.set_page_config(
    page_title="🤖 Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import custom modules
from config import Config
from utils import (
    create_custom_css, create_sidebar_info,
    display_success_message, display_error_message, display_warning_message,
    validate_file_upload, format_sources, format_file_size
)
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from agentic_workflow import AgenticWorkflow

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = []
    
    if 'vector_store_initialized' not in st.session_state:
        st.session_state.vector_store_initialized = False
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = {}

def handle_document_upload(doc_processor: DocumentProcessor, vector_store_manager: VectorStoreManager):
    """Handle document upload interface"""
    st.header("📁 Document Upload")
    st.markdown("Upload your documents to build the knowledge base. Supported formats: " + ", ".join(Config.SUPPORTED_FORMATS))
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files",
        accept_multiple_files=True,
        type=Config.SUPPORTED_FORMATS,
        key="file_uploader"
    )
    
    if uploaded_files:
        # Process button
        if st.button("📝 Process Documents", type="primary"):
            with st.spinner("🔄 Processing documents..."):
                for uploaded_file in uploaded_files:
                    try:
                        # Validate file
                        if not validate_file_upload(uploaded_file, Config.MAX_FILE_SIZE):
                            continue
                        
                        # Process document
                        st.info(f"Processing {uploaded_file.name}...")
                        documents = doc_processor.process_file(uploaded_file)
                        
                        # Add to vector store
                        if documents:
                            if vector_store_manager.add_documents(documents):
                                st.session_state.documents_processed.append(uploaded_file.name)
                                st.session_state.vector_store_initialized = True
                                display_success_message(f"Successfully processed {uploaded_file.name}")
                            else:
                                display_error_message(f"Failed to add {uploaded_file.name} to vector store")
                        else:
                            display_warning_message(f"No content extracted from {uploaded_file.name}")
                            
                    except Exception as e:
                        display_error_message(f"Error processing {uploaded_file.name}: {str(e)}")
                        continue
    
    # Show processed documents
    if st.session_state.documents_processed:
        st.markdown("### 📂 Processed Documents")
        for doc in st.session_state.documents_processed:
            st.write(f"- {doc}")
    else:
        st.info("No documents processed yet")

def main():
    """Main application function"""
    
    # Initialize session state
    initialize_session_state()
    
    # Create custom CSS
    create_custom_css()
    
    # Create sidebar info
    create_sidebar_info()
    
    # Check OpenAI API key
    if not Config.OPENAI_API_KEY:
        st.error("🔑 Please set your OPENAI_API_KEY in the environment variables or .env file")
        st.info("Create a .env file in the project directory and add: OPENAI_API_KEY=your_api_key_here")
        st.stop()
    
    try:
        # Initialize components
        doc_processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        vector_store_manager = VectorStoreManager(
            openai_api_key=Config.OPENAI_API_KEY,
            vector_db_path=Config.VECTOR_DB_PATH
        )
        
        agentic_workflow = AgenticWorkflow(
            openai_api_key=Config.OPENAI_API_KEY,
            vector_store_manager=vector_store_manager
        )
        
        # Check if vector store is initialized
        store_info = vector_store_manager.get_store_info()
        if store_info.get("initialized", False):
            st.session_state.vector_store_initialized = True
        
        # Create main tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📁 Document Upload", "🔍 Query Interface", "📊 Knowledge Base", "⚙️ Settings"])
        
        with tab1:
            handle_document_upload(doc_processor, vector_store_manager)
        
        with tab2:
            if not st.session_state.vector_store_initialized:
                st.warning("⚠️ Please upload and process some documents first")
                st.stop()
            
            st.header("🔍 Query Interface")
            st.markdown("Ask questions about your documents and get AI-powered answers with source citations.")
            
            # Query input
            query = st.text_area("Enter your question:", height=100)
            
            if st.button("Submit Query", type="primary"):
                if not query:
                    st.warning("Please enter a question")
                    st.stop()
                
                with st.spinner("🤔 Processing your query..."):
                    # Run agentic workflow
                    results = agentic_workflow.run_workflow(
                        query=query,
                        chat_history=st.session_state.chat_history
                    )
                    
                    if results["success"]:
                        # Display response
                        st.markdown("### 📝 Response")
                        st.markdown(results["response"])
                        
                        # Display sources
                        if results["sources"]:
                            st.markdown("### 📚 Sources")
                            for source in results["sources"]:
                                with st.expander(f"Source {source['id']}: {source['filename']} (Score: {source['similarity_score']:.3f})"):
                                    st.markdown(source["content"])
                        
                        # Display analysis
                        with st.expander("🔬 Query Analysis"):
                            st.json(results["analysis"])
                        
                        # Update chat history
                        st.session_state.chat_history.extend([
                            HumanMessage(content=query),
                            AIMessage(content=results["response"])
                        ])
                    else:
                        st.error(f"Error: {results['error']}")
        

        
        with tab3:
            handle_knowledge_base(vector_store_manager)
        
        with tab4:
            handle_settings(vector_store_manager)
    
    except Exception as e:
        st.error(f"Application initialization error: {str(e)}")
        st.info("Please check your configuration and try again.")

def handle_document_upload(doc_processor: DocumentProcessor, vector_store_manager: VectorStoreManager):
    """Handle document upload interface"""
    
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.subheader("📁 Upload Documents")
    st.markdown("Upload your documents to build the knowledge base. Supported formats: PDF, DOCX, TXT, CSV, XLSX, PPTX, JSON, HTML, XML")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Choose files",
        type=Config.SUPPORTED_FORMATS,
        accept_multiple_files=True,
        help=f"Maximum file size: {format_file_size(Config.MAX_FILE_SIZE)}"
    )
    
    if uploaded_files:
        st.write(f"📄 {len(uploaded_files)} file(s) selected")
        
        # Display file information
        for file in uploaded_files:
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"📎 {file.name}")
            with col2:
                st.write(f"{format_file_size(file.size)}")
            with col3:
                st.write(f".{file.name.split('.')[-1].upper()}")
        
        # Process files button
        if st.button("🚀 Process Documents", type="primary"):
            process_documents(uploaded_files, doc_processor, vector_store_manager)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display processing status
    if st.session_state.documents_processed:
        st.subheader("📋 Processing History")
        for doc_info in st.session_state.documents_processed[-5:]:  # Show last 5
            with st.expander(f"✅ {doc_info['filename']} - {doc_info['timestamp']}"):
                st.write(f"📊 Chunks created: {doc_info['chunks']}")
                st.write(f"📝 Characters: {doc_info['characters']:,}")
                st.write(f"🔤 Words: {doc_info['words']:,}")

def process_documents(uploaded_files, doc_processor: DocumentProcessor, vector_store_manager: VectorStoreManager):
    """Process uploaded documents"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_files = len(uploaded_files)
    all_documents = []
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")
        progress_bar.progress((i + 0.5) / total_files)
        
        # Validate file
        is_valid, message = validate_file_upload(
            uploaded_file, 
            Config.MAX_FILE_SIZE, 
            Config.SUPPORTED_FORMATS
        )
        
        if not is_valid:
            display_error_message(f"File {uploaded_file.name}: {message}")
            continue
        
        try:
            # Process document
            documents = doc_processor.process_file(uploaded_file)
            all_documents.extend(documents)
            
            # Store processing info
            doc_info = {
                'filename': uploaded_file.name,
                'chunks': len(documents),
                'characters': sum(len(doc.page_content) for doc in documents),
                'words': sum(len(doc.page_content.split()) for doc in documents),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.documents_processed.append(doc_info)
            
            display_success_message(f"Processed {uploaded_file.name}: {len(documents)} chunks created")
            
        except Exception as e:
            display_error_message(f"Error processing {uploaded_file.name}: {str(e)}")
        
        progress_bar.progress((i + 1) / total_files)
    
    # Add to vector store
    if all_documents:
        status_text.text("Adding documents to vector store...")
        success = vector_store_manager.add_documents(all_documents, show_progress=False)
        
        if success:
            st.session_state.vector_store_initialized = True
            display_success_message(f"Successfully added {len(all_documents)} document chunks to knowledge base!")
        else:
            display_error_message("Failed to add documents to vector store")
    
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()

def handle_query_interface(agentic_workflow: AgenticWorkflow, vector_store_manager: VectorStoreManager):
    """Handle query interface"""
    
    st.markdown('<div class="query-section">', unsafe_allow_html=True)
    st.subheader("🔍 Ask Questions")
    
    # Check if vector store is initialized
    if not st.session_state.vector_store_initialized:
        st.warning("📚 Please upload and process documents first before asking questions.")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    # Query input
    query = st.text_area(
        "Enter your question:",
        height=100,
        placeholder="Ask anything about your uploaded documents...",
        help="Ask detailed questions about the content in your documents. The AI will analyze your query and provide comprehensive answers with source citations."
    )
    
    # Query options
    col1, col2 = st.columns([3, 1])
    with col1:
        search_depth = st.selectbox(
            "Search Depth:",
            options=["Standard", "Deep", "Comprehensive"],
            help="Standard: 5 sources, Deep: 8 sources, Comprehensive: 12 sources"
        )
    
    with col2:
        if st.button("🔍 Ask Question", type="primary"):
            if query.strip():
                process_query(query, agentic_workflow, search_depth)
            else:
                display_warning_message("Please enter a question")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💬 Chat History")
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
            with st.expander(f"Q: {chat['query'][:50]}... - {chat['timestamp']}"):
                st.markdown(f"**Question:** {chat['query']}")
                st.markdown(f"**Answer:** {chat['response']}")
                if chat.get('sources'):
                    st.markdown("**Sources:**")
                    for j, source in enumerate(chat['sources'][:3]):  # Show top 3 sources
                        st.markdown(f"- {source.get('filename', 'Unknown')} (Score: {source.get('similarity_score', 0):.3f})")

def process_query(query: str, agentic_workflow: AgenticWorkflow, search_depth: str):
    """Process user query"""
    
    # Show processing status
    with st.spinner("🤖 Analyzing your question and searching through documents..."):
        
        # Determine search parameters
        k_map = {"Standard": 5, "Deep": 8, "Comprehensive": 12}
        
        # Run agentic workflow
        results = agentic_workflow.run_workflow(query)
        
        if results["success"]:
            # Display response
            st.markdown('<div class="response-section">', unsafe_allow_html=True)
            st.subheader("🎯 Answer")
            st.markdown(results["response"])
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display sources
            if results["sources"]:
                st.subheader("📚 Sources")
                for i, source in enumerate(results["sources"]):
                    with st.expander(f"📄 Source {i+1}: {source.get('filename', 'Unknown')} (Relevance: {source.get('similarity_score', 0):.3f})"):
                        st.markdown(f"**File:** {source.get('filename', 'Unknown')}")
                        st.markdown(f"**Relevance Score:** {source.get('similarity_score', 0):.3f}")
                        st.markdown(f"**Content Preview:**")
                        st.text(source.get('content', 'No content available'))
            
            # Display analysis info
            if results.get("analysis"):
                with st.expander("🔍 Query Analysis Details"):
                    analysis = results["analysis"]
                    if "query_type" in analysis:
                        st.write(f"**Query Type:** {analysis.get('query_type', 'Unknown')}")
                    if "complexity" in analysis:
                        st.write(f"**Complexity:** {analysis.get('complexity', 'Unknown')}")
                    if "search_results_count" in results:
                        st.write(f"**Sources Searched:** {results['search_results_count']}")
            
            # Store in chat history
            chat_entry = {
                'query': query,
                'response': results["response"],
                'sources': results["sources"],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state.chat_history.append(chat_entry)
            
            display_success_message("Question answered successfully!")
            
        else:
            display_error_message(f"Error processing query: {results.get('error', 'Unknown error')}")

def handle_knowledge_base(vector_store_manager: VectorStoreManager):
    """Handle knowledge base interface"""
    
    st.subheader("📊 Knowledge Base Status")
    
    # Get store information
    store_info = vector_store_manager.get_store_info()
    
    if store_info.get("initialized", False):
        # Display statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📄 Total Documents", store_info.get("total_documents", 0))
        
        with col2:
            st.metric("🔢 Total Embeddings", store_info.get("total_embeddings", 0))
        
        with col3:
            st.metric("📁 Unique Files", store_info.get("unique_files", 0))
        
        with col4:
            avg_chunk_size = store_info.get("total_documents", 0)
            st.metric("📋 Avg Chunk Size", f"{Config.CHUNK_SIZE}")
        
        # Display files
        if store_info.get("files"):
            st.subheader("📁 Files in Knowledge Base")
            for file in store_info["files"]:
                st.write(f"📎 {file}")
        
        # Search functionality
        st.subheader("🔍 Search Knowledge Base")
        search_query = st.text_input("Search documents:", placeholder="Enter search terms...")
        
        if search_query:
            with st.spinner("Searching..."):
                results = vector_store_manager.similarity_search(search_query, k=5)
                
                if results:
                    st.write(f"Found {len(results)} relevant documents:")
                    for i, (doc, score) in enumerate(results):
                        with st.expander(f"Result {i+1} - {doc.metadata.get('filename', 'Unknown')} (Score: {score:.3f})"):
                            st.write(f"**Content:**")
                            st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                            st.write(f"**Metadata:** {doc.metadata}")
                else:
                    st.info("No relevant documents found.")
    
    else:
        st.info("📚 Knowledge base is empty. Please upload documents first.")

def handle_settings(vector_store_manager: VectorStoreManager):
    """Handle settings interface"""
    
    st.subheader("⚙️ Application Settings")
    
    # Configuration display
    st.subheader("🔧 Current Configuration")
    
    config_data = {
        "OpenAI Model": Config.OPENAI_MODEL,
        "Embedding Model": Config.OPENAI_EMBEDDING_MODEL,
        "Chunk Size": Config.CHUNK_SIZE,
        "Chunk Overlap": Config.CHUNK_OVERLAP,
        "Max File Size": format_file_size(Config.MAX_FILE_SIZE),
        "Supported Formats": ", ".join(Config.SUPPORTED_FORMATS)
    }
    
    for key, value in config_data.items():
        st.write(f"**{key}:** {value}")
    
    st.markdown("---")
    
    # Danger zone
    st.subheader("⚠️ Danger Zone")
    st.warning("These actions are irreversible!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Knowledge Base", type="secondary"):
            if st.session_state.get('confirm_clear', False):
                success = vector_store_manager.clear_vector_store()
                if success:
                    st.session_state.vector_store_initialized = False
                    st.session_state.documents_processed = []
                    st.session_state.chat_history = []
                    display_success_message("Knowledge base cleared successfully!")
                else:
                    display_error_message("Failed to clear knowledge base")
                st.session_state.confirm_clear = False
            else:
                st.session_state.confirm_clear = True
                st.error("Click again to confirm clearing the knowledge base")
    
    with col2:
        if st.button("🔄 Reset Application", type="secondary"):
            if st.session_state.get('confirm_reset', False):
                # Clear all session state
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.experimental_rerun()
            else:
                st.session_state.confirm_reset = True
                st.error("Click again to confirm resetting the application")
    
    # Environment variables
    st.subheader("🔐 Environment Variables")
    st.write("**OPENAI_API_KEY:** " + ("✅ Set" if Config.OPENAI_API_KEY else "❌ Not Set"))
    st.write("**LANGCHAIN_API_KEY:** " + ("✅ Set" if Config.LANGCHAIN_API_KEY else "❌ Not Set (Optional)"))

if __name__ == "__main__":
    main()
