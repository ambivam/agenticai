import streamlit as st
import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    validate_file_upload, format_sources, format_file_size,
    get_file_type
)
from document_processor import DocumentProcessor
from vector_store import VectorStoreManager
from agentic_workflow import AgenticWorkflow
from chat_interface import handle_chat_interface, add_chat_styles
from repl_interface import handle_repl_interface

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'vector_store_initialized' not in st.session_state:
        st.session_state.vector_store_initialized = False
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = []
    if 'show_response' not in st.session_state:
        st.session_state.show_response = False
    if 'current_query' not in st.session_state:
        st.session_state.current_query = None
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = None
    if 'query_submitted' not in st.session_state:
        st.session_state.query_submitted = False
    if 'is_follow_up' not in st.session_state:
        st.session_state.is_follow_up = False

def handle_document_upload(doc_processor: DocumentProcessor, vector_store_manager: VectorStoreManager):
    """Handle document upload interface"""
    try:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.subheader("📁 Upload Documents")
        st.markdown("Upload your documents to build the knowledge base. Supported formats: PDF, DOCX, TXT, CSV, XLSX, PPTX, JSON, HTML, XML")
        
        # File upload
        st.session_state.uploaded_files = st.file_uploader(
            "Choose files",
            type=Config.SUPPORTED_FORMATS,
            accept_multiple_files=True,
            help=f"Maximum file size: {format_file_size(Config.MAX_FILE_SIZE)}"
        )
        
        if st.session_state.uploaded_files:
            st.write(f"📄 {len(st.session_state.uploaded_files)} file(s) selected")
            
            # Display file information
            for file in st.session_state.uploaded_files:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"📎 {file.name}")
                with col2:
                    st.write(f"{format_file_size(file.size)}")
                with col3:
                    st.write(f"{get_file_type(file.name)}")
            
            # Process button
            if st.button("🔄 Process Documents", type="primary", use_container_width=True):
                with st.spinner("Processing documents..."):
                    all_documents = []
                    
                    # Process each file
                    for file in st.session_state.uploaded_files:
                        if file.size > Config.MAX_FILE_SIZE:
                            st.error(f"⚠️ {file.name} exceeds maximum file size of {format_file_size(Config.MAX_FILE_SIZE)}")
                            continue
                        
                        try:
                            # Process file
                            documents = doc_processor.process_file(file)
                            if documents:
                                all_documents.extend(documents)
                                st.success(f"✅ Successfully processed {file.name}: {len(documents)} chunks created")
                            else:
                                st.warning(f"⚠️ No content extracted from {file.name}")
                        except Exception as e:
                            st.error(f"❌ Failed to process {file.name}: {str(e)}")
                            logging.error(f"Error processing file {file.name}: {str(e)}")
                    
                    # Update vector store
                    if all_documents:
                        try:
                            # Add documents to vector store
                            success = vector_store_manager.add_documents(all_documents)
                            if success:
                                st.session_state.vector_store_initialized = True
                                store_info = vector_store_manager.get_store_info()
                                st.success(f"✨ Vector store updated successfully! Total documents: {store_info.get('total_documents', 0)}")
                                
                                # Clear uploaded files from session state
                                st.session_state.uploaded_files = None
                                st.experimental_rerun()
                            else:
                                st.error("Failed to add documents to vector store")
                        except Exception as e:
                            st.error(f"❌ Failed to update vector store: {str(e)}")
                            logging.error(f"Vector store error: {str(e)}")
                    else:
                        st.warning("⚠️ No documents were processed successfully")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Document upload interface error: {str(e)}")
        logging.error(f"Document upload interface error: {str(e)}")

def main():
    """Main application function"""
    try:
        # Initialize session state
        initialize_session_state()
        
        # Create custom CSS
        create_custom_css()
        add_chat_styles()
        
        # Create sidebar
        create_sidebar_info()
        
        # Ensure OpenAI API key is set
        if not Config.OPENAI_API_KEY:
            st.error("❌ OpenAI API key is not set. Please set it in your environment variables.")
            return
        
        # Initialize components
        try:
            doc_processor = DocumentProcessor(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )
            
            # Ensure vector store path is absolute
            vector_db_path = os.path.abspath(Config.VECTOR_DB_PATH)
            os.makedirs(vector_db_path, exist_ok=True)
            
            vector_store_manager = VectorStoreManager(
                openai_api_key=Config.OPENAI_API_KEY,
                vector_db_path=vector_db_path
            )
            
            # Check vector store status
            store_info = vector_store_manager.get_store_info()
            if store_info.get("initialized", False):
                st.session_state.vector_store_initialized = True
                logger.info(f"Vector store loaded with {store_info.get('total_documents', 0)} documents")
            else:
                logger.info("Vector store is empty")
                
            # Initialize workflow with vector store manager
            agentic_workflow = AgenticWorkflow(
                openai_api_key=Config.OPENAI_API_KEY,
                vector_store_manager=vector_store_manager
            )
            
            # Create tabs
            chat_tab, query_tab, code_tab, kb_tab, settings_tab = st.tabs([
                "💬 Chat", "🔍 Query Interface", "💻 Code Playground",
                "📚 Knowledge Base", "⚙️ Settings"
            ])
            
            # Handle each tab
            with chat_tab:
                handle_chat_interface(agentic_workflow, vector_store_manager)
            
            with query_tab:
                handle_document_upload(doc_processor, vector_store_manager)
            
            with code_tab:
                handle_repl_interface()
            
            with kb_tab:
                handle_knowledge_base(vector_store_manager)
            
            with settings_tab:
                handle_settings(vector_store_manager)
                
        except Exception as e:
            st.error(f"Failed to initialize components: {str(e)}")
            logger.error(f"Component initialization error: {str(e)}")
            return
        
        # Document upload section (always visible)
        if not st.session_state.vector_store_initialized:
            handle_document_upload(doc_processor, vector_store_manager)
        
        # Create tabs
        chat_tab, query_tab, code_tab, kb_tab, settings_tab = st.tabs([
            "💬 Chat", "🔍 Query Interface", "💻 Code Playground",
            "📚 Knowledge Base", "⚙️ Settings"
        ])
        
        # Handle each tab
        with chat_tab:
            handle_chat_interface(agentic_workflow, vector_store_manager)
        
        with query_tab:
            handle_document_upload(doc_processor, vector_store_manager)
        
        with code_tab:
            handle_repl_interface()
        
        with kb_tab:
            handle_knowledge_base(vector_store_manager)
        
        with settings_tab:
            handle_settings(vector_store_manager)
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your configuration and try again.")


    
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
        
        try:
            # Validate file size
            if uploaded_file.size > Config.MAX_FILE_SIZE:
                st.warning(f"⚠️ {uploaded_file.name} exceeds size limit of {format_file_size(Config.MAX_FILE_SIZE)}")
                continue
            
            # Process document
            documents = doc_processor.process_file(uploaded_file)
            if documents:
                all_documents.extend(documents)
                
                # Record document info
                doc_info = {
                    'filename': uploaded_file.name,
                    'chunks': len(documents),
                    'characters': sum(len(doc.page_content) for doc in documents),
                    'words': sum(len(doc.page_content.split()) for doc in documents),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                st.session_state.documents_processed.append(doc_info)
                st.success(f"✅ Processed {uploaded_file.name}: {len(documents)} chunks created")
            else:
                st.warning(f"⚠️ No content extracted from {uploaded_file.name}")
                
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            logging.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        progress_bar.progress((i + 1) / total_files)
    
    # Add to vector store
    if all_documents:
        status_text.text("Adding documents to vector store...")
        try:
            vector_store_manager.add_documents(all_documents)
            st.session_state.vector_store_initialized = True
            st.success("✨ Documents added to vector store successfully!")
            # Clear uploaded files immediately after successful vector store update
            st.session_state.uploaded_files = []
            st.rerun()  # Refresh the UI
        except Exception as e:
            st.error(f"❌ Failed to add documents to vector store: {str(e)}")
            logging.error(f"Vector store error: {str(e)}")
    else:
        st.warning("⚠️ No documents were processed successfully")
    
    progress_bar.progress(1.0)
    status_text.text("Processing complete!")
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()

def handle_query_interface(agentic_workflow: AgenticWorkflow, vector_store_manager: VectorStoreManager):
    """Handle the query interface section of the application"""
    try:
        # Check if vector store has documents
        if not vector_store_manager.has_documents():
            st.warning("⚠️ Please upload and process some documents first")
            return
    except Exception as e:
        st.error(f"Error checking vector store: {str(e)}")
        return
    
    # Initialize session state for query interface
    if 'show_response' not in st.session_state:
        st.session_state.show_response = False
    if 'current_query' not in st.session_state:
        st.session_state.current_query = None
    
    st.header("🔍 Query Interface")
    st.markdown("Ask questions about your documents and get AI-powered answers with source citations.")
    
    # Follow-up checkbox
    is_follow_up = st.checkbox(
        "💬 Is this a follow-up question?"
    )
    
    # Query input
    user_query = st.text_area(
        "Enter your question:",
        placeholder="Ask anything about your documents...",
        height=100
    )
    
    # Search depth selector
    search_depth = st.select_slider(
        "Search depth:",
        options=["Standard", "Deep", "Comprehensive"],
        value="Standard",
        help="Adjust how thoroughly to search through documents"
    )
    
    # Submit button
    if st.button("🔍 Submit Query", type="primary", use_container_width=True):
        if not user_query:
            st.warning("⚠️ Please enter a question")
            return
        
        with st.spinner("🤖 Processing your query..."):
            response = process_query(
                query=user_query,
                agentic_workflow=agentic_workflow,
                search_depth=search_depth,
                is_follow_up=is_follow_up
            )
            
            if response:
                try:
                    st.markdown('<div class="response-section">', unsafe_allow_html=True)
                    st.subheader("🎯 Answer")
                    
                    # Handle both string and dict responses
                    if isinstance(response, dict):
                        if "response" in response:
                            st.markdown(response["response"])
                        else:
                            st.markdown(str(response))
                        
                        # Display sources if available
                        if response.get("sources"):
                            st.markdown("**📚 Sources:**")
                            for source in response["sources"]:
                                st.markdown(f"- {source}")
                        
                        # Display suggested follow-ups
                        if response.get("suggested_follow_ups"):
                            st.markdown("**💡 Suggested follow-up questions:**")
                            for suggestion in response["suggested_follow_ups"]:
                                if st.button(f"🔍 {suggestion}", key=f"follow_up_{hash(suggestion)}"):
                                    return process_query(
                                        query=suggestion,
                                        agentic_workflow=agentic_workflow,
                                        search_depth=search_depth,
                                        is_follow_up=True
                                    )
                    else:
                        st.markdown(str(response))
                except Exception as e:
                    st.error(f"Error displaying response: {str(e)}")
                    logger.error(f"Error displaying response: {str(e)}")

def process_query(query: str, agentic_workflow: AgenticWorkflow, search_depth: str, is_follow_up: bool = False):
    """Process user query"""
    try:
        # Get conversation history
        try:
            with st.spinner("Thinking..."):
                # Get conversation context
                context = []
                for msg in st.session_state.chat_history[-3:]:  # Last 3 messages for context
                    if msg.get("role") == "user":
                        context.append(HumanMessage(content=msg["content"]))
                    else:
                        context.append(AIMessage(content=msg["content"]))
                
                # Get response
                # Get user profile
                user_profile_data = user_profile.get_profile(st.session_state.session_id)
                
                response = agentic_workflow.run_workflow(
                    query=query,
                    chat_history=context,
                    context={
                        "user_profile": user_profile_data,
                        "search_sources": st.session_state.search_sources
                    }
                )
        except Exception as e:
            st.error(f"Error processing query: {str(e)}")
            logger.error(f"Query processing error: {str(e)}")
            return
        
        if response:
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": query,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            return response
        return None
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise

def handle_knowledge_base(vector_store_manager: VectorStoreManager):
    """Handle the knowledge base section of the application"""
    st.header("📚 Knowledge Base")
    
    try:
        # Get store information
        store_info = vector_store_manager.get_store_info()
        
        if store_info.get("initialized", False):
            # Display statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Total Documents",
                    store_info.get("total_documents", 0)
                )
            
            with col2:
                st.metric(
                    "Total Embeddings",
                    store_info.get("total_embeddings", 0)
                )
            
            # Show document list
            st.subheader("📁 Document List")
            if store_info.get("total_documents", 0) > 0:
                st.info("Document listing coming soon...")
            else:
                st.info("ℹ️ No documents in knowledge base")
        else:
            st.warning("⚠️ Knowledge base is empty. Please upload some documents first.")
    
    except Exception as e:
        st.error(f"Error displaying knowledge base: {str(e)}")
        logger.error(f"Knowledge base error: {str(e)}")

def handle_repl_interface():
    """Handle the REPL interface section of the application"""
    st.header("💻 Code Playground")
    st.markdown("Coming soon...")

def handle_settings(vector_store_manager: VectorStoreManager):
    """Handle settings section of the application"""
    st.header("⚙️ Settings")
    
    # Initialize debug mode in session state if not exists
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = False
    
    # Debug mode toggle
    debug_mode = st.checkbox("🔍 Debug Mode", value=st.session_state.debug_mode,
                          help="Show additional debugging information in the chat interface")
    
    if debug_mode != st.session_state.debug_mode:
        st.session_state.debug_mode = debug_mode
        st.rerun()
    
    st.divider()
    
    # Vector store management
    st.subheader("📂 Vector Store Management")
    
    # Display vector store info
    store_info = vector_store_manager.get_store_info()
    st.markdown(f"**Total Documents:** {store_info.get('total_documents', 0)}")
    st.markdown(f"**Total Embeddings:** {store_info.get('total_embeddings', 0)}")
    
    # Clear vector store button
    if st.button("🗑️ Clear Vector Store"):
        if vector_store_manager.clear_vector_store():
            st.session_state.vector_store_initialized = False
            st.success("✨ Vector store cleared successfully!")
            st.rerun()
        else:
            st.error("❌ Failed to clear vector store")

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

def main():
    try:
        # Initialize components
        doc_processor = DocumentProcessor()
        vector_store_manager = VectorStoreManager(openai_api_key=Config.OPENAI_API_KEY)
        
        # Initialize workflow with vector store manager
        agentic_workflow = AgenticWorkflow(
            openai_api_key=Config.OPENAI_API_KEY,
            vector_store_manager=vector_store_manager
        )
        
        # Initialize session state
        initialize_session_state()
        
        # Create custom CSS
        create_custom_css()
        add_chat_styles()
        
        # Create sidebar
        create_sidebar_info()
        
        # Create tabs
        chat_tab, query_tab, code_tab, kb_tab, settings_tab = st.tabs([
            "💬 Chat", "🔍 Query Interface", "💻 Code Playground",
            "📚 Knowledge Base", "⚙️ Settings"
        ])
        
        # Handle each tab
        with chat_tab:
            handle_chat_interface(agentic_workflow, vector_store_manager)
        
        with query_tab:
            handle_document_upload(doc_processor, vector_store_manager)
        
        with code_tab:
            handle_repl_interface()
        
        with kb_tab:
            handle_knowledge_base(vector_store_manager)
        
        with settings_tab:
            handle_settings(vector_store_manager)
    
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
