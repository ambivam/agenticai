import streamlit as st
import os
import time
import logging
import uuid
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
from chat_memory_manager import ChatMemoryManager
from chat_memory_tab import ChatMemoryTab
from rag_chat_tab import RagChatTab
from user_profile import UserProfile
from langchain_openai import ChatOpenAI

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
    if 'score_threshold' not in st.session_state:
        st.session_state.score_threshold = 0.1  # Default similarity threshold
    if 'llm' not in st.session_state:
        st.session_state.llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            temperature=0.7,
            api_key=Config.OPENAI_API_KEY
        )
    if 'chat_memory_tab' not in st.session_state:
        st.session_state.chat_memory_tab = ChatMemoryTab()
    if 'rag_chat_tab' not in st.session_state:
        st.session_state.rag_chat_tab = RagChatTab()

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
                            vector_store_manager.add_documents(all_documents)
                            st.session_state.vector_store_initialized = True
                            st.success("✨ Vector store updated successfully!")
                        except Exception as e:
                            st.error(f"❌ Failed to update vector store: {str(e)}")
                            logging.error(f"Vector store error: {str(e)}")
                    else:
                        st.warning("⚠️ No documents were processed successfully")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Document upload interface error: {str(e)}")
        logging.error(f"Document upload interface error: {str(e)}")

def handle_chat_memory():
    """Handle the chat memory interface"""
    if 'chat_memory_tab' not in st.session_state:
        st.session_state.chat_memory_tab = ChatMemoryTab()
    st.session_state.chat_memory_tab.display_chat_interface()

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
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "📤 Upload",
            "💬 Chat",
            "💭 Memory Chat",
            "🔍 RAG Chat",
            "🔎 Query",
            "📚 Knowledge Base",
            "⚙️ Settings"
        ])
        
        with tab1:
            handle_document_upload(doc_processor, vector_store_manager)
        
        with tab2:
            from mysql_chat import main as mysql_chat_main
            mysql_chat_main()
        
        with tab3:
            handle_chat_memory()
        
        with tab4:
            # Handle RAG Chat
            try:
                docs = vector_store_manager.get_all_documents()
                if docs:  # Check if there are actual documents
                    # Add documents to RAG chat if not already added
                    if not getattr(st.session_state.rag_chat_tab, 'documents_added', False):
                        texts = [doc.page_content for doc in docs]
                        metadatas = [doc.metadata for doc in docs]
                        if st.session_state.rag_chat_tab.add_documents(texts, metadatas):
                            st.session_state.rag_chat_tab.documents_added = True
                            logger.info(f"Added {len(docs)} documents to RAG Chat")
                    # Display RAG chat interface
                    st.session_state.rag_chat_tab.display_chat_interface()
                else:
                    st.info("ℹ️ The knowledge base is empty. Upload some documents in the Upload tab to start using RAG Chat.")
            except Exception as e:
                logger.error(f"Error initializing RAG Chat: {str(e)}")
                st.error("There was an error loading the knowledge base. Please try uploading your documents again.")
        
        with tab5:
            handle_query_interface(agentic_workflow, vector_store_manager)
        
        with tab6:
            handle_knowledge_base(vector_store_manager)
        
        with tab7:
            handle_settings(vector_store_manager)
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your configuration and try again.")


    
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
                st.write(f".{file.name.split('.')[-1].upper()}")
        
        # Process files button
        if st.button("🚀 Process Documents", type="primary"):
            process_documents(st.session_state.uploaded_files, doc_processor, vector_store_manager)
    
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
        # Initialize chat memory manager and user profile
        chat_memory = ChatMemoryManager()
        user_profile = UserProfile()
        
        # Initialize session ID if not exists
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
            
        # Initialize conversation ID if not exists
        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = str(uuid.uuid4())
            
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = chat_memory.get_chat_history(st.session_state.session_id) or []
            
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
def process_query(query: str, agentic_workflow: AgenticWorkflow, search_depth: str, is_follow_up: bool = False):
    """Process user query"""
    try:
        # Initialize chat memory manager
        chat_memory = ChatMemoryManager()
        
        # Initialize session and conversation IDs if not exists
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
            chat_memory.create_session(st.session_state.session_id)
            
        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = str(uuid.uuid4())
        
        # Initialize chat history if not exists
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = chat_memory.get_chat_history(st.session_state.session_id) or []
        
        # Get conversation context - use last 5 messages for context
        context = []
        current_conversation = [msg for msg in st.session_state.chat_history 
                            if msg.get('conversation_id') == st.session_state.conversation_id]
        
        for msg in current_conversation[-5:]:
            if isinstance(msg.get('content'), (str, dict)):
                if msg.get('role') == 'user':
                    context.append(HumanMessage(content=msg['content']))
                else:
                    content = msg['content']
                    if isinstance(content, dict):
                        content = content.get('response', str(content))
                    context.append(AIMessage(content=content))
        
        # Get user profile
        user_profile = UserProfile()
        user_profile_data = user_profile.get_profile(st.session_state.session_id)
        
        # Process query with context
        response = agentic_workflow.run_workflow(
            query=query,
            chat_history=context,
            context={
                "user_profile": user_profile_data,
                "score_threshold": st.session_state.get("score_threshold", 0.1),
                "search_depth": search_depth,
                "is_follow_up": is_follow_up
            }
        )
        
        if not response or not response.get("response"):
            logger.error("Empty response from workflow")
            return None
            
        # Create user message
        user_message = {
            "role": "user",
            "content": query,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "conversation_id": st.session_state.conversation_id,
            "session_id": st.session_state.session_id
        }
        
        # Create assistant message
        assistant_message = {
            "role": "assistant",
            "content": response["response"],
            "sources": response.get("sources", []),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "conversation_id": st.session_state.conversation_id,
            "session_id": st.session_state.session_id
        }
        
        if response.get("suggested_follow_ups"):
            assistant_message["suggested_follow_ups"] = response["suggested_follow_ups"]
        
        # Update chat history
        st.session_state.chat_history.append(user_message)
        st.session_state.chat_history.append(assistant_message)
        
        # Save messages to persistent storage
        chat_memory.save_message(
            session_id=st.session_state.session_id,
            conversation_id=st.session_state.conversation_id,
            message=user_message
        )
        
        chat_memory.save_message(
            session_id=st.session_state.session_id,
            conversation_id=st.session_state.conversation_id,
            message=assistant_message
        )
        
        # Update conversation context
        chat_memory.update_context(
            conversation_id=st.session_state.conversation_id,
            session_id=st.session_state.session_id,
            topic=response.get("current_topic"),
            context_data={
                "last_query": query,
                "score_threshold": st.session_state.get("score_threshold", 0.1),
                "user_profile": user_profile_data
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        st.error(f"Error processing query: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise

def handle_knowledge_base(vector_store_manager: VectorStoreManager):
    """Handle knowledge base interface"""
    
    st.subheader("📊 Knowledge Base Status")
    
    try:
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
                st.metric("📋 Avg Chunk Size", f"{Config.CHUNK_SIZE}")
            
            # Display file list
            if store_info.get("files", []):
                st.subheader("📂 Processed Files")
                for file_info in store_info["files"]:
                    st.text(f"📄 {file_info}")
            else:
                st.info("⚠️ No files have been processed yet")
        else:
            st.warning("⚠️ Knowledge base is not initialized. Please process some documents first.")
    except Exception as e:
        st.error(f"Error accessing knowledge base: {str(e)}")
        return
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
