import streamlit as st
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from agentic_workflow import AgenticWorkflow
from vector_store import VectorStoreManager
import logging
from chat_memory import ChatMemory
from user_profile import UserProfile
from datetime import datetime

logger = logging.getLogger(__name__)

def display_chat_message(message: Dict[str, Any], is_user: bool):
    """Display a chat message with appropriate styling"""
    if is_user:
        st.markdown(
            f'<div class="chat-message user-message">👤 You: {message["content"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="chat-message assistant-message">🤖 Assistant: {message["content"]}</div>',
            unsafe_allow_html=True
        )
        if "sources" in message:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.markdown(f"📄 **{source['filename']}**")
                    st.markdown(f"```\n{source['content']}\n```")

def handle_chat_interface(agentic_workflow: AgenticWorkflow, vector_store_manager: VectorStoreManager):
    """Handle the chat interface section of the application"""
    try:
        st.header("💬 Chat Interface")
        st.markdown("Chat with your documents using natural language.")
        
        # Search Settings in a collapsible section
        with st.expander("🔍 Search Settings", expanded=False):
            if "score_threshold" not in st.session_state:
                st.session_state.score_threshold = 0.5
            new_threshold = st.slider(
                "Similarity Score Threshold",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.score_threshold,
                step=0.05,
                help="Adjust the minimum similarity score required for search results. Lower values show more results but may be less relevant."
            )
            if new_threshold != st.session_state.score_threshold:
                st.session_state.score_threshold = new_threshold
                st.success(f"✅ Search threshold updated to {new_threshold}")
        
        # Initialize chat memory and user profile
        chat_memory = ChatMemory()
        user_profile = UserProfile()
        
        # Initialize session ID in session state if not exists
        if "session_id" not in st.session_state:
            # Generate a more persistent session ID that won't change on reload
            import uuid
            st.session_state.session_id = str(uuid.uuid4())
        
        # Initialize chat history in session state if not exists
        if "chat_history" not in st.session_state:
            # Try to load from persistent storage
            st.session_state.chat_history = chat_memory.get_chat_history(st.session_state.session_id)
        
        st.subheader("💬 Chat Interface")
        
        # Check if knowledge base is empty
        if not vector_store_manager.has_documents():
            st.info("📚 Please upload some documents to the knowledge base first.")
            return
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                display_chat_message(
                    message,
                    isinstance(message, dict) and message.get("role") == "user"
                )
        
        # Chat input
        with st.form(key="chat_form", clear_on_submit=True):
            col1, col2 = st.columns([6, 1])
            
            with col1:
                user_input = st.text_input(
                    "Your message:",
                    placeholder="Ask me anything about the documents...",
                    key="chat_input"
                )
            
            with col2:
                submit_button = st.form_submit_button("Send")
            
            if submit_button and user_input:
                # Add user message to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                # Extract and update user profile information
                user_info = user_profile.extract_user_info(user_input)
                if user_info:
                    current_profile = user_profile.get_profile(st.session_state.session_id)
                    current_profile.update(user_info)
                    user_profile.update_profile(st.session_state.session_id, current_profile)
                
                # Process query with context
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
                        query=user_input,
                        chat_history=context,
                        context={
                            "user_profile": user_profile_data,
                            "score_threshold": st.session_state.score_threshold
                        }
                    )
                    
                    # Add assistant message to chat history
                    assistant_message = {
                        "role": "assistant",
                        "content": response["response"],
                        "sources": response["sources"],
                        "timestamp": st.session_state.get("current_time", "")
                    }
                    
                    # If there are suggested follow-ups, add them
                    if response.get("suggested_follow_ups"):
                        assistant_message["suggested_follow_ups"] = response["suggested_follow_ups"]
                    st.session_state.chat_history.append(assistant_message)
                    
                    # Save to persistent storage
                    chat_memory.save_chat_history(
                        st.session_state.session_id,
                        st.session_state.chat_history,
                        metadata={"last_query": user_input}
                    )
                
                # Force a rerun to update the chat display
                st.rerun()
        
        # Add a clear chat button
        if st.session_state.chat_history:
            if st.button("Clear chat history"):
                st.session_state.chat_history = []
                # Clear from persistent storage
                chat_memory.delete_chat_history(st.session_state.session_id)
                st.rerun()
    
    except Exception as e:
        logger.error(f"Error in chat interface: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

def add_chat_styles():
    """Add custom CSS styles for chat interface"""
    st.markdown("""
        <style>
        .chat-message {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
            white-space: pre-wrap;
        }
        .user-message {
            background-color: #e6f3ff;
            border-left: 5px solid #2196F3;
        }
        .assistant-message {
            background-color: #f0f0f0;
            border-left: 5px solid #4CAF50;
        }
        .stButton > button {
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)
