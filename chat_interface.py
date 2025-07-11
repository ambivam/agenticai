import streamlit as st
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from agentic_workflow import AgenticWorkflow
from vector_store import VectorStoreManager
import logging

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
        # Initialize chat history if not exists
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
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
                user_message = {
                    "role": "user",
                    "content": user_input,
                    "timestamp": st.session_state.get("current_time", "")
                }
                st.session_state.chat_history.append(user_message)
                
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
                    response = agentic_workflow.run_workflow(
                        query=user_input,
                        chat_history=context
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
                
                # Force a rerun to update the chat display
                st.rerun()
        
        # Add a clear chat button
        if st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
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
