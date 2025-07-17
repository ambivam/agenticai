import streamlit as st
from typing import List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage
from agentic_workflow import AgenticWorkflow
from vector_store import VectorStoreManager
import logging
from chat_memory_manager import ChatMemoryManager
from user_profile import UserProfile
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

def display_chat_message(message: Dict[str, Any], is_user: bool):
    """Display a chat message with appropriate styling"""
    try:
        content = message.get("content", "")
        if not content:
            logger.warning("Empty message content")
            return
            
        if is_user:
            st.markdown(
                f'<div class="chat-message user-message">👤 You: {content}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-message assistant-message">🤖 Assistant: {content}</div>',
                unsafe_allow_html=True
            )
            
            # Display sources if available
            sources = message.get("sources", [])
            if sources:
                with st.expander("View Sources"):
                    for source in sources:
                        filename = source.get("filename", "Unknown source")
                        content = source.get("content", "No content available")
                        st.markdown(f"📄 **{filename}**")
                        st.markdown(f"```\n{content}\n```")
                        
            # Display suggested follow-ups if available
            follow_ups = message.get("suggested_follow_ups", [])
            if follow_ups:
                with st.expander("Suggested Follow-up Questions"):
                    for question in follow_ups:
                        st.markdown(f"• {question}")
    except Exception as e:
        logger.error(f"Error displaying message: {str(e)}")
        st.error("Error displaying message")

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
        
        # Initialize chat memory manager and user profile
        chat_memory = ChatMemoryManager()
        user_profile = UserProfile()
        
        # Initialize conversation ID if not exists
        if "conversation_id" not in st.session_state:
            st.session_state.conversation_id = str(uuid.uuid4())
        
        # Initialize session ID in session state if not exists
        if "session_id" not in st.session_state:
            # Generate a more persistent session ID that won't change on reload
            st.session_state.session_id = str(uuid.uuid4())
        
        # Initialize chat history in session state if not exists
        if "chat_history" not in st.session_state:
            # Create new session and load chat history
            chat_memory.create_session(st.session_state.session_id)
            st.session_state.chat_history = chat_memory.get_chat_history(st.session_state.session_id)
            if st.session_state.chat_history is None:
                st.session_state.chat_history = []
        
        st.subheader("💬 Chat Interface")
        
        # Check if knowledge base is empty
        if not vector_store_manager.has_documents():
            st.info("📚 Please upload some documents to the knowledge base first.")
            return
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            if "chat_history" in st.session_state and st.session_state.chat_history:
                for message in st.session_state.chat_history:
                    if isinstance(message, dict) and "role" in message and "content" in message:
                        display_chat_message(
                            message,
                            message["role"] == "user"
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
                    
                    try:
                        response = agentic_workflow.run_workflow(
                            query=user_input,
                            chat_history=context,
                            context={
                                "user_profile": user_profile_data,
                                "score_threshold": st.session_state.score_threshold
                            }
                        )
                        
                        if not response.get("success"):
                            error_msg = response.get("error", "Unknown error occurred")
                            logger.error(f"Workflow error: {error_msg}")
                            st.error(f"Error processing query: {error_msg}")
                            return
                        
                        if not response.get("response"):
                            logger.error("Empty response from workflow")
                            st.error("I couldn't generate a response. Please try again.")
                            return
                            
                        logger.info(f"Got response with {len(response.get('sources', []))} sources")
                        
                        # Add messages to chat history and save to storage
                        user_message = {
                            "role": "user",
                            "content": user_input,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        st.session_state.chat_history.append(user_message)
                        
                        assistant_message = {
                            "role": "assistant",
                            "content": response["response"],
                            "sources": response.get("sources", []),
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        if response.get("suggested_follow_ups"):
                            assistant_message["suggested_follow_ups"] = response["suggested_follow_ups"]
                        
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
                                "last_query": user_input,
                                "score_threshold": st.session_state.score_threshold,
                                "user_profile": user_profile_data
                            }
                        )
                        
                    except Exception as e:
                        logger.error(f"Error processing message: {str(e)}")
                        st.error(f"An error occurred while processing your message: {str(e)}")
                        return
                
                # Force a rerun to update the chat display
                st.rerun()
        
        # Add a clear chat button
        if st.session_state.chat_history:
            if st.button("Clear chat history"):
                st.session_state.chat_history = []
                # Clear from persistent storage
                chat_memory.clear_session_history(st.session_state.session_id)
                # Reset conversation ID
                st.session_state.conversation_id = str(uuid.uuid4())
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
