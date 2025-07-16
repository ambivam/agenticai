import sqlite3
from typing import List, Dict, Any, Optional
import json
import logging
from datetime import datetime, timedelta
from config import Config
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

class ChatMemory:
    def __init__(self, db_path: str = Config.MEMORY_PATH):
        """Initialize chat memory with SQLite database"""
        self.db_path = db_path
        self.chat_table = Config.CHAT_HISTORY_TABLE
        self.context_table = Config.CONTEXT_TABLE
        self.ttl_hours = Config.MEMORY_TTL
        self._init_db()
    
    def _init_db(self):
        """Initialize the database and create tables if they don't exist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create chat history table
                cursor.execute(f'''
                    CREATE TABLE IF NOT EXISTS {self.chat_table} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        message_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        metadata TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create context table
                cursor.execute(f'''
                    CREATE TABLE IF NOT EXISTS {self.context_table} (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        context_type TEXT NOT NULL,
                        content TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes
                cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_session_time ON {self.chat_table} (session_id, timestamp)')
                cursor.execute(f'CREATE INDEX IF NOT EXISTS idx_context_session ON {self.context_table} (session_id, context_type)')
                
                conn.commit()
        except Exception as e:
            logger.error(f"Error initializing chat memory database: {str(e)}")
            raise
    
    def save_message(self, session_id: str, message: BaseMessage):
        """Save a message to chat history."""
        try:
            message_type = message.__class__.__name__
            content = message.content
            metadata = json.dumps(message.additional_kwargs) if message.additional_kwargs else None
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"INSERT INTO {self.chat_table} (session_id, message_type, content, metadata) VALUES (?, ?, ?, ?)",
                    (session_id, message_type, content, metadata)
                )
                conn.commit()
                
            # Check if we need to generate a summary
            self._maybe_generate_summary(session_id)
            
        except Exception as e:
            logger.error(f"Error saving message to chat history: {str(e)}")
            raise
    
    def get_chat_history(self, session_id: str, limit: Optional[int] = None) -> List[BaseMessage]:
        """Retrieve chat history for a session."""
        try:
            if not limit:
                limit = Config.MAX_HISTORY_MESSAGES
                
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"""SELECT message_type, content, metadata 
                    FROM {self.chat_table} 
                    WHERE session_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?""",
                    (session_id, limit)
                )
                
                messages = []
                for msg_type, content, metadata in cursor.fetchall():
                    metadata = json.loads(metadata) if metadata else {}
                    if msg_type == "HumanMessage":
                        messages.append(HumanMessage(content=content, additional_kwargs=metadata))
                    elif msg_type == "AIMessage":
                        messages.append(AIMessage(content=content, additional_kwargs=metadata))
                    elif msg_type == "SystemMessage":
                        messages.append(SystemMessage(content=content, additional_kwargs=metadata))
                
                return list(reversed(messages))  # Return in chronological order
                
        except Exception as e:
            logger.error(f"Error retrieving chat history: {str(e)}")
            return []
    
    def save_context(self, session_id: str, context_type: str, content: Dict[str, Any]):
        """Save context information for a session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"INSERT INTO {self.context_table} (session_id, context_type, content) VALUES (?, ?, ?)",
                    (session_id, context_type, json.dumps(content))
                )
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving context: {str(e)}")
            raise
    
    def get_context(self, session_id: str, context_type: str) -> Optional[Dict[str, Any]]:
        """Retrieve context information for a session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"""SELECT content 
                    FROM {self.context_table} 
                    WHERE session_id = ? AND context_type = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1""",
                    (session_id, context_type)
                )
                
                result = cursor.fetchone()
                return json.loads(result[0]) if result else None
                
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return None
    
    def _maybe_generate_summary(self, session_id: str):
        """Generate a summary of the conversation if needed."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    f"""SELECT COUNT(*) 
                    FROM {self.chat_table} 
                    WHERE session_id = ? AND 
                    timestamp > (SELECT COALESCE(MAX(timestamp), '1970-01-01') 
                               FROM {self.context_table} 
                               WHERE session_id = ? AND context_type = 'summary')""",
                    (session_id, session_id)
                )
                
                count = cursor.fetchone()[0]
                if count >= Config.SUMMARY_TRIGGER_LENGTH:
                    recent_messages = self.get_chat_history(session_id, Config.SUMMARY_TRIGGER_LENGTH)
                    summary = self._generate_conversation_summary(recent_messages)
                    self.save_context(session_id, "summary", {"content": summary})
                    
        except Exception as e:
            logger.error(f"Error in summary generation check: {str(e)}")
    
    def _generate_conversation_summary(self, messages: List[BaseMessage]) -> str:
        """Generate a summary of the conversation."""
        try:
            # Convert messages to a format suitable for summarization
            conversation = "\n".join(
                f"{msg.__class__.__name__}: {msg.content}"
                for msg in messages
            )
            
            # Use the LLM to generate a summary
            summary_prompt = ChatPromptTemplate.from_messages([
                ("system", """Summarize the key points of this conversation segment. 
                Focus on:
                1. Main topics discussed
                2. Important decisions or conclusions
                3. Any pending questions or tasks
                
                Keep the summary concise but informative."""),
                ("human", "Conversation:\n{conversation}")
            ])
            
            llm = ChatOpenAI(
                model=Config.OPENAI_MODEL,
                temperature=0.3,
                openai_api_key=Config.OPENAI_API_KEY
            )
            
            response = llm.invoke(
                summary_prompt.format_messages(conversation=conversation)
            )
            
            return response.content
            
        except Exception as e:
            logger.error(f"Error generating conversation summary: {str(e)}")
            return "Error generating summary"
    
    def save_chat_history(self, session_id: str, messages: List[Dict[str, Any]], metadata: Dict[str, Any] = None):
        """Save a list of chat messages for a session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # First, delete existing messages for this session
                cursor.execute(f"DELETE FROM {self.chat_table} WHERE session_id = ?", (session_id,))
                
                # Insert new messages
                for message in messages:
                    message_type = message.get("role", "unknown")
                    content = message.get("content", "")
                    msg_metadata = {
                        **message,  # Include all message data
                        **(metadata or {}),  # Add any additional metadata
                        "timestamp": datetime.now().isoformat()
                    }
                    msg_metadata_json = json.dumps(msg_metadata)
                    
                    cursor.execute(
                        f"INSERT INTO {self.chat_table} (session_id, message_type, content, metadata) VALUES (?, ?, ?, ?)",
                        (session_id, message_type, content, msg_metadata_json)
                    )
                
                conn.commit()
                
            # Check if we need to generate a summary
            self._maybe_generate_summary(session_id)
            
        except Exception as e:
            logger.error(f"Error saving chat history: {str(e)}")
            raise
    
    def cleanup_expired(self):
        """Clean up expired chat histories and contexts."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                expiry_time = (datetime.now() - timedelta(hours=self.ttl_hours)).isoformat()
                
                # Clean up expired chat messages
                cursor.execute(
                    f"DELETE FROM {self.chat_table} WHERE timestamp < ?",
                    (expiry_time,)
                )
                
                # Clean up expired context
                cursor.execute(
                    f"DELETE FROM {self.context_table} WHERE timestamp < ?",
                    (expiry_time,)
                )
                
                conn.commit()
        except Exception as e:
            logger.error(f"Error cleaning up expired data: {str(e)}")
            raise
