import sqlite3
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ChatMemoryManager:
    """Manages persistent chat history and context using SQLite"""
    
    def __init__(self, db_path: str = "chat_memory.db"):
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize the SQLite database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create chat_sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_sessions (
                        session_id TEXT PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_active TIMESTAMP,
                        metadata TEXT
                    )
                """)
                
                # Create chat_messages table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS chat_messages (
                        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        conversation_id TEXT,
                        role TEXT,
                        content TEXT,
                        timestamp TIMESTAMP,
                        metadata TEXT,
                        FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id)
                    )
                """)
                
                # Create conversation_context table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_context (
                        conversation_id TEXT PRIMARY KEY,
                        session_id TEXT,
                        current_topic TEXT,
                        context_data TEXT,
                        last_updated TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES chat_sessions (session_id)
                    )
                """)
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise
    
    def create_session(self, session_id: str, metadata: Optional[Dict] = None) -> None:
        """Create a new chat session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO chat_sessions (session_id, last_active, metadata)
                    VALUES (?, CURRENT_TIMESTAMP, ?)
                    """,
                    (session_id, json.dumps(metadata or {}))
                )
                conn.commit()
                logger.info(f"Created new session: {session_id}")
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
            raise
    
    def save_message(self, session_id: str, conversation_id: str, message: Dict[str, Any]) -> None:
        """Save a chat message to the database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Update session last active time
                cursor.execute(
                    "UPDATE chat_sessions SET last_active = CURRENT_TIMESTAMP WHERE session_id = ?",
                    (session_id,)
                )
                
                # Extract message metadata
                metadata = {
                    "sources": message.get("sources", []),
                    "suggested_follow_ups": message.get("suggested_follow_ups", [])
                }
                
                # Save message
                cursor.execute(
                    """
                    INSERT INTO chat_messages (session_id, conversation_id, role, content, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        conversation_id,
                        message["role"],
                        message["content"],
                        message.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                        json.dumps(metadata)
                    )
                )
                conn.commit()
                logger.info(f"Saved message for session {session_id}")
        except Exception as e:
            logger.error(f"Error saving message: {str(e)}")
            raise
    
    def get_chat_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve chat history for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT role, content, timestamp, metadata
                    FROM chat_messages
                    WHERE session_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                    """,
                    (session_id, limit)
                )
                
                messages = []
                for row in cursor.fetchall():
                    role, content, timestamp, metadata = row
                    message = {
                        "role": role,
                        "content": content,
                        "timestamp": timestamp
                    }
                    
                    # Add metadata if present
                    if metadata:
                        meta_dict = json.loads(metadata)
                        message.update(meta_dict)
                    
                    messages.append(message)
                
                return list(reversed(messages))  # Return in chronological order
                
        except Exception as e:
            logger.error(f"Error retrieving chat history: {str(e)}")
            return []
    
    def update_context(self, conversation_id: str, session_id: str, topic: Optional[str] = None, context_data: Optional[Dict] = None) -> None:
        """Update conversation context"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO conversation_context
                    (conversation_id, session_id, current_topic, context_data, last_updated)
                    VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        conversation_id,
                        session_id,
                        topic,
                        json.dumps(context_data or {})
                    )
                )
                conn.commit()
                logger.info(f"Updated context for conversation {conversation_id}")
        except Exception as e:
            logger.error(f"Error updating context: {str(e)}")
            raise
    
    def get_context(self, conversation_id: str) -> Dict[str, Any]:
        """Retrieve conversation context"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT current_topic, context_data
                    FROM conversation_context
                    WHERE conversation_id = ?
                    """,
                    (conversation_id,)
                )
                
                row = cursor.fetchone()
                if row:
                    topic, context_data = row
                    return {
                        "current_topic": topic,
                        "context": json.loads(context_data) if context_data else {}
                    }
                return {"current_topic": None, "context": {}}
                
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return {"current_topic": None, "context": {}}
    
    def clear_session_history(self, session_id: str) -> None:
        """Clear chat history for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM chat_messages WHERE session_id = ?", (session_id,))
                cursor.execute("DELETE FROM conversation_context WHERE session_id = ?", (session_id,))
                conn.commit()
                logger.info(f"Cleared history for session {session_id}")
        except Exception as e:
            logger.error(f"Error clearing session history: {str(e)}")
            raise
