import sqlite3
from typing import List, Dict, Any
import json
import logging
from datetime import datetime, timedelta
from config import Config

logger = logging.getLogger(__name__)

class ChatMemory:
    def __init__(self, db_path: str = Config.MEMORY_PATH):
        """Initialize chat memory with SQLite database"""
        self.db_path = db_path
        self.table_name = Config.MEMORY_TABLE
        self.ttl_hours = Config.MEMORY_TTL
        self._init_db()
    
    def _init_db(self):
        """Initialize the database and create table if it doesn't exist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f'''
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        session_id TEXT PRIMARY KEY,
                        chat_history TEXT,
                        last_updated TIMESTAMP,
                        metadata TEXT
                    )
                ''')
                conn.commit()
        except Exception as e:
            logger.error(f"Error initializing chat memory database: {str(e)}")
            raise
    
    def save_chat_history(self, session_id: str, chat_history: List[Dict[str, Any]], metadata: Dict[str, Any] = None):
        """Save chat history for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                current_time = datetime.now().isoformat()
                
                # Convert chat history to JSON string
                chat_history_json = json.dumps(chat_history)
                metadata_json = json.dumps(metadata) if metadata else '{}'
                
                # Insert or update chat history
                cursor.execute(f'''
                    INSERT OR REPLACE INTO {self.table_name}
                    (session_id, chat_history, last_updated, metadata)
                    VALUES (?, ?, ?, ?)
                ''', (session_id, chat_history_json, current_time, metadata_json))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Error saving chat history: {str(e)}")
            raise
    
    def get_chat_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve chat history for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get chat history
                cursor.execute(f'''
                    SELECT chat_history, last_updated
                    FROM {self.table_name}
                    WHERE session_id = ?
                ''', (session_id,))
                
                result = cursor.fetchone()
                if result:
                    chat_history_json, last_updated = result
                    
                    # Check TTL
                    last_updated_dt = datetime.fromisoformat(last_updated)
                    if datetime.now() - last_updated_dt > timedelta(hours=self.ttl_hours):
                        # History expired, delete it
                        self.delete_chat_history(session_id)
                        return []
                    
                    return json.loads(chat_history_json)
                return []
        except Exception as e:
            logger.error(f"Error retrieving chat history: {str(e)}")
            return []
    
    def delete_chat_history(self, session_id: str):
        """Delete chat history for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f'''
                    DELETE FROM {self.table_name}
                    WHERE session_id = ?
                ''', (session_id,))
                conn.commit()
        except Exception as e:
            logger.error(f"Error deleting chat history: {str(e)}")
            raise
    
    def cleanup_expired(self):
        """Clean up expired chat histories"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                expiry_time = (datetime.now() - timedelta(hours=self.ttl_hours)).isoformat()
                
                cursor.execute(f'''
                    DELETE FROM {self.table_name}
                    WHERE last_updated < ?
                ''', (expiry_time,))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Error cleaning up expired chat histories: {str(e)}")
            raise
