import sqlite3
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from config import Config

logger = logging.getLogger(__name__)

class UserProfile:
    def __init__(self, db_path: str = Config.MEMORY_PATH):
        """Initialize user profile manager"""
        self.db_path = db_path
        self.table_name = "user_profiles"
        self._init_db()
    
    def _init_db(self):
        """Initialize the database and create table if it doesn't exist"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f'''
                    CREATE TABLE IF NOT EXISTS {self.table_name} (
                        session_id TEXT PRIMARY KEY,
                        profile_data TEXT NOT NULL,
                        last_updated TEXT NOT NULL
                    )
                ''')
                conn.commit()
        except Exception as e:
            logger.error(f"Error initializing user profile database: {str(e)}")
            raise
    
    def get_profile(self, session_id: str) -> Dict[str, Any]:
        """Get user profile for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(f'''
                    SELECT profile_data
                    FROM {self.table_name}
                    WHERE session_id = ?
                ''', (session_id,))
                
                result = cursor.fetchone()
                if result:
                    return json.loads(result[0])
                return {}
        except Exception as e:
            logger.error(f"Error retrieving user profile: {str(e)}")
            return {}
    
    def update_profile(self, session_id: str, profile_data: Dict[str, Any]):
        """Update user profile for a session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                current_time = datetime.now().isoformat()
                
                # Convert profile data to JSON string
                profile_json = json.dumps(profile_data)
                
                # Insert or update profile
                cursor.execute(f'''
                    INSERT OR REPLACE INTO {self.table_name}
                    (session_id, profile_data, last_updated)
                    VALUES (?, ?, ?)
                ''', (session_id, profile_json, current_time))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating user profile: {str(e)}")
            raise
    
    def extract_user_info(self, message: str) -> Optional[Dict[str, Any]]:
        """Extract user information from a message"""
        # This is a simple implementation - in production you'd want to use
        # a more sophisticated NLP approach
        info = {}
        
        # Look for name mentions
        name_indicators = [
            "my name is",
            "i am",
            "i'm",
            "call me",
        ]
        
        message = message.lower()
        for indicator in name_indicators:
            if indicator in message:
                # Get the text after the indicator
                start_idx = message.index(indicator) + len(indicator)
                # Get the first word after the indicator
                name = message[start_idx:].strip().split()[0]
                if name:
                    info["name"] = name.title()
                    break
        
        return info if info else None
