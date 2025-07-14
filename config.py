
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings for the Agentic RAG application"""
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_BASE_URL = "https://api.openai.com/v1"  # Default OpenAI API endpoint
    OPENAI_MODEL = "gpt-4-1106-preview"
    OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
    
    # Langchain Configuration
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "agentic-rag-app")
    
    # Vector Database Configuration
    VECTOR_DB_PATH = "./vector_store"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    
    # Memory Configuration
    MEMORY_TYPE = os.getenv("MEMORY_TYPE", "sqlite")
    MEMORY_PATH = os.getenv("MEMORY_PATH", "./data/memory.db")
    MEMORY_TABLE = os.getenv("MEMORY_TABLE", "chat_memory")
    # Memory TTL in hours
    memory_ttl_str = os.getenv("MEMORY_TTL", "24")
    MEMORY_TTL = int(memory_ttl_str.split('#')[0].strip())  # Extract number before any comments
    
    # File Upload Configuration
    MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB
    SUPPORTED_FORMATS = [
        "pdf", "docx", "txt", "csv", "xlsx", "pptx", 
        "json", "html", "xml", "doc", "xls", "ppt", "md"
    ]
    
    # UI Configuration
    PAGE_TITLE = "🤖 Agentic RAG Assistant"
    PAGE_ICON = "🤖"
    LAYOUT = "wide"
    
    @classmethod
    def validate_openai_key(cls):
        """Validate OpenAI API key is set"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        return True
