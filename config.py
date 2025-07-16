
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
    CHAT_HISTORY_TABLE = os.getenv("CHAT_HISTORY_TABLE", "chat_history")
    CONTEXT_TABLE = os.getenv("CONTEXT_TABLE", "chat_context")
    # Memory TTL in hours
    memory_ttl_str = os.getenv("MEMORY_TTL", "24")
    MEMORY_TTL = int(memory_ttl_str.split('#')[0].strip())  # Extract number before any comments
    # Chat history settings
    MAX_HISTORY_MESSAGES = 50  # Maximum number of messages to keep in history
    CONTEXT_WINDOW_MESSAGES = 10  # Number of previous messages to include for context
    SUMMARY_TRIGGER_LENGTH = 5  # Number of messages after which to generate a summary
    
    # File Upload Configuration
    MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB
    SUPPORTED_FORMATS = [
        "pdf", "docx", "txt", "csv", "xlsx", "pptx", 
        "json", "html", "xml", "doc", "xls", "ppt", "md"
    ]
    
    # Search Tool Configuration
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")  # Custom Search Engine ID
    DUCKDUCKGO_RESULTS_COUNT = 5
    WIKIPEDIA_RESULTS_COUNT = 3
    
    # UI Configuration
    PAGE_TITLE = "🤖 Agentic RAG Assistant"
    PAGE_ICON = "🤖"
    LAYOUT = "wide"
    
    # REPL Configuration
    REPL_ENABLED = True
    SUPPORTED_REPL_LANGUAGES = [
        "python",
        "javascript",
        "typescript"
    ]
    REPL_TIMEOUT_SECONDS = 30
    REPL_MAX_OUTPUT_LENGTH = 5000  # Maximum characters of output to display
    # Security settings for REPL
    REPL_BLOCKED_MODULES = [
        "os", "sys", "subprocess", "pathlib",  # System access
        "socket", "requests", "urllib",  # Network access
        "shutil", "glob"  # File operations
    ]
    REPL_ALLOWED_PACKAGES = [
        # Python data science & utilities
        "numpy", "pandas", "matplotlib", "seaborn",
        "sklearn", "scipy", "math", "random", "datetime",
        # JavaScript/TypeScript utilities
        "lodash", "moment", "axios", "react", "vue"
    ]
    
    @classmethod
    def validate_openai_key(cls):
        """Validate OpenAI API key is set"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        return True
