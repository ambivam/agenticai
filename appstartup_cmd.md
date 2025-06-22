# Agentic RAG Assistant - Command Prompt (CMD) Setup Guide

## Prerequisites
- Python 3.11 or higher
- pip (Python package installer)
- OpenAI API key

## Initial Setup

### 1. Clone the Repository
```cmd
git clone <repository-url>
cd agentic_rag_app
```

### 2. Create and Activate Virtual Environment
```cmd
:: Create virtual environment
python -m venv venv

:: Activate virtual environment
venv\Scripts\activate.bat
```

### 3. Install Dependencies
```cmd
pip install -r requirements.txt
```

### 4. Configure Environment Variables

1. Create a `.env` file:
```cmd
:: Create .env file
type nul > .env
```

2. Open the `.env` file in a text editor and add:
```
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agentic-rag-app
```

## Starting the Application

### 1. Ensure Virtual Environment is Activated
```cmd
:: Check if venv is activated (should show virtual environment path)
where python

:: If not activated, run:
venv\Scripts\activate.bat
```

### 2. Start Streamlit Application
```cmd
streamlit run app.py
```

The application will be available at:
- Local URL: http://localhost:8501
- Network URL: http://192.168.0.177:8501 (or similar, depending on your network)

## Stopping the Application

### Method 1: Using Command Prompt
```cmd
:: Find Streamlit process ID
tasklist | findstr "streamlit"

:: Stop Streamlit process (replace <PID> with actual Process ID)
taskkill /F /PID <PID>

:: Or stop all Streamlit processes
taskkill /F /IM "streamlit.exe"
```

### Method 2: Using Terminal
Press `Ctrl+C` in the terminal where Streamlit is running.

## Troubleshooting

### 1. Check OpenAI API Key
```cmd
:: View .env file content
type .env
```

### 2. Check Running Streamlit Processes
```cmd
:: List all running Streamlit processes
tasklist | findstr "streamlit"
```

### 3. Clear Vector Store
If you need to reset the vector store:
1. Stop the application
2. Delete the `vector_store` directory
```cmd
:: Remove vector store directory
rd /s /q vector_store
```

### 4. Restart Application After Configuration Changes
Always restart the Streamlit application after making changes to:
- `.env` file
- Python code files
- Configuration settings

```cmd
:: Stop existing instances
taskkill /F /IM "streamlit.exe"

:: Start fresh instance
streamlit run app.py
```

### 5. Environment Variables
```cmd
:: Set OpenAI API key temporarily (for current session)
set OPENAI_API_KEY=your_openai_api_key_here

:: View current environment variables
set | findstr "OPENAI"
```

## Common Issues and Solutions

### 1. Virtual Environment Not Activating
```cmd
:: Try alternative activation method
call venv\Scripts\activate.bat
```

### 2. Port Already in Use
```cmd
:: Find process using port 8501
netstat -ano | findstr :8501

:: Kill process using specific port (replace <PID> with actual Process ID)
taskkill /F /PID <PID>
```

### 3. Clean Restart
```cmd
:: Stop all Python and Streamlit processes
taskkill /F /IM "python.exe"
taskkill /F /IM "streamlit.exe"

:: Reactivate virtual environment
venv\Scripts\activate.bat

:: Start application
streamlit run app.py
```

## Application Features and Support

The application provides several interfaces:
1. Document Upload - Upload and process documents
2. Query Interface - Ask questions about uploaded documents
3. Knowledge Base - View and manage uploaded documents
4. Settings - Configure application settings

## Supported File Formats
- Documents: PDF, DOCX, TXT
- Spreadsheets: XLSX, CSV
- Presentations: PPTX
- Web: HTML, XML
- Data: JSON

## Models Used
- LLM: GPT-4 Turbo
- Embeddings: text-embedding-3-small
- Vector DB: FAISS (Local)

## Note
- Commands prefixed with `::` are comments in CMD
- Make sure to keep your OpenAI API key confidential
- Never commit the `.env` file to version control
- Use `%USERPROFILE%` instead of `~` for home directory in CMD
