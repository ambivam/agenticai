# Agentic RAG Assistant - Application Setup and Management Guide

## Prerequisites
- Python 3.11 or higher
- pip (Python package installer)
- OpenAI API key

## Initial Setup

### 1. Clone the Repository
```powershell
git clone <repository-url>
cd agentic_rag_app
```

### 2. Create and Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate

# If using Command Prompt (cmd)
.\venv\Scripts\activate.bat
```

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Configure Environment Variables

1. Create a `.env` file in the root directory:
```powershell
# Create .env file
New-Item -Path .env -Type File
```

2. Add the following content to `.env`:
```
OPENAI_API_KEY=your_openai_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agentic-rag-app
```

Replace `your_openai_api_key_here` with your actual OpenAI API key.

## Starting the Application

### 1. Ensure Virtual Environment is Activated
```powershell
# Check if venv is activated (should show virtual environment path)
Get-Command python | Format-List
```

If not activated, run:
```powershell
.\venv\Scripts\Activate
```

### 2. Start Streamlit Application
```powershell
streamlit run app.py
```

The application will be available at:
- Local URL: http://localhost:8501
- Network URL: http://192.168.0.177:8501 (or similar, depending on your network)

## Stopping the Application

### Method 1: Using PowerShell
```powershell
# Find and stop all Streamlit processes
Get-Process -Name streamlit | Stop-Process
```

### Method 2: Using Terminal
Press `Ctrl+C` in the terminal where Streamlit is running.

## Troubleshooting

### 1. Check OpenAI API Key
```powershell
# View .env file content (make sure it's properly configured)
Get-Content .env
```

### 2. Check Running Streamlit Processes
```powershell
# List all running Streamlit processes
Get-Process -Name streamlit
```

### 3. Clear Vector Store
If you need to reset the vector store:
1. Stop the application
2. Delete the `vector_store` directory
```powershell
Remove-Item -Recurse -Force .\vector_store
```

### 4. Restart Application After Configuration Changes
Always restart the Streamlit application after making changes to:
- `.env` file
- Python code files
- Configuration settings

```powershell
# Stop existing instances
Get-Process -Name streamlit | Stop-Process

# Start fresh instance
streamlit run app.py
```

## Application Features

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
Make sure to keep your OpenAI API key confidential and never commit the `.env` file to version control.
