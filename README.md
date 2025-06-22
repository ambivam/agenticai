
# 🤖 Agentic RAG Assistant

A powerful Retrieval-Augmented Generation (RAG) application that combines Langchain, Langgraph, OpenAI GPT-4, FAISS vector database, and Streamlit to provide intelligent document analysis and question answering.

## ✨ Features

### 🔄 Agentic Workflows
- **Multi-step Reasoning**: Uses Langgraph for sophisticated query analysis and planning
- **Query Decomposition**: Breaks down complex questions into manageable parts
- **Context Synthesis**: Intelligently combines information from multiple sources
- **Quality Assurance**: Built-in response validation and quality checking

### 📚 Comprehensive Document Support
- **PDF Documents**: Full text extraction with page attribution
- **Microsoft Office**: Word (DOCX), Excel (XLSX), PowerPoint (PPTX)
- **Text Files**: Plain text, CSV, JSON
- **Web Formats**: HTML, XML
- **Intelligent Chunking**: Smart text segmentation for optimal retrieval

### 🧠 Advanced AI Capabilities
- **GPT-4 Powered**: Uses latest OpenAI models for superior understanding
- **Vector Search**: FAISS-powered similarity search for relevant context
- **Source Attribution**: Detailed citations with relevance scores
- **Conversational Memory**: Maintains chat history for context

### 🎨 Professional UI
- **Clean Interface**: Modern Streamlit-based design
- **Progress Tracking**: Real-time processing status
- **Interactive Exploration**: Expandable result sections
- **Responsive Design**: Works on desktop and mobile

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Required packages (see requirements.txt)

### Installation
1. Clone or download the project files
2. Install dependencies:
   ```bash
   pip install streamlit langchain langgraph openai faiss-cpu python-dotenv pypdf python-docx openpyxl python-pptx beautifulsoup4 lxml pandas
   ```

3. Set up environment variables:
   ```bash
   # Create .env file
   OPENAI_API_KEY=your_api_key_here
   ```

### Running the Application
```bash
# Navigate to project directory
cd agentic_rag_app

# Run Streamlit app
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## 📖 Usage Guide

### 1. Document Upload
- Navigate to the "📁 Document Upload" tab
- Select one or more files from supported formats
- Click "🚀 Process Documents" to add them to the knowledge base
- Monitor progress and view processing statistics

### 2. Asking Questions
- Go to the "🔍 Query Interface" tab
- Enter your question in the text area
- Choose search depth (Standard/Deep/Comprehensive)
- Click "🔍 Ask Question" to get intelligent responses

### 3. Knowledge Base Management
- Use the "📊 Knowledge Base" tab to:
  - View document statistics
  - Search through uploaded content
  - Monitor storage usage

### 4. Configuration
- Access the "⚙️ Settings" tab for:
  - System configuration viewing
  - Knowledge base management
  - Application reset options

## 🏗️ Architecture

### Core Components

1. **Document Processor** (`document_processor.py`)
   - Handles multiple file formats
   - Intelligent text chunking
   - Metadata extraction

2. **Vector Store Manager** (`vector_store.py`)
   - FAISS vector database operations
   - Embedding generation and storage
   - Similarity search functionality

3. **Agentic Workflow** (`agentic_workflow.py`)
   - Langgraph-powered agent system
   - Multi-step reasoning pipeline
   - Query analysis and response generation

4. **Streamlit Interface** (`app.py`)
   - User interface and interaction
   - File upload handling
   - Result visualization

### Workflow Process

```
User Query → Query Analysis → Document Search → Context Synthesis → Response Generation → Quality Check → Final Response
```

## 🔧 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key
- `LANGCHAIN_API_KEY`: Langchain API key (optional)
- `LANGCHAIN_TRACING_V2`: Enable tracing (optional)

### Customizable Settings
- **Chunk Size**: Default 1000 tokens
- **Chunk Overlap**: Default 200 tokens
- **Max File Size**: Default 200MB
- **Vector DB Path**: Default `./vector_store`

## 📁 File Structure

```
agentic_rag_app/
├── app.py                    # Main Streamlit application
├── config.py                 # Configuration settings
├── document_processor.py     # Document handling
├── vector_store.py          # Vector database operations
├── agentic_workflow.py      # Langgraph workflow
├── utils.py                 # Utility functions
├── requirements.txt         # Python dependencies
├── .env.example            # Environment template
└── README.md               # This file
```

## 🎯 Supported File Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| PDF | `.pdf` | Portable Document Format |
| Word | `.docx`, `.doc` | Microsoft Word documents |
| Excel | `.xlsx`, `.xls` | Microsoft Excel spreadsheets |
| PowerPoint | `.pptx`, `.ppt` | Microsoft PowerPoint presentations |
| Text | `.txt` | Plain text files |
| CSV | `.csv` | Comma-separated values |
| JSON | `.json` | JavaScript Object Notation |
| HTML | `.html` | HyperText Markup Language |
| XML | `.xml` | eXtensible Markup Language |

## 🔍 Example Queries

### Simple Factual Questions
- "What is the main topic of the uploaded documents?"
- "Who are the key people mentioned in the presentations?"

### Analytical Questions
- "Compare the financial performance across different quarters"
- "What are the main risks identified in the reports?"

### Complex Reasoning
- "Based on the uploaded strategy documents, what are the recommended next steps?"
- "How do the customer feedback patterns relate to product development priorities?"

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all required packages are installed
2. **API Key Issues**: Verify OpenAI API key is correctly set
3. **File Processing Errors**: Check file format and size limits
4. **Memory Issues**: For large documents, consider reducing chunk size

### Performance Tips

- **Optimal Chunk Size**: Balance between context and performance
- **File Size**: Smaller files process faster
- **Query Specificity**: More specific questions yield better results

## 🤝 Contributing

This is a complete, production-ready application. For improvements:
1. Follow the existing code structure
2. Add comprehensive error handling
3. Include unit tests for new features
4. Update documentation

## 📄 License

This project is provided as-is for educational and practical use.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review error messages in the Streamlit interface
3. Ensure all dependencies are properly installed

---

**Built with ❤️ using Streamlit, Langchain, Langgraph, OpenAI, and FAISS**
