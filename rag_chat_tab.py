import streamlit as st
from typing import List, Dict, Any
import logging
import uuid
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from config import Config

logger = logging.getLogger(__name__)

class RagChatTab:
    def __init__(self):
        # Initialize in-memory vector store
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.from_texts(
            ["Initial RAG context"],
            self.embeddings,
            metadatas=[{"type": "system", "source": "initialization", "timestamp": datetime.now().isoformat()}]
        )
        self._setup_rag_graph()

    def _setup_rag_graph(self):
        # Define the RAG prompt template with chat history
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant that answers questions based on the provided knowledge base and chat history. 
            
Previous conversation context:
{chat_history}

Relevant information from knowledge base:
{context}

Use both the chat history and knowledge base information to provide a comprehensive and contextual response. 
When asked about specific files, focus on the content from those files if available. 
If referring to previous conversation, be explicit about it. 
If the information isn't sufficient, say so and explain what's missing.

Important: If asked about a specific file and you find content from that file, make sure to provide information 
from that file rather than making general assumptions."""),
            ("human", "{question}")
        ])

        # Define the workflow nodes
        def retrieve_context(state):
            question = state["question"]
            chat_history = state.get("chat_history", "")

            # Extract file query if present
            file_query = None
            if "." in question.lower():
                words = question.split()
                for word in words:
                    if "." in word:
                        file_query = word.strip("?.,!'\"")
                        break

            # Log initial search state
            logger.info(f"Processing query: '{question}' with file_query: '{file_query}'")

            # Get relevant documents from vector store
            docs = []
            if file_query:
                # Try exact match first
                docs = self.vector_store.similarity_search(
                    question,
                    k=5,
                    filter={"source": file_query}
                )
                logger.info(f"Exact match search found {len(docs)} documents")

                # If no exact matches, try case-insensitive match
                if not docs:
                    docs = self.vector_store.similarity_search(
                        question,
                        k=5,
                        filter={"source": lambda x: x.lower() == file_query.lower() if x else False}
                    )
                    logger.info(f"Case-insensitive match found {len(docs)} documents")

                # If still no matches, try partial match
                if not docs:
                    docs = self.vector_store.similarity_search(
                        question,
                        k=5,
                        filter={"source": lambda x: file_query.lower() in x.lower() if x else False}
                    )
                    logger.info(f"Partial match found {len(docs)} documents")

            # If no specific file matches or no file query, do a general search
            if not docs:
                search_text = f"{chat_history}\n{question}" if chat_history else question
                docs = self.vector_store.similarity_search(
                    search_text,
                    k=5
                )
                logger.info(f"General search found {len(docs)} documents")

            # Format context with document sources
            context_parts = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'Unknown source')
                content = doc.page_content
                # Add file-specific context if this is a file query
                if file_query and source.lower().endswith(file_query.lower()):
                    context_parts.insert(0, f"File {source}:\n{content}")
                else:
                    context_parts.append(f"Document {i} (from {source}):\n{content}")

            return {
                "context": "\n\n".join(context_parts),
                "question": question,
                "chat_history": chat_history,
                "docs": docs
            }

        def generate_response(inputs: Dict[str, Any]) -> Dict[str, Any]:
            try:
                # Process the inputs through the prompt chain
                chain = prompt | st.session_state.llm | StrOutputParser()
                response = chain.invoke(inputs)
                
                return {
                    "response": response,
                    "context": inputs["context"],
                    "question": inputs["question"],
                    "sources": [doc.metadata.get('source', 'Unknown') for doc in inputs["docs"]]
                }
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                return {
                    "response": f"I apologize, but I encountered an error: {str(e)}",
                    "context": inputs["context"],
                    "question": inputs["question"],
                    "sources": []
                }

        # Create the graph workflow
        workflow = Graph()
        workflow.add_node("retrieve_context", retrieve_context)
        workflow.add_node("generate_response", generate_response)
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", END)
        workflow.set_entry_point("retrieve_context")
        self.rag_graph = workflow.compile()

    def __init__(self):
        """Initialize RAG Chat tab"""
        try:
            # Initialize document tracking
            self.doc_count = 0
            self.document_sources = set()
            
            # Initialize embedding function
            self.embeddings = OpenAIEmbeddings(
                api_key=Config.OPENAI_API_KEY,
                model=Config.OPENAI_EMBEDDING_MODEL
            )
            logger.info("Successfully initialized embeddings")
            
            # Initialize LLM
            self.llm = ChatOpenAI(
                model=Config.OPENAI_MODEL,
                temperature=0.7,
                api_key=Config.OPENAI_API_KEY
            )
            logger.info("Successfully initialized LLM")
            
            # Initialize vector store with a dummy document (required by FAISS)
            try:
                self.vector_store = FAISS.from_texts(
                    texts=["Initialization document"],
                    embedding=self.embeddings,
                    metadatas=[{
                        "source": "initialization",
                        "source_lower": "initialization",
                        "file_type": "none",
                        "timestamp": datetime.now().isoformat(),
                        "chunk_id": str(uuid.uuid4())
                    }]
                )
                logger.info("Successfully initialized vector store with dummy document")
            except Exception as e:
                logger.error(f"Error initializing vector store: {str(e)}")
                raise
            
            # Initialize chat history
            if "rag_chat_history" not in st.session_state:
                st.session_state.rag_chat_history = []
            
            # Initialize RAG settings
            if "rag_top_k" not in st.session_state:
                st.session_state.rag_top_k = 3
            
            logger.info("Successfully initialized RAG Chat tab with doc_count=0")
        except Exception as e:
            logger.error(f"Error initializing RAG Chat tab: {str(e)}")
            raise  # Re-raise to ensure proper error handling

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """Add documents to the vector store"""
        try:
            if not texts:
                logger.warning("No texts provided to add_documents")
                return False

            if not metadatas:
                metadatas = [{}] * len(texts)

            if len(texts) != len(metadatas):
                logger.error("Number of texts and metadatas must match")
                return False

            try:
                # Create new vector store with the documents
                new_vectorstore = FAISS.from_texts(
                    texts=texts,
                    embedding=self.embeddings,
                    metadatas=metadatas
                )
                
                # If we have an initialization document, create fresh vector store
                if self.doc_count == 0:
                    self.vector_store = new_vectorstore
                else:
                    # Otherwise merge with existing vector store
                    self.vector_store.merge_from(new_vectorstore)
                
                # Update document sources
                for metadata in metadatas:
                    if source := metadata.get("source"):
                        if source != "initialization":
                            self.document_sources.add(source)
                
                # Update doc count (excluding initialization document)
                self.doc_count = len(self.document_sources)
                
                logger.info(f"Successfully added {len(texts)} documents to vector store")
                logger.info(f"Document sources: {self.document_sources}")
                logger.info(f"Updated doc_count: {self.doc_count}")
                return True

            except Exception as e:
                logger.error(f"Error adding documents to vector store: {str(e)}")
                return False

        except Exception as e:
            logger.error(f"Error in add_documents: {str(e)}")
            return False

    def format_chat_history(self, history: List[Dict[str, Any]], max_history: int = 3) -> str:
        """Format recent chat history into a string"""
        formatted_history = []
        # Take only the most recent conversations, excluding the current query
        recent_history = history[:-1][-max_history*2:] if history else []
        
        for msg in recent_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted_history.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted_history)

    def process_query(self, query: str, chat_history: List[Dict[str, Any]]):
        """Process a query through the RAG system with chat memory"""
        try:
            # Format recent chat history
            chat_history_text = self.format_chat_history(chat_history)
            logger.info(f"Formatted chat history: {chat_history_text}")

            # Extract file query if present
            file_query = None
            words = query.lower().split()
            available_sources = {src.lower() for src in self.document_sources if src != "initialization"}
            
            # First try to find exact filename matches
            for word in words:
                cleaned_word = word.strip("?.,!'\"()[]{}")
                if cleaned_word in available_sources:
                    file_query = cleaned_word
                    break
            
            # If no exact match, try looking for file extensions
            if not file_query:
                for word in words:
                    if "." in word:
                        cleaned_word = word.strip("?.,!'\"()[]{}")
                        # Check if any available source ends with this word
                        for source in available_sources:
                            if source.endswith(cleaned_word.lower()):
                                file_query = source
                                break
                        if file_query:
                            break

            # Log search parameters
            logger.info(f"Processing query: '{query}' with file_query: '{file_query}'")
            logger.info(f"Available sources: {available_sources}")

            # Get relevant documents from vector store
            docs = []
            if file_query:
                try:
                    # Use source_lower for case-insensitive matching
                    docs = self.vector_store.similarity_search(
                        query,
                        k=st.session_state.rag_top_k,
                        filter={"source_lower": file_query}
                    )
                    logger.info(f"File-specific search found {len(docs)} documents")
                except Exception as e:
                    logger.warning(f"File-specific search failed: {str(e)}")

            # If no specific file matches or no file query, do a general search
            if not docs:
                try:
                    search_text = f"{chat_history_text}\n{query}" if chat_history_text else query
                    docs = self.vector_store.similarity_search(
                        search_text,
                        k=st.session_state.rag_top_k
                    )
                    logger.info(f"General search found {len(docs)} documents")
                except Exception as e:
                    logger.error(f"General search failed: {str(e)}")
                    raise

            # Format context and collect sources
            context_parts = []
            sources = set()
            for doc in docs:
                if isinstance(doc, Document):
                    context_parts.append(doc.page_content)
                    source = doc.metadata.get("source", "Unknown")
                    if source != "initialization":
                        sources.add(source)

            context = "\n\n".join(context_parts)
            logger.info(f"Retrieved context length: {len(context)} characters")
            logger.info(f"Retrieved sources: {sources}")

            # Prepare chat messages
            messages = [
                SystemMessage(content=f"""You are a helpful AI assistant that answers questions based on the provided knowledge base and chat history.

Previous conversation context:
{chat_history_text}

Relevant information from knowledge base:
{context}

Use both the chat history and knowledge base information to provide a comprehensive and contextual response.
When asked about specific files, focus on the content from those files if available.
If referring to previous conversation, be explicit about it.
If the information isn't sufficient, say so and explain what's missing.

IMPORTANT: If asked about a specific file and you find content from that file, make sure to provide information
from that file rather than making general assumptions.

Available document sources: {', '.join(sorted(available_sources))}"""),
                HumanMessage(content=query)
            ]

            # Get response from LLM
            response = self.llm.invoke(messages)

            return {
                "response": response.content,
                "sources": sorted(list(sources)) if sources else []
            }

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "sources": []
            }

    def display_chat_interface(self):
        """Display the RAG chat interface in Streamlit"""
        st.title("🔍 RAG Chat")
        st.write("Chat with your knowledge base using RAG (Retrieval-Augmented Generation)")

        # RAG Settings in sidebar
        with st.sidebar:
            st.subheader("🎯 RAG Settings")
            st.session_state.rag_top_k = st.slider(
                "Number of documents to retrieve",
                min_value=1,
                max_value=5,
                value=3,
                help="Number of most relevant documents to consider for each query"
            )

        # Display available sources
        if self.document_sources:
            sources_list = [source for source in self.document_sources if source != "initialization"]
            if sources_list:
                st.sidebar.subheader("📚 Available Documents")
                for source in sorted(sources_list):
                    st.sidebar.markdown(f"- {source}")

        # Initialize chat history
        if "rag_chat_history" not in st.session_state:
            st.session_state.rag_chat_history = []

        # Display chat history
        for message in st.session_state.rag_chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant" and message.get("sources"):
                    with st.expander("📚 Source Documents"):
                        for source in message["sources"]:
                            st.markdown(f"- {source}")

        # Chat input
        if query := st.chat_input("Ask a question about your documents..."):
            with st.chat_message("user"):
                st.write(query)

            # Add user message to history
            st.session_state.rag_chat_history.append({
                "role": "user",
                "content": query,
                "timestamp": datetime.now().isoformat()
            })

            # Process query with chat history
            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge base..."):
                    try:
                        result = self.process_query(query, st.session_state.rag_chat_history)
                        st.write(result["response"])

                        # Show sources if available
                        if result.get("sources"):
                            with st.expander("📚 Source Documents"):
                                st.write("\n".join(result["sources"]))

                        # Add assistant response to history
                        st.session_state.rag_chat_history.append({
                            "role": "assistant",
                            "content": result["response"],
                            "sources": result.get("sources", []),
                            "timestamp": datetime.now().isoformat()
                        })
                    except Exception as e:
                        error_msg = f"Error processing query: {str(e)}"
                        st.error(error_msg)
                        logger.error(error_msg)

        # Clear chat button
        if st.session_state.rag_chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.rag_chat_history = []
                st.rerun()
