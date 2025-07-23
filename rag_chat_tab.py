import streamlit as st
from typing import Dict, Any, List
from langchain_core.vectorstores import VectorStore
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END, Graph
import uuid
from datetime import datetime
import logging

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
            
            # Check if the question is about a specific file
            file_query = None
            if "." in question.lower():
                # Extract potential file names from the question
                words = question.split()
                for word in words:
                    if "." in word:
                        file_query = word.strip("?.,!'\"").lower()
                        break
            
            # Get relevant documents from vector store
            docs = []
            if file_query:
                # First try to find documents specifically from that file
                docs = self.vector_store.similarity_search(
                    question,
                    k=10,  # Increase k to find more potential matches
                    filter={"source": lambda x: x.lower().endswith(file_query) if x else False}
                )
            
            # If no specific file matches or no file query, do a general search
            if not docs:
                search_text = f"{chat_history}\n{question}" if chat_history else question
                docs = self.vector_store.similarity_search(
                    search_text,
                    k=st.session_state.get('rag_top_k', 3)
                )
            
            # Format context with document sources
            context_parts = []
            for i, doc in enumerate(docs, 1):
                source = doc.metadata.get('source', 'Unknown source')
                content = doc.page_content
                # Add file-specific context if this is a file query
                if file_query and source.lower().endswith(file_query):
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

    def add_documents(self, texts: List[str], metadatas: List[Dict[str, Any]] = None):
        """Add documents to the vector store"""
        try:
            self.vector_store.add_texts(texts, metadatas=metadatas)
            return True
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")
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

    def process_query(self, query: str, chat_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process a query through the RAG graph with chat history context"""
        try:
            # Format chat history
            chat_history_text = self.format_chat_history(chat_history)
            
            # Get response from graph
            result = self.rag_graph.invoke({
                "question": query,
                "chat_history": chat_history_text
            })
            
            return result
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "context": "",
                "question": query,
                "chat_history": chat_history_text,
                "sources": []
            }

    def display_chat_interface(self):
        """Display the RAG chat interface in Streamlit"""
        st.header("🔍 RAG Chat")
        st.markdown("Chat with your knowledge base using RAG (Retrieval-Augmented Generation)")

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
            st.session_state.rag_show_sources = st.checkbox(
                "Show source documents",
                value=True,
                help="Display the source documents used for generating the response"
            )

        # Initialize chat history
        if "rag_chat_history" not in st.session_state:
            st.session_state.rag_chat_history = []

        # Display chat history
        for message in st.session_state.rag_chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant":
                    if st.session_state.rag_show_sources and message.get("sources"):
                        with st.expander("📚 Source Documents"):
                            for source in message["sources"]:
                                st.markdown(f"- {source}")
                    if "context" in message and st.session_state.rag_show_sources:
                        with st.expander("🔍 Retrieved Context"):
                            st.markdown(message["context"])

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
                    result = self.process_query(query, st.session_state.rag_chat_history)
                    st.write(result["response"])

                    # Add assistant response to history
                    st.session_state.rag_chat_history.append({
                        "role": "assistant",
                        "content": result["response"],
                        "context": result["context"],
                        "sources": result["sources"],
                        "timestamp": datetime.now().isoformat()
                    })

        # Clear chat button
        if st.session_state.rag_chat_history:
            if st.button("🗑️ Clear RAG Chat History"):
                st.session_state.rag_chat_history = []
                st.rerun()
