import streamlit as st
from typing import Dict, Any, List
from langchain_core.vectorstores import VectorStore
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import END, Graph
import uuid
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ChatMemoryTab:
    def __init__(self):
        # Initialize memory store with OpenAI embeddings
        self.embeddings = OpenAIEmbeddings()
        self.memory_store = FAISS.from_texts(
            ["Initial memory context"], 
            self.embeddings,
            metadatas=[{
                "type": "system", 
                "source": "initialization",
                "timestamp": datetime.now().isoformat()
            }]
        )
        
        # Initialize the chat graph for conversation flow
        self._setup_chat_graph()
        
        # Ensure session state is initialized
        if 'memory_chat_history' not in st.session_state:
            st.session_state.memory_chat_history = []
        if 'messages' not in st.session_state:
            st.session_state.messages = []

    def _setup_chat_graph(self):
        # Define the chat prompt template with better context handling
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant with memory. Below is the relevant context from previous conversations and knowledge base:\n\n{context}\n\nUse this context to provide a more informed response. If the context is relevant, incorporate it naturally into your response. If not, you can rely on your general knowledge."),
            ("human", "{question}")
        ])

        # Define the workflow nodes
        def retrieve_context(state):
            question = state["question"]
            # Get relevant context from memory store
            memory_docs = self.memory_store.similarity_search(question, k=3)
            memory_context = "\n\n".join([
                f"Previous Conversation ({doc.metadata.get('timestamp', 'Unknown time')}):\n" +
                doc.page_content
                for doc in memory_docs
            ])
            
            return {
                "context": memory_context,
                "question": question
            }

        def generate_response(inputs: Dict[str, Any]) -> Dict[str, Any]:
            try:
                # Process the inputs through the prompt chain
                chain = prompt | st.session_state.llm | StrOutputParser()
                response = chain.invoke(inputs)
                
                # Store the conversation in memory with structured format
                conversation_text = (
                    f"User: {inputs['question']}\n" +
                    f"Assistant: {response}\n" +
                    f"Context Used: {inputs['context'][:200]}..."
                )
                
                self.memory_store.add_texts(
                    [conversation_text],
                    metadatas=[{
                        "type": "conversation",
                        "timestamp": datetime.now().isoformat(),
                        "question": inputs["question"],
                        "has_context": bool(inputs["context"].strip())
                    }]
                )
                
                return {
                    "response": response,
                    "context": inputs["context"],
                    "question": inputs["question"]
                }
            except Exception as e:
                logger.error(f"Error generating response: {str(e)}")
                return {
                    "response": f"I apologize, but I encountered an error: {str(e)}",
                    "context": inputs["context"],
                    "question": inputs["question"]
                }

        # Create the graph workflow
        workflow = Graph()
        
        # Add nodes to the graph
        workflow.add_node("retrieve_context", retrieve_context)
        workflow.add_node("generate_response", generate_response)
        
        # Add edges
        workflow.add_edge("retrieve_context", "generate_response")
        workflow.add_edge("generate_response", END)
        
        # Set the entry point
        workflow.set_entry_point("retrieve_context")
        
        # Compile the graph
        self.chat_graph = workflow.compile()

    def process_message(self, message: str) -> Dict[str, Any]:
        """Process a user message through the chat graph"""
        try:
            # Run the message through the graph
            result = self.chat_graph.invoke({"question": message})
            return result
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return {
                "response": f"An error occurred: {str(e)}",
                "context": "",
                "question": message
            }

    def display_chat_interface(self):
        """Display the chat memory interface in Streamlit"""
        st.header("💭 Chat Memory")
        st.markdown("Chat with memory-enhanced context awareness. The assistant remembers previous conversations and uses them to provide more informed responses.")

        # Settings in sidebar
        with st.sidebar:
            st.subheader("💡 Memory Settings")
            memory_depth = st.slider(
                "Memory Depth",
                min_value=1,
                max_value=5,
                value=3,
                help="Number of previous conversations to consider for context"
            )

        # Display chat history
        for message in st.session_state.memory_chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant" and "context" in message:
                    with st.expander("🧠 View Memory Context"):
                        context_lines = message["context"].split("\n")
                        for line in context_lines:
                            if line.strip():
                                st.markdown(f"- {line.strip()}")

        # Chat input
        if prompt := st.chat_input("Ask me anything..."):
            with st.chat_message("user"):
                st.write(prompt)

            # Add user message to chat history
            st.session_state.memory_chat_history.append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            })

            # Show thinking indicator
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Process the message
                    result = self.process_message(prompt)

                    # Display response
                    st.write(result["response"])

                    # Show context if available
                    if result["context"].strip():
                        with st.expander("🧠 Memory Used"):
                            context_lines = result["context"].split("\n")
                            for line in context_lines:
                                if line.strip():
                                    st.markdown(f"- {line.strip()}")

            # Add assistant response to chat history
            st.session_state.memory_chat_history.append({
                "role": "assistant",
                "content": result["response"],
                "context": result["context"],
                "timestamp": datetime.now().isoformat()
            })

        # Memory management buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.memory_chat_history and st.button("🗑️ Clear Chat History"):
                st.session_state.memory_chat_history = []
                st.rerun()
                
        with col2:
            if st.session_state.memory_chat_history and st.button("🔄 Reset Memory"):
                # Reinitialize the memory store
                self.memory_store = FAISS.from_texts(
                    ["Initial memory context"], 
                    OpenAIEmbeddings(),
                    metadatas=[{"type": "system", "timestamp": datetime.now().isoformat()}]
                )
                st.success("Memory has been reset!")
                st.rerun()
