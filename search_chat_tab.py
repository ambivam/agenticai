import streamlit as st
from typing import List, Dict, Any
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.messages import HumanMessage, AIMessage
from config import Config
from utils import display_error_message, display_success_message

class SearchChatTab:
    def __init__(self):
        self.search = SerpAPIWrapper(serpapi_api_key=Config.SERPAPI_API_KEY)
        
    def search_web(self, query: str) -> str:
        """
        Perform a web search using SERPAPI
        """
        try:
            search_results = self.search.run(query)
            return search_results
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            return ""

    def handle_chat(self, llm):
        """
        Handle the search chat interface
        """
        st.subheader("🔎 Search Chat")
        st.markdown("Ask questions and get answers from the web using SERPAPI")

        # Initialize chat history
        if 'search_chat_history' not in st.session_state:
            st.session_state.search_chat_history = []

        # Display chat history
        for message in st.session_state.search_chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("What would you like to know?"):
            # Add user message to chat history
            st.session_state.search_chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Perform web search
            with st.spinner("Searching the web..."):
                search_results = self.search_web(prompt)

            # Generate response using LLM
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    messages = [
                        HumanMessage(content=f"""Based on the following search results, please provide a comprehensive answer to the question: "{prompt}"

Search Results:
{search_results}

Please provide a well-structured response that directly answers the question using the search results. Include relevant information and cite sources when possible.""")
                    ]
                    response = llm.invoke(messages)
                    st.markdown(response.content)
                    st.session_state.search_chat_history.append(
                        {"role": "assistant", "content": response.content}
                    )

        # Add a clear chat button
        if st.session_state.search_chat_history:
            if st.button("Clear Chat History", key="clear_search_chat"):
                st.session_state.search_chat_history = []
                display_success_message("Chat history cleared!")
                st.experimental_rerun()
