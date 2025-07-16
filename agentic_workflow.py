import json
import logging
import uuid
from typing import List, Dict, Any, Optional, TypedDict
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from config import Config
from search_tools import SearchTools
from chat_memory import ChatMemory
from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessageGraph
import numpy as np

logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

class AgentState(TypedDict, total=False):
    """State for the agentic workflow"""
    # Required fields
    query: str
    search_sources: List[str]
    current_step: str
    session_id: str
    
    # Optional fields
    is_follow_up: bool
    chat_history: List[BaseMessage]
    search_results: List[Dict[str, Any]]
    analysis_results: Dict[str, Any]
    response: str
    sources: List[Dict[str, Any]]
    error: Optional[str]
    context: Optional[Dict[str, Any]]
    conversation_context: Optional[str]
    recent_messages: List[BaseMessage]
    suggested_follow_ups: List[str]

class AgenticWorkflow:
    """Agentic workflow for RAG using Langgraph"""
    
    def __init__(self, openai_api_key: str):
        """Initialize workflow with OpenAI API key"""
        self.config = Config
        self.llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            temperature=0.3,
            api_key=openai_api_key
        )
        self.memory = ChatMemory()
        self.current_session_id = None
        self.search_tools = SearchTools()
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the agentic workflow graph"""
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("query_analysis", self._analyze_query)
        workflow.add_node("document_search", self._search_documents)
        workflow.add_node("context_synthesis", self._synthesize_context)
        workflow.add_node("response_generation", self._generate_response)
        workflow.add_node("quality_check", self._quality_check)
        
        # Add edges
        workflow.set_entry_point("query_analysis")
        workflow.add_edge("query_analysis", "document_search")
        workflow.add_edge("document_search", "context_synthesis")
        workflow.add_edge("context_synthesis", "response_generation")
        workflow.add_edge("response_generation", "quality_check")
        workflow.add_edge("quality_check", END)
        
        return workflow.compile()
    
    def _analyze_query(self, state: AgentState) -> AgentState:
        """Analyze the user query to understand intent and plan search strategy"""
        logger.info("Analyzing query...")
        
        try:
            query = state["query"]
            
            # Create analysis prompt
            analysis_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert query analyzer. Analyze the user's query and provide:
                1. Query type (factual, analytical, comparative, creative, real-time, code_generation, etc.)
                2. Key concepts and entities
                3. Search strategy recommendations
                4. Complexity level (simple, moderate, complex)
                5. Expected answer type (short, detailed, list, explanation, code, etc.)
                6. Search sources to use (local_docs, wikipedia, web_search, google, or combinations)
                7. For code generation requests, specify:
                   - programming_language (python, javascript, typescript, java, etc.)
                   - code_type (algorithm, data_structure, utility, class, etc.)
                   - features_needed (list of required features/functionality)
                
                Special Instructions:
                - For code generation queries (e.g., 'write a program for...', 'implement...in python', etc.),
                  set query_type to 'code_generation' and use ['web_search'] as sources
                - For real-time information, use ["google", "web_search"] as sources
                - For historical or general knowledge, use ["wikipedia", "web_search"]
                - For project-specific or technical questions, use ["local_docs"]
                
                Respond in JSON format with these fields:
                - query_type
                - key_concepts
                - entities
                - search_strategy
                - complexity
                - expected_answer_type
                - search_keywords
                - search_sources: list of sources to search
                - code_generation_info (only if query_type is 'code_generation'):
                  - programming_language
                  - code_type
                  - features_needed
                """),
                ("human", "Query: {query}")
            ])
            
            # Get analysis results
            analysis = self.llm.invoke(
                analysis_prompt.format_messages(
                    query=query
                )
            )
            
            # Parse JSON response
            try:
                analysis_results = json.loads(analysis.content)
            except json.JSONDecodeError:
                logger.error("Failed to parse analysis results as JSON")
                analysis_results = {
                    "query_type": "unknown",
                    "key_concepts": [],
                    "entities": [],
                    "search_strategy": "default",
                    "complexity": "moderate",
                    "expected_answer_type": "detailed",
                    "search_keywords": [query],
                    "search_sources": ["google", "web_search"]
                }
            
            # Update state
            state["analysis_results"] = analysis_results
            state["current_step"] = "query_analysis"
            
            logger.info("Query analysis complete")
            return state
            
        except Exception as e:
            logger.error(f"Error in query analysis: {str(e)}")
            state["error"] = f"Query analysis failed: {str(e)}"
            return state
    
    def _search_documents(self, state: AgentState) -> AgentState:
        """Search for relevant documents using web search APIs"""
        logger.info("Searching documents...")
        
        try:
            # Get search configuration
            query = state["query"]
            search_sources = state.get("search_sources", [])
            if not search_sources:
                search_sources = ["web_search"]
            
            search_results = []
            
            # Search using configured sources
            for source in search_sources:
                try:
                    if source == "web_search":
                        # Search using DuckDuckGo
                        web_results = self.search_tools.search_duckduckgo(query, max_results=5)
                        search_results.extend([
                            {
                                "content": result["body"],
                                "metadata": {
                                    "title": result.get("title", ""),
                                    "url": result.get("link", "")
                                },
                                "source": "web_search"
                            }
                            for result in web_results
                        ])
                    elif source == "google":
                        # Search using Google Custom Search
                        google_results = self.search_tools.search_google(query, max_results=5)
                        search_results.extend([
                            {
                                "content": result.get("snippet", ""),
                                "metadata": {
                                    "title": result.get("title", ""),
                                    "url": result.get("link", "")
                                },
                                "source": "google"
                            }
                            for result in google_results
                        ])
                    elif source == "wikipedia":
                        # Search Wikipedia
                        wiki_results = self.search_tools.search_wikipedia(query, max_results=3)
                        search_results.extend([
                            {
                                "content": result["content"],
                                "metadata": {
                                    "title": result["title"],
                                    "url": result.get("url", "")
                                },
                                "source": "wikipedia"
                            }
                            for result in wiki_results
                        ])
                except Exception as e:
                    logger.error(f"Error searching {source}: {str(e)}")
                    continue
            # Update state
            state["search_results"] = search_results
            state["current_step"] = "documents_searched"
            
            logger.info(f"Search complete: {len(search_results)} results found")
            return state
            
        except Exception as e:
            logger.error(f"Error in document search: {str(e)}")
            state["error"] = f"Document search failed: {str(e)}"
            return state
    
    def _synthesize_context(self, state: AgentState) -> AgentState:
        """Synthesize search results into a coherent context"""
        try:
            logger.info("Synthesizing context...")
            
            query = state["query"]
            analysis = state.get("analysis_results", {})
            search_results = state.get("search_results", [])
            
            # Create synthesis prompt
            synthesis_prompt = ChatPromptTemplate.from_messages([
                ("system", """
                You are an expert at synthesizing information from multiple sources.
                Your task is to analyze the search results and create a coherent context summary
                that will help in answering the user's query.
                
                Focus on:
                1. Key facts and concepts
                2. Relationships between different pieces of information
                3. Any contradictions or gaps in the information
                4. Relevance to the original query
                """),
                ("human", """
                Query: {query}
                Analysis: {analysis}
                
                Search Results:
                {context}
                
                Please provide a comprehensive response to the user's query.
                Include relevant source attributions in your response.
                """)
            ])
            
            # Format search results for synthesis
            results_text = ""
            for i, result in enumerate(search_results):
                results_text += f"\n--- Result {i+1} ---\n"
                results_text += f"Content: {result['content']}\n"
            
            # Generate synthesis
            synthesis = self.llm.invoke(
                synthesis_prompt.format_messages(
                    query=state["query"],
                    analysis=json.dumps(convert_numpy_types(analysis), indent=2),
                    context=results_text
                )
            )
            
            # Update state
            state["context_synthesis"] = synthesis.content
            state["current_step"] = "context_synthesis"
            
            logger.info("Context synthesis complete")
            return state
            
        except Exception as e:
            logger.error(f"Error in context synthesis: {str(e)}")
            state["error"] = f"Context synthesis failed: {str(e)}"
    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate the final response using the LLM"""
        try:
            logger.info("Generating response...")
            
            # Process search results into context
            context = ""
            sources = []
            for result in state.get("search_results", []):
                try:
                    content = result["content"]
                    metadata = result["metadata"]
                    source_type = result["source"]
                    
                    # Add to sources list
                    if source_type == "local_docs":
                        source = {
                            "type": source_type,
                            "filename": metadata.get('filename', 'Unknown'),
                            "content": content,
                            "similarity_score": result.get('score', 0),
                            "source_name": "Local Document"
                        }
                        context += f"\nFrom document '{metadata.get('filename', 'Unknown')}': {content}\n"
                    else:
                        source_display_name = {
                            'wikipedia': 'Wikipedia',
                            'web_search': 'DuckDuckGo',
                            'google': 'Google'
                        }.get(source_type, source_type.title())
                        source = {
                            "type": source_type,
                            "title": metadata.get('title', 'Unknown'),
                            "url": metadata.get('url', ''),
                            "content": content,  # Store as content
                            "snippet": content,  # Also store as snippet for backward compatibility
                            "source_name": source_display_name
                        }
                        context += f"\nFrom {source_display_name} '{metadata.get('title', 'Unknown')}': {content}\n"
                    sources.append(source)
                except Exception as e:
                    logger.error(f"Error processing search result: {str(e)}")
                    continue
            
            # Create response prompt
            analysis_results = state.get("analysis_results", {})
            if analysis_results.get("query_type") == "code_generation":
                code_info = analysis_results.get("code_generation_info", {})
                system_prompt = """You are an expert programmer. Generate well-documented, production-quality code based on the user's request.
                    
Previous Conversation and Code:
{chat_history}

Use the following guidelines:
1. Include clear comments and docstrings
2. Follow best practices and design patterns
3. Include error handling and input validation
4. Add usage examples in comments
5. Structure the code logically
6. Include any necessary imports

Language: {language}
Code Type: {code_type}
Required Features: {features}

If the user is asking for the same implementation in a different language, use the previous code's structure and features as a reference.

Format your response as follows:
1. Brief explanation of the implementation
2. Complete code with comments
3. Usage examples
4. Any important notes or considerations

Context from search results:
{context}
"""

                # Format chat history
                chat_history_text = ""
                if state.get("chat_history"):
                    for msg in state["chat_history"]:
                        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                        chat_history_text += f"{role}: {msg.content}\n"

                response_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{query}")
                ])
                messages = response_prompt.format_messages(
                    query=state["query"],
                    context=context,
                    chat_history=chat_history_text,
                    language=code_info.get("programming_language", "python"),
                    code_type=code_info.get("code_type", "unknown"),
                    features=", ".join(code_info.get("features_needed", []))
                )
            else:
                # Regular non-code query
                system_prompt = """You are a helpful assistant. Use the following information to answer the user's question:
                
Previous Conversation:
{chat_history}

Context from search:
{context}

Analysis:
{analysis}

Make sure to maintain context from the previous conversation when answering. If the user refers to previous questions or code, use that context in your response.
                """
                
                # Format chat history
                chat_history_text = ""
                if state.get("chat_history"):
                    for msg in state["chat_history"]:
                        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
                        chat_history_text += f"{role}: {msg.content}\n"
                
                response_prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{query}")
                ])
                
                messages = response_prompt.format_messages(
                    query=state["query"],
                    context=context,
                    chat_history=chat_history_text,
                    analysis=json.dumps(state.get("analysis_results", {}), indent=2)
                )
            
            # Generate response using LLM
            response = self.llm.invoke(messages)
            
            # Process response content
            response_content = response.content if hasattr(response, 'content') else str(response)
            state["response"] = response_content
            state["sources"] = sources
            state["current_step"] = "response_generated"
            
            # Save messages to chat memory
            session_id = state["session_id"]
            self.memory.save_message(session_id, HumanMessage(content=state["query"]))
            self.memory.save_message(session_id, AIMessage(content=response_content))
            
            # Generate follow-up suggestions
            if not state.get("is_follow_up"):
                follow_up_prompt = ChatPromptTemplate.from_messages([
                    ("system", "Based on the query and response, suggest 2-3 relevant follow-up questions.\n\nQuery: {query}\n\nResponse: {response}"),
                    ("human", "What follow-up questions would be relevant?")
                ])
                
                follow_up_response = self.llm.invoke(
                    follow_up_prompt.format_messages(
                        query=state["query"],
                        response=response_content
                    )
                )
                
                # Extract suggestions (one per line)
                suggestions = [q.strip() for q in follow_up_response.content.split('\n') if q.strip()]
                state["suggested_follow_ups"] = suggestions[:3]  # Limit to top 3
            
            return state
            
        except Exception as e:
            logger.error(f"Error in response generation: {str(e)}")
            state["error"] = f"Response generation failed: {str(e)}"
            return state

    def _quality_check(self, state: AgentState) -> AgentState:
        """Perform quality checks on the generated response"""
        try:
            logger.info("Performing quality check...")
            
            response = state.get("response", "")
            query = state["query"]
            
            # Simple quality checks
            quality_issues = []
            
            # Check response length
            if len(response) < 100:
                quality_issues.append("Response might be too short")
            
            # Check if response addresses the query
            query_keywords = set(query.lower().split())
            response_keywords = set(response.lower().split())
            
            # Calculate keyword overlap safely
            if query_keywords:
                overlap = len(query_keywords.intersection(response_keywords))
                overlap_ratio = overlap / len(query_keywords)
                if overlap_ratio < 0.3:
                    quality_issues.append("Response might not fully address the query")
            else:
                overlap = 0
                overlap_ratio = 1.0  # If no keywords, assume full overlap
            
            # Store quality check results
            state["analysis_results"]["quality_check"] = {
                "issues": quality_issues,
                "response_length": len(response),
                "keyword_overlap": overlap_ratio
            }
            
            state["current_step"] = "complete"
            
            logger.info(f"Quality check complete: {len(quality_issues)} issues found")
            return state
            
        except Exception as e:
            logger.error(f"Error in quality check: {str(e)}")
            state["error"] = f"Quality check failed: {str(e)}"
            return state

    def run_workflow(self, query: str, chat_history: List[Any] = None, search_sources: List[str] = None,
                    context: Dict[str, Any] = None, session_id: str = None, is_follow_up: bool = False) -> Dict[str, Any]:
        """Run the workflow with a query and search sources"""
        try:
            # Set or update session ID
            if session_id:
                self.current_session_id = session_id
            elif not self.current_session_id:
                self.current_session_id = str(uuid.uuid4())
            
            # Initialize state with required fields
            state: AgentState = {
                "query": query,
                "search_sources": search_sources if search_sources else ["web_search"],
                "current_step": "start",
                "session_id": self.current_session_id,
                "chat_history": chat_history if chat_history else [],
                "context": context if context else {},
                # Optional fields
                "is_follow_up": is_follow_up,
                "search_results": [],
                "analysis_results": {},
                "response": "",
                "sources": [],
                "error": None,
                "suggested_follow_ups": []
            }
            
            # Run workflow
            final_state = self.workflow.invoke(state)
            
            # Format results
            results = {
                "response": final_state.get("response", ""),
                "sources": final_state.get("sources", []),
                "analysis": final_state.get("analysis_results", {}),
                "search_results_count": len(final_state.get("search_results", [])),
                "current_step": final_state.get("current_step", "unknown"),
                "error": final_state.get("error"),
                "success": final_state.get("error") is None,
                "suggested_follow_ups": final_state.get("suggested_follow_ups", [])
            }
            
            logger.info(f"Workflow completed successfully: {results['current_step']}")
            return results
            
        except Exception as e:
            logger.error(f"Error running workflow: {str(e)}")
            return {
                "response": "I apologize, but I encountered an error while processing your query.",
                "sources": [],
                "analysis": {},
                "search_results_count": 0,
                "current_step": "error",
                "error": str(e),
                "success": False,
                "suggested_follow_ups": []
            }
