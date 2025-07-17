import json
from typing import Dict, List, Any, Optional, TypedDict
from langchain_openai import ChatOpenAI
from config import Config
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessageGraph
from search_tools import SearchTools
import logging
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

class AgentState(TypedDict):
    """State for the agentic workflow"""
    query: str
    chat_history: List[BaseMessage]
    search_results: List[Dict[str, Any]]
    analysis_results: Dict[str, Any]
    response: str
    sources: List[Dict[str, Any]]
    current_step: str
    error: Optional[str]
    context: Optional[Dict[str, Any]]
    suggested_follow_ups: List[str]

class AgenticWorkflow:
    """Agentic workflow for RAG using Langgraph"""
    
    def __init__(self, openai_api_key: str, vector_store_manager):
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model=Config.OPENAI_MODEL,
            temperature=0.1,
            base_url=Config.OPENAI_BASE_URL
        )
        self.vector_store_manager = vector_store_manager
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
                1. Query type (factual, analytical, comparative, creative, real-time, etc.)
                2. Key concepts and entities
                3. Search strategy recommendations
                4. Complexity level (simple, moderate, complex)
                5. Expected answer type (short, detailed, list, explanation, etc.)
                6. Search sources to use (local_docs, wikipedia, web_search, google, or combinations)
                
                Special Instructions:
                - For real-time information (weather, news, current events), use ["google", "web_search"] as sources
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
        """Search for relevant documents using vector similarity and web search"""
        try:
            logger.info("Searching documents and web sources...")
            
            # Get search keywords and sources
            search_keywords = state["analysis_results"].get("search_keywords", [])
            
            # Get user-selected search sources or use defaults
            context = state.get("context", {})
            search_sources = context.get("search_sources", None)
            if not search_sources:  # If no sources selected, use recommended ones from analysis
                search_sources = state["analysis_results"].get("search_sources", ["local_docs"])
                if not search_keywords:
                    search_keywords = [state["query"]]
            
            search_results = []
            
            # Vector store search
            if "local_docs" in search_sources:
                vector_results = self.vector_store_manager.similarity_search(state["query"])
                if vector_results:
                    for doc, score in vector_results:
                        # Get filename from metadata
                        metadata = doc.metadata or {}
                        filename = metadata.get('source') or metadata.get('filename', 'Unknown Document')
                        
                        result = {
                            "content": doc.page_content,
                            "metadata": {**metadata, 'filename': filename},
                            "source": "local_docs",
                            "score": float(score),
                            "source_name": f"Local Document: {filename}"
                        }
                        search_results.append(result)
            
            # Search Wikipedia if requested
            if "wikipedia" in search_sources:
                wiki_results = self.search_tools.search_wikipedia(state["query"])
                wiki_results = self.search_tools.search_wikipedia(query)
                for result in wiki_results:
                    search_results.append({
                        "content": result["summary"],
                        "metadata": {"title": result["title"], "url": result["url"]},
                        "source": "wikipedia",
                        "source_name": "Wikipedia"
                    })
                    wiki_results = self.search_tools.search_wikipedia(keyword)
                    for result in wiki_results:
                        search_results.append({
                            "content": result["summary"],
                            "metadata": {"title": result["title"], "url": result["url"]},
                            "source": "wikipedia",
                            "source_name": "Wikipedia"
                        })
            
            # Search web (DuckDuckGo) if requested
            if "web_search" in search_sources:
                for keyword in search_keywords:
                    web_results = self.search_tools.search_duckduckgo(keyword)
                    for result in web_results:
                        search_results.append({
                            "content": result["body"],
                            "metadata": {"title": result["title"], "url": result["url"]},
                            "source": "web_search",
                            "source_name": "DuckDuckGo"
                        })
            
            # Search Google if credentials are available
            if "google" in search_sources and Config.GOOGLE_API_KEY:
                for keyword in search_keywords:
                    google_results = self.search_tools.search_google(keyword)
                    for result in google_results:
                        search_results.append({
                            "content": result["snippet"],
                            "metadata": {"title": result["title"], "url": result["url"]},
                            "source": "google",
                            "source_name": "Google"
                        })
            
            # Update state
            state["search_results"] = search_results
            state["current_step"] = "document_search"
            
            logger.info(f"Found {len(search_results)} relevant results from various sources")
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
            return state

    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate the final response using the LLM"""
        try:
            # Format context from search results
            search_results = state.get("search_results", [])
            sources = []
            context = ""
            
            for result in search_results:
                try:
                    content = result["content"]
                    metadata = result["metadata"]
                    source_type = result["source"]
                    
                    if source_type == "local_docs":
                        # Get filename from metadata
                        filename = metadata.get('source') or metadata.get('filename', 'Unknown')
                        source = {
                            "type": "local_docs",
                            "metadata": metadata,  # Include full metadata
                            "content": content,
                            "similarity_score": result.get('score', 0),
                            "source_name": f"Local Document: {filename}"
                        }
                        context += f"\nFrom document '{filename}': {content}\n"
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
            response_prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a helpful assistant. Use the following context to answer the user's question. Include relevant information from the context, but do not make up information. If you cannot find relevant information in the context, say so.\n\nContext:\n{context}\n\nAnalysis:\n{analysis}"),
                ("human", "{query}")
            ])
            
            # Generate response
            response = self.llm.invoke(
                response_prompt.format_messages(
                    query=state["query"],
                    context=context,
                    analysis=json.dumps(state.get("analysis_results", {}), indent=2)
                )
            )
            
            # Extract response content
            response_content = response.content if hasattr(response, 'content') else str(response)
            state["response"] = response_content
            state["sources"] = sources
            state["current_step"] = "response_generated"
            
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

    def run_workflow(self, query: str, chat_history: List[BaseMessage] = None, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the complete agentic workflow"""
        try:
            logger.info(f"Running agentic workflow for query: {query[:50]}...")
            
            # Initialize state
            initial_state = AgentState(
                query=query,
                chat_history=chat_history or [],
                search_results=[],
                analysis_results={},
                response="",
                sources=[],
                current_step="initialized",
                error=None,
                context=context,
                suggested_follow_ups=[]
            )
            
            # Run workflow
            final_state = self.workflow.invoke(initial_state)
            
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
