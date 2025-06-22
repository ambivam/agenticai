
import json
from typing import Dict, List, Any, Optional, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessageGraph
import logging

logger = logging.getLogger(__name__)

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

class AgenticWorkflow:
    """Agentic workflow for RAG using Langgraph"""
    
    def __init__(self, openai_api_key: str, vector_store_manager):
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model="gpt-4-1106-preview",
            temperature=0.1,
            openai_api_base="https://apps.abacus.ai/v1"
        )
        self.vector_store_manager = vector_store_manager
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
        try:
            logger.info("Analyzing query...")
            
            analysis_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert query analyzer. Analyze the user's query and provide:
                1. Query type (factual, analytical, comparative, creative, etc.)
                2. Key concepts and entities
                3. Search strategy recommendations
                4. Complexity level (simple, moderate, complex)
                5. Expected answer type (short, detailed, list, explanation, etc.)
                
                Respond in JSON format with these fields:
                - query_type
                - key_concepts
                - entities
                - search_strategy
                - complexity
                - expected_answer_type
                - search_keywords
                """),
                ("human", "Query: {query}")
            ])
            
            response = self.llm.invoke(
                analysis_prompt.format_messages(query=state["query"])
            )
            
            try:
                analysis_results = json.loads(response.content)
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                analysis_results = {
                    "query_type": "general",
                    "key_concepts": [state["query"]],
                    "entities": [],
                    "search_strategy": "semantic_search",
                    "complexity": "moderate",
                    "expected_answer_type": "detailed",
                    "search_keywords": state["query"].split()
                }
            
            state["analysis_results"] = analysis_results
            state["current_step"] = "query_analysis_complete"
            
            logger.info(f"Query analysis complete: {analysis_results.get('query_type', 'unknown')}")
            return state
            
        except Exception as e:
            logger.error(f"Error in query analysis: {str(e)}")
            state["error"] = f"Query analysis failed: {str(e)}"
            return state
    
    def _search_documents(self, state: AgentState) -> AgentState:
        """Search for relevant documents using vector similarity"""
        try:
            logger.info("Searching documents...")
            
            query = state["query"]
            analysis = state.get("analysis_results", {})
            
            # Determine search parameters based on analysis
            k = 8 if analysis.get("complexity") == "complex" else 5
            
            # Perform similarity search
            search_results = self.vector_store_manager.similarity_search(
                query=query,
                k=k,
                score_threshold=0.1
            )
            
            # Format search results
            formatted_results = []
            for i, (doc, score) in enumerate(search_results):
                result = {
                    "id": i,
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": float(score),
                    "filename": doc.metadata.get("filename", "Unknown"),
                    "chunk_id": doc.metadata.get("chunk_id", 0)
                }
                formatted_results.append(result)
            
            state["search_results"] = formatted_results
            state["current_step"] = "document_search_complete"
            
            logger.info(f"Document search complete: {len(formatted_results)} results found")
            return state
            
        except Exception as e:
            logger.error(f"Error in document search: {str(e)}")
            state["error"] = f"Document search failed: {str(e)}"
            return state
    
    def _synthesize_context(self, state: AgentState) -> AgentState:
        """Synthesize context from search results"""
        try:
            logger.info("Synthesizing context...")
            
            search_results = state.get("search_results", [])
            analysis = state.get("analysis_results", {})
            
            if not search_results:
                state["error"] = "No search results to synthesize"
                return state
            
            # Create context synthesis prompt
            synthesis_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert information synthesizer. Given search results and query analysis, 
                create a comprehensive context summary that:
                1. Identifies key themes and patterns
                2. Highlights relevant facts and details
                3. Notes any conflicting information
                4. Organizes information logically
                5. Maintains source attribution
                
                Focus on information most relevant to answering the user's query.
                """),
                ("human", """
                Query: {query}
                Query Analysis: {analysis}
                
                Search Results:
                {search_results}
                
                Please synthesize this information into a coherent context summary.
                """)
            ])
            
            # Format search results for synthesis
            results_text = ""
            for i, result in enumerate(search_results):
                results_text += f"\n--- Result {i+1} (Score: {result['similarity_score']:.3f}) ---\n"
                results_text += f"Source: {result['filename']}\n"
                results_text += f"Content: {result['content']}\n"
            
            response = self.llm.invoke(
                synthesis_prompt.format_messages(
                    query=state["query"],
                    analysis=json.dumps(analysis, indent=2),
                    search_results=results_text
                )
            )
            
            # Store synthesis results
            state["analysis_results"]["context_synthesis"] = response.content
            state["current_step"] = "context_synthesis_complete"
            
            logger.info("Context synthesis complete")
            return state
            
        except Exception as e:
            logger.error(f"Error in context synthesis: {str(e)}")
            state["error"] = f"Context synthesis failed: {str(e)}"
            return state
    
    def _generate_response(self, state: AgentState) -> AgentState:
        """Generate the final response"""
        try:
            logger.info("Generating response...")
            
            query = state["query"]
            analysis = state.get("analysis_results", {})
            search_results = state.get("search_results", [])
            context_synthesis = analysis.get("context_synthesis", "")
            
            # Create response generation prompt
            response_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert AI assistant providing detailed, accurate responses based on provided context.

                Guidelines:
                1. Answer the user's question comprehensively and accurately
                2. Use information from the provided context and search results
                3. Maintain a helpful and professional tone
                4. Cite sources when making specific claims
                5. If information is incomplete, acknowledge limitations
                6. Structure your response clearly with appropriate formatting
                7. Provide actionable insights when possible
                
                Always base your response on the provided context and search results.
                """),
                ("human", """
                User Query: {query}
                
                Query Analysis: {analysis}
                
                Context Synthesis: {context_synthesis}
                
                Search Results:
                {search_results}
                
                Please provide a comprehensive response to the user's query.
                """)
            ])
            
            # Format search results
            results_text = ""
            sources = []
            for i, result in enumerate(search_results):
                results_text += f"\n--- Source {i+1} ---\n"
                results_text += f"File: {result['filename']}\n"
                results_text += f"Relevance Score: {result['similarity_score']:.3f}\n"
                results_text += f"Content: {result['content'][:500]}...\n"
                
                # Collect source information
                sources.append({
                    "id": i + 1,
                    "filename": result['filename'],
                    "content": result['content'][:200] + "..." if len(result['content']) > 200 else result['content'],
                    "similarity_score": result['similarity_score'],
                    "metadata": result['metadata']
                })
            
            response = self.llm.invoke(
                response_prompt.format_messages(
                    query=query,
                    analysis=json.dumps(analysis, indent=2),
                    context_synthesis=context_synthesis,
                    search_results=results_text
                )
            )
            
            state["response"] = response.content
            state["sources"] = sources
            state["current_step"] = "response_generation_complete"
            
            logger.info("Response generation complete")
            return state
            
        except Exception as e:
            logger.error(f"Error in response generation: {str(e)}")
            state["error"] = f"Response generation failed: {str(e)}"
            return state
    
    def _quality_check(self, state: AgentState) -> AgentState:
        """Perform quality check on the generated response"""
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
            overlap = len(query_keywords.intersection(response_keywords))
            
            if overlap / len(query_keywords) < 0.3:
                quality_issues.append("Response might not fully address the query")
            
            # Store quality check results
            state["analysis_results"]["quality_check"] = {
                "issues": quality_issues,
                "response_length": len(response),
                "keyword_overlap": overlap / len(query_keywords) if query_keywords else 0
            }
            
            state["current_step"] = "complete"
            
            logger.info(f"Quality check complete: {len(quality_issues)} issues found")
            return state
            
        except Exception as e:
            logger.error(f"Error in quality check: {str(e)}")
            state["error"] = f"Quality check failed: {str(e)}"
            return state
    
    def run_workflow(self, query: str, chat_history: List[BaseMessage] = None) -> Dict[str, Any]:
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
                error=None
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
                "success": final_state.get("error") is None
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
                "success": False
            }
