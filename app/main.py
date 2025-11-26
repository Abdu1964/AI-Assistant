from .llm_handle.llm_models import (
    LLMInterface,
    OpenAIModel,
    get_llm_model,
    openai_embedding_model,
)
from .annotation_graph.annotated_graph import Graph
from .annotation_graph.schema_handler import SchemaHandler
from .rag.rag import RAG
from .rag.utils.web_search import SimpleWebSearch
from .prompts.conversation_handler import conversation_prompt
from .prompts.classifier_prompt import (
    classifier_prompt,
    answer_from_graph,
    main_classifier_prompt
)
from .summarizer import Graph_Summarizer
from .hypothesis_generation.hypothesis import HypothesisGeneration
from .storage.history_manager import HistoryManager
from .storage.mongo_storage import mongo_db_manager
from .socket_manager import emit_to_user
from .Galaxy_integration.galaxy import GalaxyHandler
import asyncio
import logging.handlers as loghandlers
from dotenv import load_dotenv
import traceback
import json
import os
from flask_socketio import emit
from typing import TypedDict, List, Annotated, Any, Dict, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
import operator
import logging
import google.generativeai as genai
from app.biogpt_agent.biogpt import BioGPTAgent


logger = logging.getLogger(__name__)
log_dir = "/AI-Assistant/logfiles"
log_file = os.path.join(log_dir, "Assistant.log")
logger.setLevel(logging.DEBUG)
loghandle = loghandlers.TimedRotatingFileHandler(
    filename="logfiles/Assistant.log",
    when="D",
    interval=1,
    backupCount=7,
    encoding="utf-8",
)
loghandle.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(loghandle)

load_dotenv()

# Updated classifier prompt for multi-agent selection
main_classifier_prompt = """
Classify this user query into one or more of the following agent types. Multiple agents can handle the same query if applicable.

Agent types:
- annotation_biological: Requests to find, retrieve, or explore specific biological entities and their relationships (e.g., "find gene BRCA1", "show transcripts for TP53", "what exons does IGF1 have")
- annotation_general: Requests for aggregate statistics, counts, or metadata about the database itself (e.g., "how many genes", "what types of variants", "database statistics")
- galaxy: Requests about Galaxy web tools, workflows, or Galaxy platform capabilities
- rag: General information requests, including queries about uploaded PDFs, web content, or document profiles

User query: {query}

Content summaries: {content_summaries}

{web_context}

Examples:
- "Find gene BRCA1 and tell me about its function" → annotation_biological, rag
- "What Galaxy tools can I use for RNA-seq?" → galaxy, rag
- "Show me genes related to diabetes from my uploaded PDF" → annotation_biological, rag

Respond ONLY with a comma-separated list of agent types that should handle this query.
If the query clearly relates to only one agent, return just that one.
"""


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_query: str
    user_id: str
    token: str
    query_types: List[str]  # Changed to list for multiple types
    response: Dict[str, Any]
    error: str
    content_ids: Optional[List[str]]
    graph_id: Optional[str]
    files: Optional[List[str]]
    pipeline_details: Dict[str, Any]
    # Agent-specific responses with source attribution
    annotation_response: Optional[Dict[str, Any]]
    rag_response: Optional[Dict[str, Any]]
    galaxy_response: Optional[Dict[str, Any]]
    content_retrieval_response: Optional[Dict[str, Any]]


class AiAssistance:

    def __init__(
        self,
        advanced_llm,
        basic_llm,
        schema_handler,
        qdrant_client=None,
        embedding_model=None,
    ) -> None:
        self.advanced_llm = advanced_llm
        self.basic_llm = basic_llm
        self.annotation_graph = Graph(advanced_llm, schema_handler)
        self.graph_summarizer = Graph_Summarizer(self.advanced_llm)
        self.rag = RAG(llm=advanced_llm, qdrant_client=qdrant_client)
        self.history = HistoryManager()
        self.store = mongo_db_manager
        self.hypothesis_generation = HypothesisGeneration(advanced_llm)
        self.galaxy_handler = GalaxyHandler(advanced_llm, qdrant_client, embedding_model)
        self.embedding_model = embedding_model
        self.biogpt = BioGPTAgent(llm=advanced_llm)

        logger.info(
            f"AiAssistance initialized with advanced_llm: {type(self.advanced_llm).__name__}"
        )
        logger.info(f"Galaxy handler initialized: {type(self.galaxy_handler).__name__}")

        # Initialize the LangGraph workflow
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile()

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow with parallel agent execution"""
        logger.info("Creating LangGraph workflow with parallel agent execution")

        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("classifier", self._classify_query)
        workflow.add_node("annotation_agent", self._annotation_agent)
        # workflow.add_node("hypothesis_agent", self._hypothesis_agent)
        workflow.add_node("rag_agent", self._rag_agent)
        workflow.add_node("galaxy_agent", self._galaxy_agent)
        workflow.add_node("content_retrieval_agent", self._content_retrieval_agent)
        workflow.add_node("biogpt_agent", self._biogpt_agent)
        workflow.add_node("aggregator", self._aggregate_responses)
        workflow.add_node("finalizer", self._finalize_response)

        # Define edges
        workflow.set_entry_point("classifier")

        # Conditional routing to multiple agents in parallel
        workflow.add_conditional_edges(
            "classifier",
            self._route_query,
            {
                "annotation_agent": "annotation_agent",
                # "hypothesis": "hypothesis_agent",
                "rag_agent": "rag_agent",
                "galaxy_agent": "galaxy_agent",
                "content_retrieval_agent": "content_retrieval_agent",
                "biogpt_agent": "biogpt_agent",
                "aggregator": "aggregator",
                "error": "finalizer",
            },
        )

        # All agents converge to aggregator
        workflow.add_edge("annotation_agent", "aggregator")
        workflow.add_edge("rag_agent", "aggregator")
        workflow.add_edge("galaxy_agent", "aggregator")
        workflow.add_edge("content_retrieval_agent", "aggregator")
        workflow.add_edge("biogpt_agent", "aggregator")
        # Aggregator flows to finalizer
        workflow.add_edge("aggregator", "finalizer")
        workflow.add_edge("finalizer", END)

        return workflow

    def get_content_summaries(self, user_id, content_ids=None):
        """Get summaries for all content types (PDF and web)"""
        content_summaries = []
        all_content = self.store.get_user_content_files(user_id)

        if content_ids:
            filtered_content = [
                content
                for content in all_content
                if content.get("content_id") in content_ids
            ]
        else:
            filtered_content = all_content

        for content in filtered_content:
            if content.get("content_type") == "pdf":
                content_summaries.append(
                    {
                        "content_id": content.get("content_id"),
                        "content_type": "pdf",
                        "filename": content.get("filename"),
                        "summary": content.get("summary") or "",
                    }
                )
            elif content.get("content_type") == "web":
                content_summaries.append(
                    {
                        "content_id": content.get("content_id"),
                        "content_type": "web",
                        "url": content.get("url"),
                        "title": content.get("title"),
                        "summary": content.get("summary") or "",
                    }
                )

        return content_summaries

    def _classify_query(self, state: AgentState) -> Dict[str, Any]:
        """Classify query and determine which agents to invoke (can be multiple)"""
        query = state["user_query"]
        user_id = state["user_id"]
        content_ids = state.get("content_ids")
        graph_id = state.get("graph_id")
        files = state.get("files")

        # Fetch content summaries
        content_summaries = self.get_content_summaries(user_id, content_ids)
        
        # Get web context
        web_urls = SimpleWebSearch().get_context_urls(query, num_results=2)
        web_context = f"Web context: {', '.join(web_urls)}" if web_urls else ""

        logger.info(f"Classifying query: {query}")
        
        classifier_prompt_text = main_classifier_prompt.format(
            query=query, 
            content_summaries=content_summaries, 
            web_context=web_context
        )

        response = self.advanced_llm.generate(classifier_prompt_text).lower()
        
        # Parse response to get multiple query types
        query_types = []
        # Clean up response and split by comma
        cleaned_response = response.replace("and", ",").replace("\n", ",")
        potential_types = [t.strip() for t in cleaned_response.split(",")]
        
        for qtype in potential_types:
            if "annotation_biological" in qtype or "annotation biological" in qtype:
                query_types.append("annotation_biological")
            elif "annotation_general" in qtype or "annotation general" in qtype:
                query_types.append("annotation_general")
            elif "galaxy" in qtype:
                query_types.append("galaxy")
            elif "rag" in qtype:
                query_types.append("rag")
        
        # Remove duplicates
        query_types = list(set(query_types))
        
        # If no types detected, default to rag
        if not query_types:
            query_types = ["rag"]

        logger.info(f"Query classified as: {query_types}")

        return {
            "query_types": query_types,
            "messages": [HumanMessage(content=f"Query classified as: {', '.join(query_types)}")],
        }

    def _route_query(self, state: AgentState) -> List[str]:
        """Route query to appropriate agents (can be multiple for parallel execution)"""
        query_types = state.get("query_types", ["rag"])
        content_ids = state.get("content_ids")
        files = state.get("files")
        graph_id = state.get("graph_id")
        
        routes = []

        # Add content retrieval agent if any content references exist
        if content_ids or files or graph_id:
            routes.append("content_retrieval_agent")

        # Map query types to agent routes
        type_to_agent = {
            "annotation_biological": "annotation_agent",
            "annotation_general": "annotation_agent",
            "galaxy": "galaxy_agent",
            "rag": "rag_agent",
            "biogpt": "biogpt_agent",
        }

        for qtype in query_types:
            agent = type_to_agent.get(qtype)
            if agent and agent not in routes:
                routes.append(agent)

        # Ensure we have at least one route
        if not routes:
            routes.append("rag_agent")

        # If we have multiple agents, they'll run in parallel before aggregation
        if len(routes) > 1:
            logger.info(f"Routing query to multiple agents in parallel: {routes}")
            # Return all routes - LangGraph will handle parallel execution
            return routes
        elif routes:
            logger.info(f"Routing query to single agent: {routes[0]}")
            return routes[0]
        else:
            # Fallback to aggregator if no specific agents
            return "aggregator"

    def _annotation_agent(self, state: AgentState) -> Dict[str, Any]:
        """Handle annotation-related queries"""
        query_types = state.get("query_types", [])
        query_type = next((qt for qt in query_types if "annotation" in qt), "annotation_biological")
        
        logger.info(
            f"Annotation agent processing query: {state['user_query']} for user: {state['user_id']}, type: {query_type}"
        )
        
        try:
            if query_type == "annotation_biological":
                emit_to_user(
                    user=state["user_id"], 
                    message="Processing your biological query..."
                )
            elif query_type == "annotation_general":
                emit_to_user(
                    user=state["user_id"], 
                    message="Analyzing database information..."
                )

            pipeline_response = self.annotation_graph.process_annotation_query(
                query=state["user_query"],
                user_id=state["user_id"],
                query_type=query_type,
            )
            
            logger.info(f"Pipeline response: {pipeline_response}")
            
            if pipeline_response.get("success", False):
                summary = pipeline_response.get("summary", "")
                json_format = pipeline_response.get("json_format", None)

                response_dict = {
                    "text": summary if summary else "",
<<<<<<< HEAD
<<<<<<< HEAD
                    "json_query": json_query,
                    "source": "annotation database"
=======
                    "json_format": json_format if json_format is not None else None
>>>>>>> d6f261f (quick fix : return parameter)
=======
                    "json_query": json_query,
                    "source": "annotation database"
>>>>>>> 1e28456 (multi agent strucuture intiated)
                }

                return {
                    "annotation_response": response_dict,
                    "messages": [AIMessage(content="Annotation processing completed")]
                }
            else:
                error_msg = pipeline_response.get("error", "Unknown error")
                logger.error(f"Annotation pipeline failed: {error_msg}")
                return {
                    "annotation_response": {
                        "text": f"Error: {error_msg}", 
                        "json_query": None,
                        "source": "annotation database"
                    },
                    "error": error_msg,
                }

        except Exception as e:
            logger.error("Unexpected error in annotation agent", exc_info=True)
            return {
                "annotation_response": {
                    "text": f"Error: {str(e)}", 
                    "json_query": None,
                    "source": "annotation database"
                },
                "error": str(e),
            }
    # def _hypothesis_agent(self, state: AgentState) -> Dict[str, Any]:
    #     """Handle hypothesis generation queries"""
    #     logger.info(
    #         f"Hypothesis agent processing query: {state['user_query']} for user: {state['user_id']}"
    #     )
    #     try:
    #         emit_to_user(user=state["user_query"], message="Generating hypothesis...")
    #         response = self.hypothesis_generation.generate_hypothesis(
    #             token=state["token"],
    #             user_query=state["user_query"],
    #             user_id=state["user_id"],
    #         )

        #     return {
        #         "response": response,
        #         "messages": [AIMessage(content=f"Hypothesis generated: {response}")],
        #     }
        # except Exception as e:
        #     logger.error("Error in hypothesis agent", exc_info=True)
        #     return {
        #         "response": f"Error generating hypothesis: {str(e)}",
        #         "error": str(e),
        #         "messages": [
        #             AIMessage(content=f"Error in hypothesis generation: {str(e)}")
        #         ],
        #     }


    def _rag_agent(self, state: AgentState) -> Dict[str, Any]:
        """Handle general information queries"""
        logger.info(
            f"RAG agent processing query: {state['user_query']} for user: {state['user_id']}"
        )
        
        try:
            emit_to_user(user=state["user_id"], message="Retrieving information...")
            
            response = self.rag.get_result_from_rag(
                state["user_query"],
                state["user_id"],
                content_ids=state.get("content_ids"),
            )

            # Normalize response to dict with text key
            if response and isinstance(response, dict) and "text" in response:
                response_text = response["text"]
            else:
                response_text = str(response) if response else "No response generated"

            return {
                "rag_response": {
                    "text": response_text, 
                    "json_query": None,
                    "source": "knowledge base"
                },
                "messages": [AIMessage(content="RAG query processed")],
            }
            
        except Exception as e:
            logger.error("Error in RAG agent", exc_info=True)
            return {
                "rag_response": {
                    "text": f"Error: {str(e)}", 
                    "json_query": None,
                    "source": "knowledge base"
                },
                "error": str(e),
            }

    def _galaxy_agent(self, state: AgentState) -> Dict[str, Any]:
        """Handle Galaxy tools and workflows queries"""
        logger.info(
            f"Galaxy agent processing query: {state['user_query']} for user: {state['user_id']}"
        )
        
        try:
            emit_to_user(
                user=state["user_id"], 
                message="Retrieving Galaxy tools information..."
            )
            
            response = self.galaxy_handler.get_galaxy_info(
                state["user_query"], 
                state["user_id"], 
                state["token"]
            )

            # Normalize response
            if isinstance(response, dict) and "text" in response:
                response_text = response["text"]
            else:
                response_text = str(response) if response else "No Galaxy information found"

            return {
                "galaxy_response": {
                    "text": response_text, 
                    "json_query": None,
                    "source": "Galaxy platform"
                },
                "messages": [AIMessage(content="Galaxy query processed")],
            }
            
        except Exception as e:
            logger.error("Error in galaxy agent", exc_info=True)
            return {
                "galaxy_response": {
                    "text": f"Error: {str(e)}", 
                    "json_query": None,
                    "source": "Galaxy platform"
                },
                "error": str(e),
            }

    def _content_retrieval_agent(self, state: AgentState) -> Dict[str, Any]:
        """
        Retrieve relevant content from multiple sources with source attribution
        """
        query = state.get("user_query")
        user_id = state.get("user_id")
        token = state.get("token")
        graph_id = state.get("graph_id")
        files = state.get("files")
        content_ids = state.get("content_ids")
        resource = state.get("resource")

        logger.info(f"ContentRetrievalAgent called for user: {user_id}")
        emit_to_user(user=user_id, message="Retrieving relevant content...")

        content_parts = []
        sources = []

        try:
            # Graph summary
            if graph_id:
                logger.info(f"Retrieving graph summary for graph_id: {graph_id}")
                graph_summary = self.graph_summarizer.summary(query=query, graph_id=graph_id, token=token,resource=resource)

            # Galaxy files
            if files:
                logger.info(f"Retrieving Galaxy files for user: {user_id}")
                files_response = self.galaxy_handler.get_galaxy_info(
                    query=query, user_id=user_id, token=token
                )
                if files_response:
                    files_text = files_response.get("text", str(files_response)) if isinstance(files_response, dict) else str(files_response)
                    for file in (files if isinstance(files, list) else [files]):
                        content_parts.append({
                            "source": f"file:{file}",
                            "content": files_text
                        })
                        sources.append(f"file:{file}")

            # RAG content
            if content_ids:
                logger.info(f"Retrieving RAG content for content_ids: {content_ids}")
                rag_content = self.rag.get_result_from_rag(query, user_id, content_ids)
                if rag_content:
                    rag_text = rag_content.get("text", str(rag_content)) if isinstance(rag_content, dict) else str(rag_content)
                    content_parts.append({
                        "source": f"content IDs: {', '.join(content_ids)}",
                        "content": rag_text
                    })
                    sources.append(f"content IDs: {', '.join(content_ids)}")

            # Build response with source attribution
            if content_parts:
                response_dict = {
                    "text": content_parts,  # Keep structured format
                    "json_query": None,
                    "sources": sources
                }
            else:
                response_dict = {
                    "text": [],
                    "json_query": None,
                    "sources": []
                }

            return {
                "content_retrieval_response": response_dict,
                "messages": [AIMessage(content="Content retrieval completed")]
            }

        except Exception as e:
            logger.error(f"Error in ContentRetrievalAgent: {str(e)}", exc_info=True)
            return {
                "content_retrieval_response": {
                    "text": [], 
                    "json_query": None,
                    "sources": []
                },
                "error": str(e),
            }

<<<<<<< HEAD
    def _biogpt_agent(self, state: AgentState) -> dict:
        try:
            return self.biogpt.biogpt_agent_function(state["user_query"], state["user_id"], state["token"])
        except Exception as e:
            logger.error(f"Error in biogpt agent: {str(e)}", exc_info=True)
            return {
                "biogpt_response": {
                    "text": f"Error: {str(e)}",
                    "json_query": None,
                    "source": "BioGPT"
                },
                "error": str(e)
            }


=======
>>>>>>> 1e28456 (multi agent strucuture intiated)
    def _aggregate_responses(self, state: AgentState) -> Dict[str, Any]:
        """
        Aggregate responses from all agents with source attribution.
        Ensures that text content is combined coherently and structured JSON data (json_query)
        is always included when available.
        """
        user_query = state.get("user_query", "")
        logger.info("Aggregating responses from multiple agents with source attribution")

        agent_outputs = []
        json_query = None

        # ---------------- Annotation Agent ----------------
        annotation_resp = state.get("annotation_response")
        if annotation_resp:
            # Prefer 'text', then 'summary'
            text_content = annotation_resp.get("text") or annotation_resp.get("summary") or ""
            if text_content:
                agent_outputs.append({
                    "agent": "annotation_agent",
                    "source": annotation_resp.get("source", "annotation database"),
                    "content": text_content
                })

            # Capture JSON structured data
            json_query = annotation_resp.get("json_query") or annotation_resp.get("json_format")

            # Add placeholder if only JSON exists
            if json_query and not text_content:
                agent_outputs.append({
                    "agent": "annotation_agent",
                    "source": annotation_resp.get("source", "annotation database"),
                    "content": "Annotation data retrieved successfully (see structured data)."
                })

        # ---------------- RAG Agent ----------------
        rag_resp = state.get("rag_response")
        if rag_resp:
            text_content = rag_resp.get("text", "")
            if text_content:
                agent_outputs.append({
                    "agent": "rag_agent",
                    "source": rag_resp.get("source", "knowledge base"),
                    "content": text_content
                })

        # ---------------- Galaxy Agent ----------------
        galaxy_resp = state.get("galaxy_response")
        if galaxy_resp:
            text_content = galaxy_resp.get("text", "")
            if text_content:
                agent_outputs.append({
                    "agent": "galaxy_agent",
                    "source": galaxy_resp.get("source", "Galaxy platform"),
                    "content": text_content
                })

        # ---------------- Content Retrieval Agent ----------------
        content_resp = state.get("content_retrieval_response")
        if content_resp:
            content_parts = content_resp.get("text", [])
            if isinstance(content_parts, list):
                for part in content_parts:
                    if isinstance(part, dict) and part.get("content"):
                        agent_outputs.append({
                            "agent": "content_retrieval_agent",
                            "source": part.get("source", "external content"),
                            "content": part["content"]
                        })
            elif isinstance(content_parts, str) and content_parts:
                sources = content_resp.get("sources", ["external content"])
                agent_outputs.append({
                    "agent": "content_retrieval_agent",
                    "source": ", ".join(sources),
                    "content": content_parts
                })

        # ---------------- Handle Empty Outputs ----------------
        if not agent_outputs and json_query:
            return {
                "response": {
                    "text": "I found the requested annotation data in the database.",
                    "json_query": json_query
                }
            }

        if not agent_outputs:
            return {
                "response": {
                    "text": "I couldn't find any relevant information to answer your query.",
                    "json_query": None
                }
            }

        # ---------------- LLM Aggregation ----------------
        try:
            sources_info = []
            for output in agent_outputs:
                content = output.get("content", "").strip()
                if content:
                    sources_info.append(f"From {output.get('source', 'unknown')}: {content}")

            combined_text = "\n\n".join(sources_info)

            # Include json_query note if present
            json_note = ""
            if json_query:
                json_note = "\n\nNote: Structured annotation data is also available for this query."

            prompt = f"""You are an AI assistant acting as a **final aggregator**. 
    Your task is to respond to the user's query: "{user_query}".

    You have outputs from multiple agents, which may provide overlapping, complementary, or missing information.

    Information from agents:
    {combined_text}{json_note}

    Write a **single, fluent, and conversational summary**:
    - Integrate all findings naturally into one flowing explanation.
    - Reference sources naturally (e.g., "Based on the annotation database..." or "From the knowledge base...").
    - Highlight conflicts if any.
    - Keep it helpful, informative, and readable.
    - Acknowledge structured annotation data if available.
    """

            aggregated_text = self.advanced_llm.generate(prompt)
            logger.info(f"Successfully aggregated response: {aggregated_text[:100]}...")

            return {
                "response": {
                    "text": aggregated_text,
                    "json_query": json_query
                }
            }

        except Exception as e:
            logger.error(f"Error in aggregation: {str(e)}", exc_info=True)
            # Fallback: simple concatenation with sources
            fallback_parts = [
                f"**From {output.get('source', 'unknown')}:**\n{output.get('content', '').strip()}"
                for output in agent_outputs if output.get('content')
            ]
            fallback_text = "\n\n".join(fallback_parts) if fallback_parts else "Annotation data retrieved."

            return {
                "response": {
                    "text": fallback_text,
                    "json_query": json_query
                }
            }

    def _finalize_response(self, state: AgentState) -> Dict[str, Any]:
        """Finalize and return the response"""
        response = state.get("response", {})
        user_id = state.get("user_id")
        
        logger.info(f"Finalizing response for user: {user_id}")
        
        # Ensure response has correct structure
        if not isinstance(response, dict):
<<<<<<< HEAD
<<<<<<< HEAD
            response = {"text": str(response), "json_query": None}
        
        response.setdefault("text", "")
        response.setdefault("json_query", None)
        
        # Save to history
=======
            response = {"text": str(response), "json_format": None}
        response.setdefault("text", "")
        response.setdefault("json_format", None)
        return response  # Only text and json_format
>>>>>>> d6f261f (quick fix : return parameter)
=======
            response = {"text": str(response), "json_query": None}
        
        response.setdefault("text", "")
        response.setdefault("json_query", None)
        
        # Save to history
>>>>>>> 1e28456 (multi agent strucuture intiated)

        try:
            self.history.create_history(
                user_id=user_id,
                user_message=state.get("user_query", ""),
                assistant_answer=response.get("text", "")
            )
        except Exception as e:
            logger.error(f"Error saving history: {str(e)}")
        
        # Emit final response
        emit_to_user(user=user_id, message=response, status="completed")
        
        return {"response": response}

    def agent(
        self,
        message: str,
        user_id: str,
        token: str,
        content_ids: Optional[List[str]] = None,
        graph_id: Optional[str] = None,
        files: Optional[List[str]] = None,
        resource: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Main entry point for processing queries with parallel agent execution"""
        logger.info(
            f"Agent called with message: {message}, user_id: {user_id}, "
            f"content_ids: {content_ids}, graph_id: {graph_id}, files: {files}"
        )
        
        try:
            # Create initial state
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "user_query": message,
                "user_id": user_id,
                "token": token,
                "query_types": [],
                "response": {"text": "", "json_query": None},
                "error": "",
                "content_ids": content_ids,
                "graph_id": graph_id,
                "files": files,
                "pipeline_details": {},
                "annotation_response": None,
                "rag_response": None,
                "galaxy_response": None,
                "content_retrieval_response": None,
                "resource": resource,
            }

            # Run the workflow
            result = self.app.invoke(initial_state)

<<<<<<< HEAD
<<<<<<< HEAD
            # Extract the final response
            response = result.get("response", {"text": "", "json_query": None})
=======
            # Always extract the structured response
            response = result.get("response", {"text": "", "json_format": None})
>>>>>>> d6f261f (quick fix : return parameter)
=======
            # Extract the final response
            response = result.get("response", {"text": "", "json_query": None})
>>>>>>> 1e28456 (multi agent strucuture intiated)

            # Ensure consistent structure
            if not isinstance(response, dict):
                response = {"text": str(response), "json_format": None}
            else:
                response.setdefault("text", "")
                response.setdefault("json_format", None)

            logger.info(f"Agent completed successfully for user: {user_id}")
            return response

        except Exception as e:
            logger.error("Error in agent processing", exc_info=True)
            error_response = {
                "text": f"I apologize, but I encountered an error while processing your request: {str(e)}",
                "json_query": None
            }
            emit_to_user(user=user_id, message=error_response, status="error")
            return error_response

    def assistant_response(
        self, 
        query: str, 
        user_id: str, 
        token: str, 
        graph_id: Optional[str] = None,
        files: Optional[List[str]] = None,
        content_ids: Optional[List[str]] = None,
        resource: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point for assistant responses.
        Routes to parallel agent execution system.
        """
        try:
            logger.info(
                f"Assistant response called with query={query}, user_id={user_id}, "
                f"graph_id={graph_id}, content_ids={content_ids}, files={files}"
            )
            logger.info(
            f"Assistant called with query: {query}, user_id: {user_id}"
            )
            try:
                user_information = self.store.get_context_and_memory(user_id)
                history = []
                memory = []
                for item in user_information:
                    q = item["QUESTION"]["question"]
                    c = item["QUESTION"]["context"]
                    m = item["MEMORIES"]
                    history.append({"question": q, "context": c})
                    memory.append(m)
            except Exception as e:
                history = " "
                memory = " "

            logger.info(f"Histories of the user are : {history} and memories are {memory}")

            prompt = conversation_prompt.format(
                memory=memory,
                query=query,
                conversation_history=history,
            )
            logger.info("Advanced llm response")
            response = self.advanced_llm.generate(prompt)
            logger.info(f"Response from the advanced LLM: {response}")
            emit_to_user(user=user_id, message="Analyzing...")
            
            if response:
                if "response:" in response:
                    result = response.split("response:")[1].strip()
                    final_response = result.strip('"')
                    self.store.save_user_information(
                        advanced_llm=self.advanced_llm,
                        query=query,
                        user_id=user_id,
                        context=None,
                        graph_id_referenced=graph_id,
                    )
                    emit_to_user(user=user_id, message=final_response, status="completed")
                    return {"text": final_response}

                elif "question:" in response:
                    refactored_question = response.split("question:")[1].strip()
                    agent_response = self.agent(
                        refactored_question,
                        user_id,
                        token,
                        content_ids=content_ids,
                    )
                    if isinstance(agent_response, str):
                        agent_response = {"text": agent_response}
                    elif isinstance(agent_response, dict):
                        pass
                    else:
                        agent_response = {"text": str(agent_response)}

                    resource_type = (
                        agent_response.get("resource", {}).get("type")
                        if agent_response
                        else None
                    )

                    response_resource = None
                    if resource_type:
                        logger.info(f"Here is the resource successfully made {resource_type}")

                    emit_to_user(user=user_id, message=agent_response, status="completed")
                    assistant_answer = (
                        agent_response.get("text", str(agent_response))
                        if isinstance(agent_response, dict)
                        else str(agent_response)
                    )
                    self.history.create_history(
                        user_id, query, assistant_answer, graph_id
                    )
                    return agent_response
            else:
                logger.error("No response generated from LLM")
                self.store.save_user_information(
                    self.advanced_llm, query, user_id, resource
                )
            
                error_msg = (
                    "I apologize, but I encountered an error while processing your request."
                )
                emit_to_user(user=user_id, message={"text": error_msg}, status="completed")
                return {"text": error_msg}, history
          
        except Exception as e:
            logger.error(f"Error in assistant_response: {e}", exc_info=True)
            return {
                "text": "I apologize, but I encountered an error while processing your request.",
                "json_query": None
            }

          
    def answer_from_graph_summaries(self, query, user_id, resource, token, graph_id):
        """Legacy method for backward compatibility"""
        logger.info(
            f"Answer from graph summaries called with query: {query}, user_id: {user_id}, "
            f"resource: {resource}, graph_id: {graph_id}"
        )
        
        try:
            if resource == "annotation":
                summary_result = self.graph_summarizer.summary(
                    token=token, graph_id=graph_id, user_query=query
                )
                summary_text = summary_result.get('text', '') if isinstance(summary_result, dict) else summary_result
                emit_to_user(user=user_id, message="Analyzing...")

            elif resource == "hypothesis":
                summary_result = self.hypothesis_generation.get_by_hypothesis_id(
                    token, graph_id, user_id, query
                )
                summary_text = summary_result.get('text', '') if isinstance(summary_result, dict) else summary_result
                emit_to_user(user=user_id, message="Analyzing...")
            else:
                return "Invalid resource type specified."
        except Exception as e:
            logger.error("Error in agent processing", exc_info=True)
            return f"Error processing query: {str(e)}"
