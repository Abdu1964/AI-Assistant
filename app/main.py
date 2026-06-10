from .llm_handle.llm_models import (
    LLMInterface,
    OpenAIModel,
    get_llm_model,
    openai_embedding_model,
)
from .prompts.classifier_prompt import (
    classifier_prompt,
    main_classifier_prompt,
    aggregator_prompt
)
from .annotation_graph.annotated_graph import Graph
from .annotation_graph.schema_handler import SchemaHandler
from .rag.rag import RAG
from .rag.utils.web_search import SimpleWebSearch
from .prompts.conversation_handler import conversation_prompt
from .summarizer import GraphSummarizer
from .hypothesis_generation.hypothesis import HypothesisGeneration
from .socket_manager import emit_to_user
from .Galaxy_integration.galaxy import GalaxyHandler
from .biogpt_agent.biogpt import BioGPTAgentOpenVINO
from typing import TypedDict, List, Annotated, Any, Dict, Optional
from flask_socketio import emit
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import tool
import asyncio
import traceback
import json
import os
import operator
import logging


logger = logging.getLogger(__name__)

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_query: str
    user_id: str
    token: str
    query_types: List[str]
    response: Dict[str, Any]
    error: str
    content_ids: Optional[List[str]]
    graph_id: Optional[str]
    urls: Optional[List[str]]
    resource: Optional[Any]
    pipeline_details: Dict[str, Any]
    # Agent-specific responses with source attribution
    annotation_response: Optional[Dict[str, Any]]
    rag_response: Optional[Dict[str, Any]]
    galaxy_response: Optional[Dict[str, Any]]
    content_retrieval_response: Optional[Dict[str, Any]]
    biogpt_response:Optional[Dict[str, Any]]
    hypothesis_response: Optional[Dict[str, Any]]
    # Parallel execution control
    agents_to_run: List[str]
    agents_completed: Annotated[List[str], operator.add]
    stop_pipeline: Optional[bool]


ANNOTATION_DB = "annotation database"
KNOWLEDGE_BASE = "knowledge base"
GALAXY_PLATFORM = "Galaxy platform"
ANALYZING_MSG = "Analyzing..."


class AiAssistance:

    def __init__(
        self,
        advanced_llm,
        basic_llm,
        schema_handler,
        fly_schema_handler=None,
        qdrant_client=None,
        embedding_model=None,
        mongo_db_manager=None,
    ) -> None:
        self.advanced_llm = advanced_llm
        self.basic_llm = basic_llm
        self.annotation_graph = Graph(advanced_llm, schema_handler, fly_schema_handler=fly_schema_handler)
        self.graph_summarizer = GraphSummarizer(self.advanced_llm)
        self.rag = RAG(llm=advanced_llm, qdrant_client=qdrant_client)
        self.store = mongo_db_manager
        self.hypothesis_generation = HypothesisGeneration(advanced_llm)
        self.galaxy_handler = GalaxyHandler(advanced_llm, qdrant_client, embedding_model)
        self.embedding_model = embedding_model
        self.biogpt = BioGPTAgentOpenVINO(llm=advanced_llm)

        logger.info(
            f"AiAssistance initialized with advanced_llm: {type(self.advanced_llm).__name__}"
        )
        logger.info(f"Galaxy handler initialized: {type(self.galaxy_handler).__name__}")

        # Initialize the LangGraph workflow
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile()

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow with proper parallel agent execution"""
        logger.info("Creating LangGraph workflow with parallel agent execution")

        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("classifier", self._classify_query)
        workflow.add_node("router", self._router)
        workflow.add_node("hypothesis_agent", self._hypothesis_agent)
        workflow.add_node("annotation_agent", self._annotation_agent)
        workflow.add_node("rag_agent", self._rag_agent)
        workflow.add_node("galaxy_agent", self._galaxy_agent)
        workflow.add_node("content_retrieval_agent", self._content_retrieval_agent)
        workflow.add_node("biogpt_agent", self._biogpt_agent)
        workflow.add_node("aggregator", self._aggregate_responses)
        workflow.add_node("finalizer", self._finalize_response)

        # Define edges
        workflow.set_entry_point("classifier")
        workflow.add_edge("classifier", "router")

        # Router decides which agents to invoke
        workflow.add_conditional_edges(
            "router",
            self._should_run_agent,
            {
                "annotation_agent": "annotation_agent",
                "hypothesis_agent": "hypothesis_agent",
                "rag_agent": "rag_agent",
                "galaxy_agent": "galaxy_agent",
                "content_retrieval_agent": "content_retrieval_agent",
                "biogpt_agent": "biogpt_agent",
                "aggregator": "aggregator",
                "error" : "finalizer"
            },
        )

        # All agents go back to router to check for next agent
        workflow.add_edge("annotation_agent", "router")
        workflow.add_edge("rag_agent", "router")
        workflow.add_edge("galaxy_agent", "router")
        workflow.add_edge("content_retrieval_agent", "router")
        workflow.add_edge("biogpt_agent", "router")
        workflow.add_edge("hypothesis_agent", "router")
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

    def _classify_query_types(self, qtype: str, query_types: list) -> None:
        if ("annotation_biological" in qtype or "annotation biological" in qtype) and "annotation_biological" not in query_types:
            query_types.append("annotation_biological")
        if ("annotation_general" in qtype or "annotation general" in qtype) and "annotation_general" not in query_types:
            query_types.append("annotation_general")
        if "galaxy" in qtype and "galaxy" not in query_types:
            query_types.append("galaxy")
        if "rag" in qtype and "rag" not in query_types:
            query_types.append("rag")
        if "hypothesis" in qtype and "hypothesis_generation" not in query_types:
            query_types.append("hypothesis_generation")
        if "biogpt" in qtype and "biogpt" not in query_types:
            query_types.append("biogpt")

    def _build_agent_list(self, query_types: list, content_ids, urls, graph_id) -> list:
        agents_to_run = []
        if content_ids or urls or graph_id:
            agents_to_run.append("content_retrieval_agent")
        type_to_agent = {
            "annotation_biological": "annotation_agent",
            "hypothesis_generation": "hypothesis_agent",
            "annotation_general": "annotation_agent",
            "galaxy": "galaxy_agent",
            "rag": "rag_agent",
            "biogpt": "biogpt_agent",
        }
        for qtype in query_types:
            agent = type_to_agent.get(qtype)
            if agent == "annotation_agent" and graph_id:
                continue
            if agent and agent not in agents_to_run:
                agents_to_run.append(agent)
        if not agents_to_run:
            agents_to_run.append("rag_agent")
        return agents_to_run

    def _classify_query(self, state: AgentState) -> Dict[str, Any]:
        """Classify query and determine which agents to invoke (can be multiple)"""
        query = state["user_query"]
        user_id = state["user_id"]
        content_ids = state.get("content_ids")
        graph_id = state.get("graph_id")
        urls = state.get("urls")
        resource = state.get("resource")

        # If the client explicitly set resource="hypothesis", skip LLM classification.
        # - graph_id present: query an existing hypothesis via content_retrieval_agent → get_by_hypothesis_id
        # - no graph_id: generate a new hypothesis via hypothesis_agent
        if resource == "hypothesis":
            if graph_id:
                logger.info("Resource='hypothesis' + graph_id — routing to content_retrieval_agent")
                return {
                    "query_types": ["hypothesis_generation"],
                    "agents_to_run": ["content_retrieval_agent"],
                    "agents_completed": [],
                    "messages": [HumanMessage(content="Query classified as: hypothesis_generation")],
                }
            else:
                logger.info("Resource='hypothesis' + no graph_id — routing to hypothesis_agent")
                return {
                    "query_types": ["hypothesis_generation"],
                    "agents_to_run": ["hypothesis_agent"],
                    "agents_completed": [],
                    "messages": [HumanMessage(content="Query classified as: hypothesis_generation")],
                }

        content_summaries = self.get_content_summaries(user_id, content_ids)
        logger.info(f"Classifying query: {query}")

        classifier_prompt_text = main_classifier_prompt.format(
            query=query,
            content_summaries=content_summaries,
        )
        response = self.advanced_llm.generate(classifier_prompt_text).lower()
        logger.info(f"question classified as {response}")

        query_types = []
        cleaned_response = response.replace("and", ",").replace("\n", ",")
        potential_types = [t.strip() for t in cleaned_response.split(",")]

        for qtype in potential_types:
            self._classify_query_types(qtype, query_types)

        if not query_types:
            query_types = ["rag"]

        logger.info(f"Query classified as: {query_types}")

        agents_to_run = self._build_agent_list(query_types, content_ids, urls, graph_id)

        logger.info(f"Agents to run: {agents_to_run}")

        return {
            "query_types": query_types,
            "agents_to_run": agents_to_run,
            "agents_completed": [],
            "messages": [HumanMessage(content=f"Query classified as: {', '.join(query_types)}")],
        }

    def _router(self, state: AgentState) -> Dict[str, Any]:
        """Router node that doesn't change state, just passes through"""
        return {}

    def _should_run_agent(self, state: AgentState) -> str:
        """
        Determine which agent to run next.
        Returns the next agent to run, or 'aggregator' if all agents have completed.
        """
        # Short-circuit: an agent signalled that no further processing is needed
        if state.get("stop_pipeline"):
            logger.info("stop_pipeline flag set — skipping remaining agents, going to aggregator")
            return "aggregator"

        agents_to_run = state.get("agents_to_run", [])
        agents_completed = state.get("agents_completed", [])

        # Find the next agent that hasn't been completed
        for agent in agents_to_run:
            if agent not in agents_completed:
                logger.info(f"Running next agent: {agent}")
                return agent

        # All agents completed, move to aggregator
        logger.info("All agents completed, moving to aggregator")
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

            if pipeline_response.get("needs_confirmation"):
                return {
                    "annotation_response": {
                        "text": pipeline_response.get("confirmation_text", ""),
                        "json_format": None,
                        "needs_confirmation": True,
                        "source": ANNOTATION_DB,
                    },
                    "agents_completed": ["annotation_agent"],
                    "messages": [AIMessage(content="Annotation needs user confirmation")],
                }

            if pipeline_response.get("success", False):
                summary = pipeline_response.get("summary", "")
                json_format = pipeline_response.get("json_format", None)
                validation_report = pipeline_response.get("validation_report", {})
                organism = pipeline_response.get("organism", "human")

                response_dict = {
                    "text": summary if summary else "",
                    "json_format": json_format,
                    "validation_report": validation_report,
                    "organism": organism,
                    "source": ANNOTATION_DB
                }

                return {
                    "annotation_response": response_dict,
                    "agents_completed": ["annotation_agent"],
                    "messages": [AIMessage(content="Annotation processing completed")]
                }

            else:
                error_msg = pipeline_response.get("error", "Unknown error")
                logger.error(f"Annotation pipeline failed: {error_msg}")
                return {
                    "annotation_response": {
                        "text": f"Error: {error_msg}", 
                        "json_format": None,
                        "source": ANNOTATION_DB
                    },
                    "agents_completed": ["annotation_agent"],
                    "error": error_msg,
                }

        except Exception as e:
            logger.error("Unexpected error in annotation agent", exc_info=True)
            return {
                "annotation_response": {
                    "text": f"Error: {str(e)}", 
                    "json_format": None,
                    "source": ANNOTATION_DB
                },
                "agents_completed": ["annotation_agent"],
                "error": str(e),
            }

    def _hypothesis_agent(self, state: AgentState) -> Dict[str, Any]:
        """Handle hypothesis generation queries"""
        logger.info(
            f"Hypothesis agent processing query: {state['user_query']} for user: {state['user_id']}"
        )
        try:
            emit_to_user(user=state["user_id"], message="Generating hypothesis...")
            response = self.hypothesis_generation.generate_hypothesis(
                token=state["token"],
                user_query=state["user_query"],
                user_id=state["user_id"],
            )

            return {
                "hypothesis_response": response,  
                "messages": [AIMessage(content=f"Hypothesis generated: {response.get('text')}")],
                "agents_completed": ["hypothesis_agent"],
            }
       
        except Exception as e:
            logger.error("Error in hypothesis agent", exc_info=True)
            return {
                "hypothesis_response": {"text": f"Error generating hypothesis: {str(e)}", "resource": None},
                "error": str(e),
                "messages": [
                    AIMessage(content=f"Error in hypothesis generation: {str(e)}")
                ],
                "agents_completed": ["hypothesis_agent"],
            }

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
            logger.debug(f"RAG response: {response_text}")
            return {
                "rag_response": {
                    "text": response_text, 
                    "json_format": None,
                    "source": KNOWLEDGE_BASE
                },
                "agents_completed": ["rag_agent"],
                "messages": [AIMessage(content="RAG query processed")],
            }
            
        except Exception as e:
            logger.error("Error in RAG agent", exc_info=True)
            return {
                "rag_response": {
                    "text": f"Error: {str(e)}", 
                    "json_format": None,
                    "source": KNOWLEDGE_BASE
                },
                "agents_completed": ["rag_agent"],
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
            logger.debug(f"Galaxy response: {response_text}")
            return {
                "galaxy_response": {
                    "text": response_text, 
                    "json_format": None,
                    "source": GALAXY_PLATFORM
                },
                "agents_completed": ["galaxy_agent"],
                "messages": [AIMessage(content="Galaxy query processed")],
            }
            
        except Exception as e:
            logger.error("Error in galaxy agent", exc_info=True)
            return {
                "galaxy_response": {
                    "text": f"Error: {str(e)}", 
                    "json_format": None,
                    "source": GALAXY_PLATFORM
                },
                "agents_completed": ["galaxy_agent"],
                "error": str(e),
            }

    def _retrieve_from_graph(self, query, user_id, graph_id, token, resource, content_parts, sources):
        logger.info(f"Retrieving graph summary for graph_id: {graph_id}")
        graph_summary = self.answer_from_graph_summaries(
            query=query,
            user_id=user_id,
            graph_id=graph_id,
            token=token,
            resource=resource
        )
        if not graph_summary:
            return None
        graph_text = graph_summary.get("text", str(graph_summary)) if isinstance(graph_summary, dict) else str(graph_summary)
        if graph_text and not graph_text.startswith("Failed to contact") and not graph_text.startswith("Error"):
            content_parts.append({"source": f"graph:{graph_id}", "content": graph_text})
            sources.append(f"graph:{graph_id}")
            return None
        if graph_text:
            logger.warning(f"Graph fetch failed for {graph_id}: {graph_text}")
            last_topic = None
            try:
                history = self.store.get_context_and_memory(user_id)
                for item in reversed(history):
                    agents_used = item.get("context", {}).get("agents_used", [])
                    if "annotation_agent" in agents_used:
                        last_topic = item.get("question")
                        break
            except Exception:
                pass
            if last_topic:
                confirmation_text = (
                    f"I couldn't find the graph you referenced (ID: `{graph_id}`). "
                    f"Did you mean to ask about your previous annotation: *\"{last_topic}\"*? "
                    f"Or would you like to ask a different question?"
                )
            else:
                confirmation_text = (
                    f"I couldn't find the graph you referenced (ID: `{graph_id}`). "
                    f"Please check that the graph exists, or let me know what you'd like to explore."
                )
            return {
                "content_retrieval_response": {
                    "text": confirmation_text,
                    "json_format": None,
                    "sources": []
                },
                "agents_completed": ["content_retrieval_agent"],
                "stop_pipeline": True,
            }
        return None

    def _retrieve_from_galaxy(self, query, user_id, token, urls, content_parts, sources):
        logger.info(f"Retrieving Galaxy urls for user: {user_id}")
        urls_response = self.galaxy_handler.get_galaxy_info(
            query=query, user_id=user_id, token=token, urls=urls
        )
        if urls_response:
            urls_text = urls_response.get("text", str(urls_response)) if isinstance(urls_response, dict) else str(urls_response)
            for file in (urls if isinstance(urls, list) else [urls]):
                content_parts.append({"source": f"file:{file}", "content": urls_text})
                sources.append(f"file:{file}")

    def _retrieve_from_rag(self, query, user_id, content_ids, content_parts, sources):
        logger.info(f"Retrieving RAG content for content_ids: {content_ids}")
        rag_content = self.rag.get_result_from_rag(query, user_id, content_ids)
        if rag_content:
            rag_text = rag_content.get("text", str(rag_content)) if isinstance(rag_content, dict) else str(rag_content)
            resources = rag_content.get("resource", {})
            content_parts.append({
                "source": f"content IDs: {', '.join(content_ids)}",
                "content": rag_text,
                "resource": resources
            })
            sources.append(f"content IDs: {', '.join(content_ids)}")

    def _content_retrieval_agent(self, state: AgentState) -> Dict[str, Any]:
        """
        Retrieve relevant content from multiple sources with source attribution
        """
        query = state.get("user_query")
        user_id = state.get("user_id")
        token = state.get("token")
        graph_id = state.get("graph_id")
        urls = state.get("urls")
        content_ids = state.get("content_ids")
        resource = state.get("resource")

        logger.info(f"ContentRetrievalAgent called for user: {user_id}")
        emit_to_user(user=user_id, message="Retrieving relevant content...")

        content_parts = []
        sources = []

        try:
            if graph_id:
                early_return = self._retrieve_from_graph(query, user_id, graph_id, token, resource, content_parts, sources)
                if early_return is not None:
                    return early_return

            if urls:
                self._retrieve_from_galaxy(query, user_id, token, urls, content_parts, sources)

            if content_ids:
                self._retrieve_from_rag(query, user_id, content_ids, content_parts, sources)

            response_dict = {
                "text": content_parts,
                "json_format": None,
                "sources": sources
            }
            logger.info(f"Content retrieval response prepared with {len(content_parts)} parts. response is {response_dict}")
            return {
                "content_retrieval_response": response_dict,
                "agents_completed": ["content_retrieval_agent"],
                "messages": [AIMessage(content="Content retrieval completed")]
            }

        except Exception as e:
            logger.error(f"Error in ContentRetrievalAgent: {str(e)}", exc_info=True)
            return {
                "content_retrieval_response": {
                    "text": [],
                    "json_format": None,
                    "sources": []
                },
                "agents_completed": ["content_retrieval_agent"],
                "error": str(e),
            }

    def _biogpt_agent(self, state: AgentState) -> dict:
        try:
            emit_to_user(user=state["user_id"], message="Analyzing biomedical information...")
            response = self.biogpt.generate_answer(state["user_query"])
            logger.info(f"BioGPT response: {response}")
            return {
                "biogpt_response": {
                    "text": response,
                    "source": "BioGPT"
                },
                "agents_completed": ["biogpt_agent"],
                "messages": [AIMessage(content="BioGPT query processed")]
            }
        except Exception as e:
            logger.error(f"Error in biogpt agent: {str(e)}", exc_info=True)
            return {
                "biogpt_response": {
                    "text": None,
                    "json_format": None,
                    "source": "BioGPT"
                },
                "agents_completed": ["biogpt_agent"],
                "error": str(e)
            }

    def _format_annotation_section(self, failed: list) -> str:
        missing_parts = []
        for n in failed:
            not_validated = n.get("not_validated")
            if not_validated:
                items = not_validated if isinstance(not_validated, list) else [not_validated]
                for item in items:
                    missing_parts.append(f'"{item}"')
            else:
                props = n.get("properties", {})
                name = next(iter(props.values()), n.get("type", "unknown"))
                missing_parts.append(f'"{name}"')
        verb = "was" if len(missing_parts) == 1 else "were"
        joined = ", ".join(missing_parts)
        return f" Note: {joined} {verb} not found in the database."

    def _build_annotation_text(self, json_format: dict) -> str:
        """Build human-readable text from annotation validation results for truly-failed nodes."""
        nodes = json_format.get("nodes", [])
        # Ignore nodes pending confirmation — those are handled by _build_confirmation_text
        failed = [n for n in nodes if n.get("status") is False and not n.get("needs_confirmation")]
        text = "The annotation structure was created successfully (see structured data)."
        if failed:
            text += self._format_annotation_section(failed)
        return text

    def _aggregate_annotation_response(self, annotation_resp: dict, agent_outputs: list) -> tuple:
        if annotation_resp.get("needs_confirmation"):
            return None, None, True
        text_content = annotation_resp.get("text") or annotation_resp.get("summary") or ""
        json_format = annotation_resp.get("json_format")
        organism = annotation_resp.get("organism") if json_format else None
        if not text_content and json_format:
            text_content = self._build_annotation_text(json_format)
        if text_content:
            agent_outputs.append({
                "agent": "annotation_agent",
                "source": annotation_resp.get("source", ANNOTATION_DB),
                "content": text_content
            })
        return json_format, organism, False

    def _aggregate_content_responses(self, state: dict, agent_outputs: list) -> Any:
        rag_resp = state.get("rag_response")
        if rag_resp:
            text_content = rag_resp.get("text", "")
            if text_content:
                agent_outputs.append({"agent": "rag_agent", "source": rag_resp.get("source", KNOWLEDGE_BASE), "content": text_content})

        galaxy_resp = state.get("galaxy_response")
        if galaxy_resp:
            text_content = galaxy_resp.get("text", "")
            if text_content:
                agent_outputs.append({"agent": "galaxy_agent", "source": galaxy_resp.get("source", GALAXY_PLATFORM), "content": text_content})

        biogpt_resp = state.get("biogpt_response")
        if biogpt_resp:
            text_content = biogpt_resp.get("text", "")
            if text_content:
                agent_outputs.append({"agent": "biogpt_agent", "source": biogpt_resp.get("source", "biogpt"), "content": text_content})

        resource_to_save = state.get("resource")
        hyp_resp = state.get("hypothesis_response")
        if hyp_resp:
            text_content = hyp_resp.get("text", "")
            if text_content:
                agent_outputs.append({"agent": "hypothesis_agent", "source": "hypothesis generator", "content": text_content})
            if hyp_resp.get("resource"):
                resource_to_save = hyp_resp.get("resource")

        content_resp = state.get("content_retrieval_response")
        if content_resp:
            content_parts = content_resp.get("text", [])
            if isinstance(content_parts, list):
                for part in content_parts:
                    if isinstance(part, dict) and part.get("content"):
                        agent_outputs.append({"agent": "content_retrieval_agent", "source": part.get("source", "external content"), "content": part["content"]})
            elif isinstance(content_parts, str) and content_parts:
                sources = content_resp.get("sources", ["external content"])
                agent_outputs.append({"agent": "content_retrieval_agent", "source": ", ".join(sources), "content": content_parts})

        return resource_to_save

    def _aggregate_responses(self, state: AgentState) -> Dict[str, Any]:
        """
        Aggregate responses from all agents with source attribution.
        Ensures that text content is combined coherently and structured JSON data (json_format)
        is always included when available.
        """
        user_query = state.get("user_query", "")
        logger.info("Aggregating responses from multiple agents with source attribution")

        agent_outputs = []
        json_format = None
        organism = None

        # ---------------- Annotation Agent ----------------
        annotation_resp = state.get("annotation_response")
        if annotation_resp:
            json_format, organism, needs_confirm = self._aggregate_annotation_response(annotation_resp, agent_outputs)
            if needs_confirm:
                return {
                    "response": {
                        "text": annotation_resp.get("text", ""),
                        "json_format": None,
                    }
                }

        resource_to_save = self._aggregate_content_responses(state, agent_outputs)

        # ---------------- Handle JSON-only case ----------------
        if json_format and not agent_outputs:
            nodes = json_format.get("nodes", [])
            logger.info(f"[note-check] node statuses: { {n.get('node_id'): n.get('status') for n in nodes} }")
            failed = [n for n in nodes if n.get("status") is False]
            logger.info(f"[note-check] failed nodes: {[n.get('node_id') for n in failed]}")
            return {
                "response": {
                    "text": self._build_annotation_text(json_format),
                    "json_format": json_format,
                    "organism": organism
                }
            }

        if not agent_outputs:
            return {
                "response": {
                    "text": "I couldn't find any relevant information to answer your query.",
                    "json_format": None
                }
            }

        # ---------------- LLM Aggregation ----------------
        try:
            sources_info = []
            for output in agent_outputs:
                logger.info("=== [%s] source=%s ===\n%s",
                output['agent'],
                output.get('source', 'unknown'),
                str(output.get('content', ''))[:300])

                content = output.get("content", "")
                # Handle if content is a dict (convert to string)
                if isinstance(content, dict):
                    content = str(content)
                content = content.strip() if isinstance(content, str) else ""
                if content:
                    sources_info.append(f"From {output.get('source', 'unknown')}: {content}")

            combined_text = "\n\n".join(sources_info)

            # Include json_format note if present
            json_note = ""
            if json_format:
                json_note = "\n\nNote: Structured annotation data is also available for this query."

            prompt = aggregator_prompt.format(user_query=user_query, combined_text=combined_text, json_note=json_note)

            aggregated_text = self.advanced_llm.generate(prompt)
            logger.info(f"Successfully aggregated response: {aggregated_text[:100]}...")

            return {
                "response": {
                    "text": aggregated_text,
                    "json_format": json_format,
                    "organism": organism
                },
                "resource": resource_to_save
            }

        except Exception as e:
            logger.error(f"Error in aggregation: {str(e)}", exc_info=True)
            fallback_parts = []
            for output in agent_outputs:
                content = output.get('content', '')
                if isinstance(content, dict):
                    content = str(content)
                if content:
                    content_str = content.strip() if isinstance(content, str) else str(content)
                    fallback_parts.append(f"**From {output.get('source', 'unknown')}:**\n{content_str}")

            fallback_text = "\n\n".join(fallback_parts) if fallback_parts else "Annotation data retrieved."

            return {
                "response": {
                    "text": fallback_text,
                    "json_format": json_format,
                    "organism": organism
                },
                "resource": resource_to_save
            }

    def _finalize_response(self, state: AgentState) -> Dict[str, Any]:
        """Finalize and return the response"""
        response = state.get("response", {})
        user_id = state.get("user_id")
        
        logger.info(f"Finalizing response for user: {user_id}")
        logger.info(f"here is the response : {response}")
        
        # Ensure response has correct structure
        if not isinstance(response, dict):
            response = {"text": str(response), "json_format": None}
        response.setdefault("text", "")
         # Include the resource (hypothesis graph) if available
        if state.get("resource"):
            response["resource"] = state.get("resource")

        emit_to_user(user=user_id, message=response, status="completed")

        return {"response": response}

    def agent(
        self,
        message: str,
        user_id: str,
        token: str,
        content_ids: Optional[List[str]] = None,
        graph_id: Optional[str] = None,
        urls: Optional[List[str]] = None,
        resource: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Main entry point for processing queries with parallel agent execution"""
        logger.info(
            f"Agent called with message: {message}, user_id: {user_id}, "
            f"content_ids: {content_ids}, graph_id: {graph_id}, urls: {urls}"
        )
        try:
            # Create initial state
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "user_query": message,
                "user_id": user_id,
                "token": token,
                "query_types": [],
                "response": {"text": "", "json_format": None},
                "error": "",
                "content_ids": content_ids,
                "graph_id": graph_id,
                "urls": urls,
                "resource": resource,
                "pipeline_details": {},
                "annotation_response": None,
                "rag_response": None,
                "galaxy_response": None,
                "biogpt_response": None,
                "content_retrieval_response": None,
                "hypothesis_response": None,
                "stop_pipeline": False,
                "agents_to_run": [],
                "agents_completed": [],
            }

            # Run the workflow
            result = self.app.invoke(initial_state)

            # Extract the structured response
            response = result.get("response", {"text": ""})
            
            # Ensure consistent structure
            if not isinstance(response, dict):
                response = {"text": str(response), "json_format": None}
            else:
                response.setdefault("text", "")
                response.setdefault("json_format", None)

            # ✅ Add agents_completed to the response so assistant_response can save it
            response["agents_completed"] = result.get("agents_completed", [])
            
            logger.info(f"Agent completed successfully for user: {user_id}")
            return response

        except Exception as e:
            logger.error("Error in agent processing", exc_info=True)
            error_response = {
                "text": f"I apologize, but I encountered an error while processing your request: {str(e)}",
                "json_format": None,
                "agents_completed": []
            }
            emit_to_user(user=user_id, message=error_response, status="error")
            return error_response


    def _route_to_agent(self, response: str, query: str, user_id: str, token: str,
                        graph_id, content_ids, urls, resource) -> Dict[str, Any]:
        if "response:" in response:
            result = response.split("response:")[1].strip()
            final_response = result.strip('"')
            self.store.create_history(
                user_id=user_id,
                user_message=query,
                assistant_answer=final_response,
                graph_id_referenced=graph_id,
                content_ids=content_ids,
                urls=urls,
                agents_used=[],
            )
            emit_to_user(user=user_id, message=final_response, status="completed")
            return {"text": final_response}

        if "question:" in response:
            refactored_question = response.split("question:")[1].strip()
            agent_response = self.agent(
                refactored_question,
                user_id,
                token,
                content_ids=content_ids,
                graph_id=graph_id,
                urls=urls,
                resource=resource,
            )
            if isinstance(agent_response, str):
                agent_response = {"text": agent_response, "agents_completed": []}
            elif not isinstance(agent_response, dict):
                agent_response = {"text": str(agent_response), "agents_completed": []}
            resource_data = agent_response.get("resource")
            if isinstance(resource_data, dict):
                resource_type = resource_data.get("type")
                if resource_type:
                    logger.info(f"Resource successfully created: {resource_type}")
            assistant_answer = agent_response.get("text", str(agent_response))
            agents_used = agent_response.get("agents_completed", [])
            self.store.create_history(
                user_id=user_id,
                user_message=query,
                assistant_answer=assistant_answer,
                graph_id_referenced=graph_id,
                content_ids=content_ids,
                urls=urls,
                agents_used=agents_used,
            )
            emit_to_user(user=user_id, message=agent_response, status="completed")
            return agent_response

        logger.error("No response generated from LLM")
        error_msg = "I apologize, but I encountered an error while processing your request."
        self.store.create_history(
            user_id=user_id,
            user_message=query,
            assistant_answer=error_msg,
            graph_id_referenced=graph_id,
            content_ids=content_ids,
            urls=urls,
            agents_used=[],
        )
        emit_to_user(user=user_id, message={"text": error_msg}, status="completed")
        return {"text": error_msg}

    def assistant_response(
        self,
        query: str,
        user_id: str,
        token: str,
        graph_id: Optional[str] = None,
        urls: Optional[List[str]] = None,
        content_ids: Optional[List[str]] = None,
        resource: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point for assistant responses.
        Routes to parallel agent execution system.
        """
        try:
            logger.info(
                f"Assistant response called with query={query}, user_id={user_id}, "
                f"graph_id={graph_id}, content_ids={content_ids}, urls={urls}"
            )

            # Delegate to annotation_graph if a confirmation is pending for this user
            if self.annotation_graph.has_pending_for(user_id):
                resp = self.annotation_graph.handle_confirmation_response(user_id, query)
                if resp is not None:
                    self.store.create_history(
                        user_id=user_id,
                        user_message=query,
                        assistant_answer=resp.get("text", ""),
                        graph_id_referenced=graph_id,
                        content_ids=content_ids,
                        urls=urls,
                        agents_used=resp.get("agents_completed", []),
                    )
                    emit_to_user(user=user_id, message=resp, status="completed")
                    return resp
                # else: new unrelated query — annotation_graph cleared the pending state, continue normally

            # Get conversation history and memory
            try:
                user_information = self.store.get_context_and_memory(user_id)
                history = []
                memory = []
                for item in user_information:
                    q = item["question"]
                    c = item["context"]
                    history.append({"question": q, "context": c})
                    memory.append(c["memory"])
            except Exception:
                history = []
                memory = []

            logger.info(f"Histories of the user are: {history} and memories are {memory}")

            prompt = conversation_prompt.format(
                memory=memory,
                query=query,
                conversation_history=history,
                graph_id=graph_id or "",
            )
            logger.info("Advanced llm response")
            response = self.advanced_llm.generate(prompt)
            logger.info(f"Response from the advanced LLM: {response}")
            emit_to_user(user=user_id, message=ANALYZING_MSG)

            return self._route_to_agent(
                response or "",
                query, user_id, token, graph_id, content_ids, urls, resource
            )

        except Exception as e:
            logger.error(f"Error in assistant_response: {e}", exc_info=True)
            error_msg = "I apologize, but I encountered an error while processing your request."
            try:
                self.store.create_history(
                    user_id=user_id,
                    user_message=query,
                    assistant_answer=error_msg,
                    graph_id_referenced=graph_id,
                    content_ids=content_ids,
                    urls=urls,
                    agents_used=[],
                )
            except Exception as save_error:
                logger.error(f"Failed to save error history: {save_error}")
            return {
                "text": error_msg,
                "json_format": None
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
                emit_to_user(user=user_id, message=ANALYZING_MSG)

            elif resource == "hypothesis":
                summary_result = self.hypothesis_generation.get_by_hypothesis_id(
                    token, graph_id, user_id, query
                )
                summary_text = summary_result.get('text', '') if isinstance(summary_result, dict) else summary_result
                emit_to_user(user=user_id, message=ANALYZING_MSG)
            else:
                return "Invalid resource type specified."
                
            # Return summary as dict for consistency
            return {"text": summary_text, "json_format": None}
            
        except Exception as e:
            logger.error("Error in answer_from_graph_summaries", exc_info=True)
            return {
                "text": f"Error processing query: {str(e)}",
                "json_format": None
            }