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
from .prompts.classifier_prompt import classifier_prompt, answer_from_graph
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

logger = logging.getLogger(__name__)
log_dir = "/AI-Assistant/logfiles"
log_file = os.path.join(log_dir, "Assistant.log")
# os.makedirs(log_dir, exist_ok=True)
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
logger = logging.getLogger(__name__)
load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    user_query: str
    user_id: str
    token: str
    query_type: str
    response: str
    error: str
    content_ids: Optional[List[str]]


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
        self.galaxy_handler = GalaxyHandler(advanced_llm,qdrant_client,embedding_model)
        self.embedding_model = embedding_model

        logger.info(
            f"AiAssistance initialized with advanced_llm: {type(self.advanced_llm).__name__}"
        )
        logger.info(f"Galaxy handler initialized: {type(self.galaxy_handler).__name__}")

        # Initialize the LangGraph workflow
        self.workflow = self._create_workflow()
        self.app = self.workflow.compile()

    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""

        logger.info("Creating LangGraph workflow with tools and nodes")

        # Define tools
        @tool
        def get_json_format(query: str, token: str) -> str:
            """Retrieve the json format provided from the annotation graph tool"""
            logger.info(f"get_json_format called with query: {query}")
            try:
                logger.info(f"Generating graph with arguments: {query}")
                response = self.annotation_graph.validated_json(query)
                return response
            except Exception as e:
                logger.error("Error in generating graph", exc_info=True)
                return f"I couldn't generate a graph for the given question {query} please try again."

        @tool
        def get_general_response(query: str, user_id: str) -> str:
            """Retrieve information for general knowledge queries."""
            logger.info(
                f"get_general_response called with query: {query}, user_id: {user_id}"
            )
            try:
                response = self.rag.get_result_from_rag(query, user_id)
                return response
            except Exception as e:
                logger.error("Error in retrieving response", exc_info=True)
                return "Error in retrieving response."

        # @tool
        # def hypothesis_generation(query: str, token: str) -> str:
        #     """Generation of hypothesis for biological mechanisms"""
        #     logger.info(f"hypothesis_generation called with query: {query}")
        #     try:
        #         logger.info(f"Here is the user query passed to the agent {query}")
        #         response = self.hypothesis_generation.generate_hypothesis(
        #             token=token, user_query=query
        #         )
        #         return response
        #     except Exception as e:
        #         logger.error("Error in hypothesis generation", exc_info=True)
        #         traceback.print_exc()
        #         return "Error in generating hypothesis."

        @tool
        def get_galaxy_tools(query: str, user_id: str,token) -> str:
            """Retrieve information about Galaxy web tools and workflows."""
            logger.info(
                f"get_galaxy_tools called with query: {query}, user_id: {user_id}"
            )
            try:
                # You'll need to implement this method in your galaxy handler
                response = self.galaxy_handler.get_galaxy_info(query, user_id,token)
                return response
            except Exception as e:
                logger.error("Error in galaxy tools retrieval", exc_info=True)
                return "Error retrieving Galaxy tools information."

        self.tools = [
            get_json_format,
            get_general_response,
            # hypothesis_generation,
            get_galaxy_tools,
        ]
        # Create workflow
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("classifier", self._classify_query)
        workflow.add_node("annotation_agent", self._annotation_agent)
        # workflow.add_node("hypothesis_agent", self._hypothesis_agent)
        workflow.add_node("rag_agent", self._rag_agent)
        workflow.add_node("galaxy_agent", self._galaxy_agent)
        workflow.add_node("finalizer", self._finalize_response)

        # Define edges
        workflow.set_entry_point("classifier")

        workflow.add_conditional_edges(
            "classifier",
            self._route_query,
            {
                "annotation": "annotation_agent",
                # "hypothesis": "hypothesis_agent",
                "rag": "rag_agent",
                "galaxy": "galaxy_agent",
                "error": "finalizer",
            },
        )

        workflow.add_edge("annotation_agent", "finalizer")
        # workflow.add_edge("hypothesis_agent", "finalizer")
        workflow.add_edge("rag_agent", "finalizer")
        workflow.add_edge("galaxy_agent", "finalizer")
        workflow.add_edge("finalizer", END)

        return workflow

    def get_content_summaries(self, user_id, content_ids=None):
        # Get summaries for all content types (PDF and web)
        content_summaries = []

        # Get all content files for the user
        all_content = self.store.get_user_content_files(user_id)

        if content_ids:
            # Filter by specific content IDs
            filtered_content = [
                content
                for content in all_content
                if content.get("content_id") in content_ids
            ]
        else:
            # Get all content
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
        query = state["user_query"]
        user_id = state["user_id"]
        token = state["token"]
        content_ids = state.get("content_ids")

        # Fetch content summaries using helper
        content_summaries = self.get_content_summaries(user_id, content_ids)
        # include web context in the prompt for better classification
        web_urls = SimpleWebSearch().get_context_urls(query, num_results=2)
        web_context = f"Web context: {', '.join(web_urls)}" if web_urls else ""

        logger.info(f"Classifying query: {query}")
        classifier_prompt = f"""Classify this query into one of these categories:
        - annotation: Requests for factual information about genes, proteins, variants, or biological graphs/networks
        - galaxy: Requests about Galaxy web tools, workflows, or Galaxy platform capabilities recommending of tools conversions 
        - rag: General information requests, including queries about uploaded PDFs, web content, or document profiles (e.g., questions about content summaries, metadata, or content)
        
        User query: {query}
        Content summaries: {content_summaries}
        {web_context}
        
        Respond ONLY with the category name."
        """

        response = self.advanced_llm.generate(classifier_prompt).lower()
        query_type = response.split()[0]  # Take first word in case LLM adds explanation

        logger.info(f"Query classified as: {query_type}")

        return {
            "query_type": query_type,
            "messages": [HumanMessage(content=f"Query classified as: {query_type}")],
        }

    def _route_query(self, state: AgentState) -> str:
        """Route query based on classification"""
        route = state.get("query_type", "rag")
        logger.info(f"Routing query to: {route}")
        return route

    def _annotation_agent(self, state: AgentState) -> Dict[str, Any]:
        """Handle annotation-related queries"""
        logger.info(
            f"Annotation agent processing query: {state['user_query']} for user: {state['user_id']}"
        )
        try:
            emit_to_user(
                user=state["user_id"], message="Creating Query Builder Format..."
            )
            # Use the annotation graph tool
            response = self.annotation_graph.validated_json(
                state["user_query"], user_id=state["user_id"]
            )

            return {
                "response": response,
                "messages": [
                    AIMessage(content=f"Annotation query processed: {response}")
                ],
            }
        except Exception as e:
            logger.error("Error in annotation agent", exc_info=True)
            return {
                "response": f"Error processing annotation query: {str(e)}",
                "error": str(e),
                "messages": [
                    AIMessage(content=f"Error in annotation processing: {str(e)}")
                ],
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

    #         return {
    #             "response": response,
    #             "messages": [AIMessage(content=f"Hypothesis generated: {response}")],
    #         }
    #     except Exception as e:
    #         logger.error("Error in hypothesis agent", exc_info=True)
    #         return {
    #             "response": f"Error generating hypothesis: {str(e)}",
    #             "error": str(e),
    #             "messages": [
    #                 AIMessage(content=f"Error in hypothesis generation: {str(e)}")
    #             ],
    #         }

    def _rag_agent(self, state: AgentState) -> Dict[str, Any]:
        """Handle general information queries"""
        logger.info(
            f"RAG agent processing query: {state['user_query']} for user: {state['user_id']} with content_ids: {state.get('content_ids')}"
        )
        try:
            emit_to_user(user=state["user_id"], message="Retrieving information...")
            response = self.rag.get_result_from_rag(
                state["user_query"],
                state["user_id"],
                content_ids=state.get("content_ids"),
            )

            # Extract the text from the RAG response
            if response and isinstance(response, dict) and "text" in response:
                response_text = response["text"]
            else:
                response_text = str(response) if response else "No response generated"

            return {
                "response": response_text,
                "messages": [
                    AIMessage(content=f"RAG query processed: {response_text}")
                ],
            }
        except Exception as e:
            logger.error("Error in RAG agent", exc_info=True)
            return {
                "response": f"Error retrieving information: {str(e)}",
                "error": str(e),
                "messages": [AIMessage(content=f"Error in RAG processing: {str(e)}")],
            }

    def _galaxy_agent(self, state: AgentState) -> Dict[str, Any]:
        """Handle Galaxy tools and workflows queries"""
        logger.info(
            f"Galaxy agent processing query: {state['user_query']} for user: {state['user_id']}"
        )
        try:
            emit_to_user(
                user=state["user_id"], message="Retrieving Galaxy tools information..."
            )
            response = self.galaxy_handler.get_galaxy_info(
                state["user_query"], state["user_id"],state["token"]
            )

            return {
                "response": response,
                "messages": [AIMessage(content=f"Galaxy query processed: {response}")],
            }
        except Exception as e:
            logger.error("Error in galaxy agent", exc_info=True)
            return {
                "response": f"Error retrieving Galaxy information: {str(e)}",
                "error": str(e),
                "messages": [
                    AIMessage(content=f"Error in Galaxy processing: {str(e)}")
                ],
            }

    def _finalize_response(self, state: AgentState) -> Dict[str, Any]:
        """Finalize and return the response"""
        response = state.get("response", "No response generated")
        logger.info(
            f"Finalizing response for user: {state.get('user_id')}, response length: {len(str(response))}"
        )

        return {"messages": [AIMessage(content=f"Final response: {response}")]}

    def agent(
        self,
        message: str,
        user_id: str,
        token: str,
        content_ids: Optional[List[str]] = None,
    ) -> str:
        """Main entry point for processing queries"""
        logger.info(
            f"Agent called with message: {message}, user_id: {user_id}, content_ids: {content_ids}"
        )
        try:
            # Create initial state
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "user_query": message,
                "user_id": user_id,
                "token": token,
                "query_type": "",
                "response": "",
                "error": "",
                "content_ids": content_ids,
            }

            # Run the workflow
            result = self.app.invoke(initial_state)

            # Extract response
            response = result.get("response", "")
            if response:
                return response

            # Fallback to last message content
            if result.get("messages"):
                last_message = result["messages"][-1]
                if hasattr(last_message, "content"):
                    return last_message.content

            return "No response generated"

        except Exception as e:
            logger.error("Error in agent processing", exc_info=True)
            return f"Error processing query: {str(e)}"

    def answer_from_graph_summaries(self, query, user_id, resource, token, graph_id):
        logger.info(
            f"Answer from graph summaries called with query: {query}, user_id: {user_id}, resource: {resource}, graph_id: {graph_id}"
        )
        summary = None
        try:
            if resource == "annotation":
                summary_result = self.graph_summarizer.summary(token=token, graph_id=graph_id, user_query=query)
                # Extract text from dictionary if needed
                summary_text = summary_result.get('text', '') if isinstance(summary_result, dict) else summary_result
                emit_to_user(user=user_id, message="Analyzing...")

            elif resource == "hypothesis":
                summary_result = self.hypothesis_generation.get_by_hypothesis_id(
                    token, graph_id, user_id, query
                )
                # Extract text from dictionary if needed  
                summary_text = summary_result.get('text', '') if isinstance(summary_result, dict) else summary_result
                emit_to_user(user=user_id, message="Analyzing...")

            else:
                logger.error(f"Unsupported resource type: '{resource}'")
                return None

            if query and summary_text:
                prompt = classifier_prompt.format(query=query, graph_summary=summary_text)
                response = self.advanced_llm.generate(prompt)
                
                # Log the raw response for debugging
                logger.info(f"Classifier raw response: {response}")

                # Handle Gemini's verbose responses - look for "related:" anywhere in the response
                if "related:" in response:
                    logger.info("Query is related with the graph")
                    # Extract everything after "related:" and clean it up
                    related_part = response.split("related:", 1)[1].strip()
                    # Remove any trailing explanatory text that might come after the answer
                    query_response = related_part.split('\n')[0].strip()
                    
                    self.history.create_history(user_id, query, query_response)
                    logger.info(f"User query: {query}, Response: {query_response}")
                    return query_response

                # Check for "not" responses (case insensitive)
                if "not" in response.lower():
                    logger.info("Query is not related to the graph")
                    return None
                    
            # Fallback: return raw summary if available
            if summary_text:
                return summary_text
                
            return None

        except Exception as e:
            logger.error(f"Error in answer_from_graph_summaries: {e}", exc_info=True)
            return None

    def assistant(
        self,
        query,
        user_id: str,
        token: str,
        graph_summary=None,
        response_from_content=None,
        files_response=None,
        content_ids=None,
        ):
        
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
            graph_summary=graph_summary,
            response_from_content=response_from_content,
            files_response=files_response,
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
                    graph_id_referenced=graph_summary,
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
                    user_id, query, assistant_answer, graph_id_referenced=graph_summary
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


    def assistant_response(self, query, user_id, token, graph=None, graph_id=None, 
                      files=None, resource=None, json_query=None, content_ids=None):
        """Simple routing logic for queries"""
        
        try:
            logger.info(
                f"Assistant response called with query={query}, user_id={user_id}, resource={resource}, graph_id={graph_id}, content_ids={content_ids}, files={files}")

            if not any([graph_id, files, content_ids, resource]):
                response = self.assistant(query=query, user_id=user_id, token=token)
                self.history.create_history(user_id, query, response)
                return response
            
            if not graph_id:
                if resource == "galaxy" and files or files:
                    response = self.galaxy_handler.get_galaxy_info(query, user_id,token,files)
                elif resource == "content" and content_ids or content_ids:
                    response = self.rag.get_result_from_rag(query, user_id, content_ids)
                else:
                    response = self.assistant(query=query, user_id=user_id, token=token)
                
                self.history.create_history(user_id, query, response)
                return response
            
            graph_summary = None
            if graph_id:
                graph_summary = self.graph_summarizer.summary(token=token, graph_id=graph_id, user_query=query)
            
            files_response = None
            content_response = None
            
            if files:
                files_response = self.galaxy_handler.get_galaxy_info(query, user_id, files,token)
            if content_ids:
                content_response = self.rag.get_result_from_rag(query, user_id, content_ids)
            
            # Check if graph summary alone can answer
            if graph_summary and self._can_graph_answer(query, graph_summary):
                self.history.create_history(user_id, query, graph_summary)
                return {"text":graph_summary["text"]}
            
            # Use full assistant with all context
            response = self.assistant(
                query=query, user_id=user_id, token=token,
                graph_summary=graph_summary,
                response_from_content=content_response,
                files_response=files_response,
                content_ids=content_ids
            )
            
            self.history.create_history(user_id, query, response)
            return response
            
        except Exception as e:
            logger.error(f"Error: {e}")
            return {"text": "Error processing request."}

    def _can_graph_answer(self, query, graph_summary):
        """Quick check if graph summary can answer the query"""
        try:
            prompt = f"""
            Query: "{query}"
            Graph Summary: "{graph_summary}"
            
            If this question is asking ANYTHING about the graph/network/data  
            then the graph summary MUST be able to provide some answer - even if it's "none", "zero", "not present", or "unknown".
            
            Questions about graphs should ALWAYS be answerable from graph data.
            
            Is this question about the graph/network? Answer only: yes or no
            """
            response = self.advanced_llm.generate(prompt)
            return 'yes' in response.lower()
        except:
            import traceback
            traceback.print_exc()
            return False