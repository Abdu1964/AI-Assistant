from typing import Dict, Any, Tuple, Optional, List, Union
from app.prompts.hypothesis_prompt import hypothesis_format_prompt, hypothesis_response
from app.storage.redis import redis_manager
from app.socket_manager import emit_to_user
import logging
import os
import difflib
import requests
import time

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

from typing import Dict, Any, Tuple, Optional, List, Union
from app.prompts.hypothesis_prompt import hypothesis_format_prompt, hypothesis_response
from app.storage.redis import redis_manager
from app.socket_manager import emit_to_user
import logging
import os
import difflib
import requests
import time

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Load API endpoints from environment variables
HYPOTHESIS_CHAT_ENDPOINT = os.getenv('HYPOTHESIS_CHAT_ENDPOINT')
HYPOTHESIS_MAIN_ENDPOINT = os.getenv('HYPOTHESIS_MAIN_ENDPOINT')
HYPOTHESIS_DATA_API = os.getenv('HYPOTHESIS_DATA_API')


class HypothesisGeneration:
    """
    Handles generation and processing of hypotheses based on user queries.
    Interacts with the Hypothesis API to generate hypotheses and retrieve related information.
    """

    def __init__(self, llm) -> None:
        """
        Initialize the HypothesisGeneration class with an LLM instance.
        
        Args:
            llm: Language model instance for generating formatted queries and responses
        """
        self.llm = llm
        logger.info("HypothesisGeneration initialized with LLM")

    def _make_api_request(self, 
                         method: str, 
                         url: str, 
                         token: str, 
                         params: Optional[Dict] = None, 
                         headers: Optional[Dict] = None,
                         data:Optional[Dict] = None) -> Dict[str, Any]:
        """
        Helper method to make API requests with proper error handling.
        """
        headers = {
            "Authorization": f"Bearer {token}"
        }
        try:
            logger.debug(f"Making {method} request to {url} with data {data} and params {params}")
            if data and method.upper() == "POST":
                # Use json=data to send application/json
                response = requests.post(url, json=data, headers=headers)    
            elif method.upper() == "GET":
                response = requests.get(url, params=params, headers=headers)
            elif method.upper() == "POST":
                response = requests.post(url, params=params, headers=headers)
            else:
                return {"error": f"Unsupported HTTP method: {method}"}
                
            response.raise_for_status()
            data = response.json()
            return data 
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {"error": f"Request failed Please Try Again"}

    def _validate_response(self, response: Dict[str, Any], required_keys: List[str] = []) -> Tuple[bool, str]:
        """
        Validate API response status and content.
        """
        if "error" in response:
             return False, response["error"]
        
        for key in required_keys:
            if key not in response:
                return False, f"Missing required key: {key}"
        
        return True, ""

    def _step_1_enrich(self, token: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Step 1: Start Enrichment"""
        logger.info(f"Step 1: Starting enrichment with params: {params}")
        # Assuming HYPOTHESIS_DATA_API points to /api/mock/hypothesis
        url = f"{HYPOTHESIS_DATA_API}/enrich"
        response = self._make_api_request("POST", url, token, data=params)
        
        valid, error = self._validate_response(response, required_keys=["hypothesis_id"])
        if not valid:
            return {"error": f"Enrichment start failed: {error}"}
            
        return response

    def _step_2_poll(self, token: str, hypothesis_id: str) -> Dict[str, Any]:
        """Step 2: Polling Loop with Retry Mechanism"""
        logger.info(f"Step 2: Polling status for hypothesis ID: {hypothesis_id}")
        
        # Retry configuration: 6 attempts * 10 seconds = 60 seconds max wait
        max_retries = 6
        retry_delay = 10 
        
        url = f"{HYPOTHESIS_DATA_API}/hypothesis"
        
        for attempt in range(max_retries):
            response = self._make_api_request("GET", url, token, params={"id": hypothesis_id})
            
            valid, error = self._validate_response(response, required_keys=["status"])
            if not valid:
                 return {"error": f"Status check failed: {error}"}
            
            status = response.get("status")
            logger.info(f"Polling attempt {attempt + 1}/{max_retries}: Status is '{status}'")

            if status == "completed":
                if "enrich_id" not in response:
                     return {"error": "Completed status but missing enrich_id"}
                return response
            elif status == "pending":
                if attempt < max_retries - 1:
                    logger.info(f"Status is pending. Waiting {retry_delay} seconds before retrying...")
                    time.sleep(retry_delay)
                    continue
                else:
                    return {"error": "Enrichment timed out after maximum retries"}
            else:
                return {"error": f"Unknown status: {status}"}
                
        return {"error": "Enrichment timed out"}

    def _step_3_get_results(self, token: str, enrich_id: str) -> Dict[str, Any]:
        """Step 3: Get Enrichment Results"""
        logger.info(f"Step 3: Fetching results for enrich ID: {enrich_id}")
        url = f"{HYPOTHESIS_DATA_API}/enrich"
        response = self._make_api_request("GET", url, token, params={"id": enrich_id})
        
        valid, error = self._validate_response(response, required_keys=["GO_terms", "causal_gene"])
        if not valid:
            return {"error": f"Result fetch failed: {error}"}
            
        return response

    def _step_4_generate(self, token: str, enrich_id: str, go_term_id: str) -> Dict[str, Any]:
        """Step 4: Generate Final Hypothesis"""
        logger.info(f"Step 4: Generating hypothesis for enrich ID: {enrich_id} and GO: {go_term_id}")
        url = f"{HYPOTHESIS_DATA_API}/hypothesis"
        response = self._make_api_request("POST", url, token, data={"id": enrich_id, "go": go_term_id})
        
        valid, error = self._validate_response(response, required_keys=["summary", "graph"])
        if not valid:
            return {"error": f"Final generation failed: {error}"}
            
        return response

    def get_by_hypothesis_id(self, token: str, hypothesis_id: str, user_id, query=None) -> Dict[str, Any]:
        """
        Retrieve hypothesis information by ID.
        """
        logger.info(f"Retrieving hypothesis by ID: {hypothesis_id}")
        emit_to_user(user=user_id,message=f"Retrieving hypothesis by ID: {hypothesis_id}")
        
        try:   
            if query: 
                emit_to_user(user=user_id,message=f"Processing query with existing hypothesis...")
                data = {
                    "query": query,
                    "hypothesis_id": hypothesis_id}
                headers = {
                    "Authorization": f"Bearer {token}"
                }
                # Use json=data
                response = requests.post(HYPOTHESIS_CHAT_ENDPOINT, json=data, headers=headers)
                response.raise_for_status()
                data = response.json()
                emit_to_user(user=user_id,message="Successfully processed query with hypothesis")
                return data
            else:
                cached_graph = redis_manager.get_graph_by_id(hypothesis_id)
                if cached_graph and cached_graph.get("graph_summary"):
                    logger.info(f"Cache hit for graph_id={hypothesis_id} {cached_graph}")
                    return {"text": cached_graph["graph_summary"]}

                data = {
                    "hypothesis_id": hypothesis_id
                }

                headers = {
                    "Authorization": f"Bearer {token}"
                }
                try:
                    # Use json=data
                    response = requests.post(HYPOTHESIS_CHAT_ENDPOINT, json=data, headers=headers)
                    response.raise_for_status()
                    data = response.json()
                    redis_manager.create_graph(graph_id=data['hypothesis_id'], graph_summary=data['summary'])
                    logger.info(f"Cached generated graph for graph id {data['resource']['id']}")
                    return data
                except Exception as e:
                    logger.error(f"Failed to retrieve hypothesis by ID: {response}")
                    emit_to_user(user=user_id,message=f"Failed to retrieve hypothesis")
                    return "NO summaries provided"
        except Exception as e:
            emit_to_user(user=user_id,message="Error retrieving hypothesis")
            return None

    def format_user_query(self, query: str, user_id) -> Dict[str, Any]:
        """
        Format user query using the LLM to extract relevant parameters.
        """
        logger.info(f"Formatting user query: {query}")
        
        try:
            prompt = hypothesis_format_prompt.format(question=query)
            response = self.llm.generate(prompt)
            
            if not response:
                logger.warning("LLM returned empty response for query formatting")
                emit_to_user(user=user_id,message="Warning: Empty response from query formatting")
            else:
                logger.info(f"Successfully formatted query with {len(response)} parameters")
                
            return response
        except Exception as e:
            logger.error(f"Error formatting user query: {str(e)}")
            emit_to_user(user=user_id,message=f"Error formatting query")
            return {}

    def generate_hypothesis(self, token: str, user_query: str, user_id: str) -> Dict[str, Any]:
        """
        Orchestrates the NLP-driven hypothesis generation workflow.
        """
        logger.info(f"Starting NLP-driven hypothesis generation for: {user_query}")
        emit_to_user(user=user_id, message="Analyzing your query...")

        # 1. NLP Extraction
        params = self.format_user_query(user_query, user_id)
        if not params:
            return {"text": "Could not understand the biological parameters in your query."}
        
        # Ensure we have the minimum required fields
        if "variant" not in params or "tissue_name" not in params:
             # Fallback or ask for clarification? For now, error.
             pass

        # Add default project_id if missing
        if "project_id" not in params:
            params["project_id"] = "project_auto_gen"

        # Step 1: Start Enrichment
        emit_to_user(user=user_id, message=f"Starting enrichment for {params.get('variant')}...")
        step1_res = self._step_1_enrich(token, params)
        if "error" in step1_res:
            logger.error(step1_res["error"])
            return {"text": f"I tried to start the enrichment, but failed: {step1_res['error']}"}
        
        hypothesis_id = step1_res["hypothesis_id"]

        # Step 2: Polling
        emit_to_user(user=user_id, message="Waiting for analysis to complete...")
        step2_res = self._step_2_poll(token, hypothesis_id)
        if "error" in step2_res:
            logger.error(step2_res["error"])
            return {"text": f"Enrichment started (ID: {hypothesis_id}), but failed during processing: {step2_res['error']}"}
        
        enrich_id = step2_res["enrich_id"]

        # Step 3: Get Results
        emit_to_user(user=user_id, message="Fetching enrichment results...")
        step3_res = self._step_3_get_results(token, enrich_id)
        if "error" in step3_res:
             logger.error(step3_res["error"])
             return {"text": f"Analysis completed, but failed to retrieve results: {step3_res['error']}"}
        
        # Select best GO term (Logic: Top Rank / Lowest P-value)
        go_terms = step3_res.get("GO_terms", [])
        if not go_terms:
             return {"text": "Analysis completed, but no significant GO terms were found."}
        
        # Sort by rank or p-value just to be safe, though API returns sorted
        # Mock API returns them. Let's pick the first one.
        best_go = go_terms[0]
        best_go_id = best_go["id"]
        best_go_name = best_go["name"]
        
        emit_to_user(user=user_id, message=f"Identified top mechanism: {best_go_name}")

        # Step 4: Final Generation
        emit_to_user(user=user_id, message="Generating final hypothesis...")
        step4_res = self._step_4_generate(token, enrich_id, best_go_id)
        if "error" in step4_res:
            logger.error(step4_res["error"])
            return {"text": f"Failed to generate final hypothesis summary: {step4_res['error']}"}

        # Success!
        summary = step4_res["summary"]
        graph = step4_res["graph"]
        
        # Cache result
        redis_manager.create_graph(graph_id=hypothesis_id, graph_summary=summary)

        return {
            "text": summary,
            "resource": {
                "id": hypothesis_id,
                "type": "hypothesis",
                "graph": graph # pass graph data if needed by frontend
            }
        }