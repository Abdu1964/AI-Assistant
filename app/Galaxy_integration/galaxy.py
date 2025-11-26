import asyncio
import logging
import traceback
from app.Galaxy_integration.galaxy_content_clean import HTMLProcessor
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import subprocess
import json
import tempfile

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GALAXY_MCP_SERVER = os.getenv("GALAXY_MCP_SERVER")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GalaxyHandler:
    def __init__(self, llm, qdrant_client=None, embedding_model=None):
        self.llm = llm
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.collection_name = "1_AI_ASSISTANT_GALAXY_DATASETS"
        logger.info("GalaxyHandler initialized")

    def get_galaxy_info(self, query, user_id, token, files=None):
        """Main entry point: returns text only for Flask"""
        logger.info(f"get_galaxy_info called with query='{query}', user_id='{user_id}', files={files}")
        try:
            if files and query:
                return self._handle_files(query=query, user_id=user_id, files=files)
            else:
                logger.info("No files provided, routing to MCP handler")
                return self._handle_mcp_sync(query, token)
        except Exception as e:
            logger.error(f"Galaxy handler failed: {e}")
            traceback.print_exc()
            return {"text": f"Error processing Galaxy request: {e}"}

    def _handle_files(self, query, user_id, files):
        """Handle file-based queries using RAG"""
        logger.info(f"_handle_files called with files={files}")
        if isinstance(files, str):
            files = [files]

        try:
            processor = HTMLProcessor(self.qdrant_client, self.llm)
            for f in files:
                logger.info(f"Searching for URL ID: {f}")
                stored = self.qdrant_client.retrieve_similar_content(
                    collection_name=self.collection_name,
                    query=query,
                    content_ids=[f],
                    filter=True
                )
                logger.info(f"Stored values: {stored}")
                if not stored:
                    new_chunks = processor.store_embedded(url=f, collection_name=self.collection_name)
                    logger.info(f"Document stored successfully: {new_chunks}")

            similar_results = self.qdrant_client.retrieve_similar_content(
                collection_name=self.collection_name,
                query=query,
                content_ids=files,
                top_k=5
            )

            if similar_results:
                context_text = "\n\n".join([chunk.get("text", "") for chunk in similar_results])
                llm_prompt = f"""
You are an expert AI assistant. You are given the following context extracted from documents:

{context_text}

User's query: "{query}"

Please provide a clear, professional, and concise answer to the user's query based solely on the context above.
- If the answer is not directly available, politely inform the user.
- Do not hallucinate information.
- Keep the response clear and concise.
"""
                response_text = self.llm.generate(llm_prompt)
            else:
                response_text = "I could not find any relevant information in the provided documents to answer your query."

            return {"text": response_text}

        except Exception as e:
            logger.error(f"Galaxy file analyzer failed: {e}")
            traceback.print_exc()
            return {"text": f"Error analyzing files: {e}"}

    def _handle_mcp_sync(self, query, token):
        """Sync wrapper - runs MCP in subprocess to avoid eventlet conflicts"""
        logger.info(f"_handle_mcp_sync called with query='{query}'")

        try:
            # Escape the query properly for the script
            query_escaped = query.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')

            # Create a temporary Python script that runs WITHOUT eventlet
            script_content = f'''
import asyncio
import json
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent

async def run_mcp():
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001")
    client = MultiServerMCPClient({{
        "galaxyTools": {{
            "transport": "streamable_http",
            "url": "{GALAXY_MCP_SERVER}",
            "headers": {{"Authorization": "Bearer {token}"}}
        }}
    }})
    query = "{query_escaped}"
    tools = await client.get_tools()
    agent = create_agent(model, tools)
    response = await agent.ainvoke({{"messages": query}})
    if isinstance(response, dict):
        output = response.get("output", str(response))
    else:
        output = str(response)
    return {{"text": output}}

if __name__ == "__main__":
    result = asyncio.run(run_mcp())
    print(json.dumps(result))
'''

            # Write script to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script_content)
                script_path = f.name

            try:
                # Run the script in a separate process (no eventlet!)
                logger.info(f"Running MCP in subprocess: {script_path}")
                result = subprocess.run(
                    ['python', script_path],
                    capture_output=True,
                    text=True,
                    timeout=120
                )

                if result.returncode == 0:
                    output = json.loads(result.stdout.strip())
                    resonse = self.llm.generate(f"clean it and Summarize the following response concisely:\n\n{output} for the user query {query}")
                    return resonse
                else:
                    logger.error(f"Subprocess failed: {result.stderr}")
                    raise Exception(f"Subprocess error: {result.stderr}")

            finally:
                try:
                    os.unlink(script_path)
                except:
                    pass

        except subprocess.TimeoutExpired:
            logger.error("MCP subprocess timed out")
            return {"text": "Request timed out after 120 seconds. Please try again."}

        except Exception as e:
            logger.error(f"MCP execution failed, falling back to RAG: {e}")
            traceback.print_exc()

            # Fallback to RAG-based approach
            try:
                from app.prompts.rag_prompts import RETRIEVE_PROMPT
                from qdrant_client import QdrantClient

                GALAXY_TOOLS_RECOMMEND_COLLECTION = os.getenv("GALAXY_TOOLS_RECOMMEND_COLLECTION")
                qdrant_url = os.getenv("galaxy_QDRANT_CLIENT")

                if not qdrant_url or not GALAXY_TOOLS_RECOMMEND_COLLECTION:
                    logger.error("Missing environment variables for RAG fallback")
                    return {"text": "Configuration error: Unable to process request"}

                client = QdrantClient(
                    url=qdrant_url,
                    port=6333,
                    grpc_port=6334,
                    prefer_grpc=False,
                )

                if not self.embedding_model:
                    logger.error("Embedding model not initialized")
                    return {"text": "Configuration error: Embedding model not available"}

                embedded_text = self.embedding_model(query)
                result = client.search(
                    collection_name=GALAXY_TOOLS_RECOMMEND_COLLECTION,
                    query_vector=embedded_text,
                    with_payload=True,
                    score_threshold=0.5,
                    limit=5
                )

                response = RETRIEVE_PROMPT.format(
                    query=query,
                    retrieved_content=result
                )
                return {"text": response}

            except Exception as fallback_error:
                logger.error(f"RAG fallback also failed: {fallback_error}")
                traceback.print_exc()
                return {"text": f"Error: Unable to process request. Please try again later."}
