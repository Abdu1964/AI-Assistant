import asyncio
import logging
import traceback
import os
import subprocess
import json
import tempfile

from app.Galaxy_integration.galaxy_content_clean import HTMLProcessor
from langchain_mcp_adapters.client import MultiServerMCPClient

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

GALAXY_MCP_SERVER = os.getenv("GALAXY_MCP_SERVER")
ADVANCED_LLM_PROVIDER = os.getenv("ADVANCED_LLM_PROVIDER", "gemini")


class GalaxyHandler:
    def __init__(self, llm, qdrant_client=None, embedding_model=None):
        self.llm = llm
        self.qdrant_client = qdrant_client
        self.embedding_model = embedding_model
        self.collection_name = "1_AI_ASSISTANT_GALAXY_DATASETS"
        logger.info(f"GalaxyHandler initialized with provider='{ADVANCED_LLM_PROVIDER}'")


    def get_galaxy_info(self, query: str, user_id: str, token: str, urls=None) -> dict:
        """Main entry point — returns a dict with key 'text'."""
        logger.info(f"get_galaxy_info called | query='{query}' | user_id='{user_id}' | urls={urls}")
        try:
            if urls and query:
                return self._handle_files(query=query, user_id=user_id, urls=urls)
            return self._handle_mcp(query, token)
        except Exception as e:
            logger.error(f"Galaxy handler failed: {e}")
            traceback.print_exc()
            return {"text": f"Error processing Galaxy request: {e}"}


    def _handle_files(self, query: str, user_id: str, urls) -> dict:
        """Handle file-based queries using RAG — supports multiple URLs."""
        logger.info(f"_handle_files called | urls={urls}")

        if isinstance(urls, str):
            urls = [urls]
        if not urls:
            return {"text": "No URLs provided for analysis."}

        try:
            processor = HTMLProcessor(self.qdrant_client, self.llm)

            collection_exists = False
            try:
                info = self.qdrant_client.client.get_collection(self.collection_name)
                collection_exists = True
                logger.info(f"Collection '{self.collection_name}' exists ({info.points_count} points)")
            except Exception:
                logger.info(f"Collection '{self.collection_name}' does not exist yet")

            urls_to_process = []
            if collection_exists:
                for url in urls:
                    try:
                        stored = self.qdrant_client.retrieve_similar_content(
                            collection_name=self.collection_name,
                            query=query,
                            content_ids=[url],
                            filter=True,
                        )
                        if not stored:
                            logger.info(f"URL not in collection, will process: {url}")
                            urls_to_process.append(url)
                        else:
                            logger.info(f"URL already in collection: {url}")
                    except Exception as e:
                        logger.warning(f"Error checking URL {url}: {e}")
                        urls_to_process.append(url)
            else:
                urls_to_process = list(urls)

            if urls_to_process:
                logger.info(f"Processing {len(urls_to_process)} new URL(s)")
                storage_results = processor.store_embedded(
                    urls=urls_to_process,
                    collection_name=self.collection_name,
                )
                successful = [u for u, r in storage_results.items() if "Successfully processed" in r]
                if not successful:
                    reasons = "\n".join(f"{u}: {r}" for u, r in storage_results.items())
                    return {"text": f"Failed to extract content from provided documents:\n{reasons}"}
            else:
                logger.info("All URLs already exist in collection")

            try:
                similar_results = self.qdrant_client.retrieve_similar_content(
                    collection_name=self.collection_name,
                    query=query,
                    content_ids=urls,
                    top_k=10,
                )
            except Exception as e:
                logger.error(f"Error retrieving similar content: {e}")
                return {"text": f"Error retrieving content: {e}"}

            if not similar_results:
                logger.warning("No relevant chunks found in collection")
                return {"text": f"I could not find relevant information in the provided {len(urls)} document(s)."}

            logger.info(f"Found {len(similar_results)} relevant chunk(s)")
            results_by_url: dict[str, list] = {}
            for chunk in similar_results:
                url = chunk.get("content_id", "Unknown")
                results_by_url.setdefault(url, []).append(chunk)

            context_text = "\n\n".join(
                f"--- From {url} ---\n" + "\n\n".join(str(c.get("text", "")) for c in chunks)
                for url, chunks in results_by_url.items()
            )

            prompt = f"""
You are an expert AI assistant. You are given context extracted from {len(urls)} document(s):

{context_text}

User's query: "{query}"

Provide a clear, professional, and concise answer based solely on the context above.
- Synthesise information from multiple documents where relevant.
- If the answer is not available, politely inform the user.
- Do not hallucinate. Mention source document(s) where helpful.
"""
            response_text = self.llm.generate(prompt)
            return {"text": response_text}

        except Exception as e:
            logger.error(f"Galaxy file analyser failed: {e}")
            traceback.print_exc()
            return {"text": f"Error analysing URLs: {e}"}


    def _handle_mcp(self, query: str, token: str) -> dict:
        """Route to the correct MCP execution strategy based on the active LLM provider."""
        provider = ADVANCED_LLM_PROVIDER

        if provider == "openai":
            logger.info("MCP path: asyncio (OpenAI)")
            return self._handle_mcp_async(query, token)

        if provider in ("gemini", "ollama"):
            logger.info(f"MCP path: subprocess ({provider} — avoids async/eventlet conflicts)")
            return self._handle_mcp_subprocess(query, token, provider)

        logger.warning(f"Unknown provider '{provider}', falling back to RAG")
        return self._rag_fallback(query)


    def _handle_mcp_subprocess(self, query: str, token: str, provider: str) -> dict:
        """
        Runs MCP agent in a child process so eventlet/gevent loops don't interfere.
        Supports gemini and ollama providers.
        """
        logger.info(f"_handle_mcp_subprocess | provider='{provider}' | query='{query}'")

        # Escape the query for safe embedding in the generated script
        query_escaped = query.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

        if provider == "gemini":
            agent_setup = """\
from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
"""
        elif provider == "ollama":
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
            ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:14b")
            agent_setup = f"""\
import openai as _openai
from langchain_openai import ChatOpenAI
model = ChatOpenAI(
    model="{ollama_model}",
    base_url="{ollama_host}/v1",
    api_key="ollama",
    temperature=0,
)
"""
        else:
            return {"text": f"Unsupported provider for subprocess path: {provider}"}

        script = f"""\
import asyncio, json
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
{agent_setup}

async def run():
    client = MultiServerMCPClient({{
        "galaxyTools": {{
            "transport": "streamable_http",
            "url": "{GALAXY_MCP_SERVER}",
            "headers": {{"Authorization": "Bearer {token}"}},
        }}
    }})
    tools = await client.get_tools()
    agent = create_react_agent(model, tools)
    response = await agent.ainvoke({{"messages": "{query_escaped}"}})

    messages = response.get("messages", [])
    output = ""
    for msg in reversed(messages):
        content = getattr(msg, "content", None) or msg.get("content", "")
        if content and isinstance(content, str):
            output = content
            break
    print(json.dumps({{"text": output or str(response)}}))

asyncio.run(run())
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            script_path = f.name

        try:
            result = subprocess.run(
                ["python", script_path],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                raw_output = json.loads(result.stdout.strip())
                logger.info(f"Subprocess completed: {raw_output}")
                # Let the main LLM clean up / summarise the agent output
                summary = self.llm.generate(
                    f"Clean and summarise the following response concisely for the user query: {query}\n\n{raw_output}"
                )
                return {"text": summary}
            else:
                logger.error(f"Subprocess failed:\n{result.stderr}")
                raise RuntimeError(f"Subprocess error: {result.stderr}")

        except subprocess.TimeoutExpired:
            logger.error("MCP subprocess timed out")
            return {"text": "Request timed out after 120 seconds. Please try again."}

        except Exception as e:
            logger.error(f"Subprocess MCP failed, falling back to RAG: {e}")
            traceback.print_exc()
            return self._rag_fallback(query)

        finally:
            try:
                os.unlink(script_path)
            except Exception:
                pass


    def _handle_mcp_async(self, query: str, token: str) -> dict:
        """Run MCP directly via asyncio — works fine with OpenAI (no eventlet)."""
        try:
            return asyncio.run(self._run_mcp_openai(query, token))
        except Exception as e:
            logger.error(f"Async MCP failed: {e}")
            traceback.print_exc()
            return self._rag_fallback(query)

    async def _run_mcp_openai(self, query: str, token: str) -> dict:
        from langgraph.prebuilt import create_react_agent

        client = MultiServerMCPClient({
            "galaxyTools": {
                "transport": "streamable_http",
                "url": GALAXY_MCP_SERVER,
                "headers": {"Authorization": f"Bearer {token}"},
            }
        })

        tools = await client.get_tools()
        agent = create_react_agent("openai:gpt-4o", tools)
        response = await agent.ainvoke({"messages": query})
        logger.info(f"OpenAI MCP raw response: {response}")

        messages = response.get("messages", [])
        output = ""
        for msg in reversed(messages):
            content = getattr(msg, "content", None) or msg.get("content", "")
            if content and isinstance(content, str):
                output = content
                break

        return {"text": output or str(response)}

   

    def _rag_fallback(self, query: str) -> dict:
        """Falls back to Qdrant vector search when MCP is unavailable."""
        logger.info("Attempting RAG fallback")
        try:
            from app.prompts.rag_prompts import RETRIEVE_PROMPT
            from qdrant_client import QdrantClient

            collection = os.getenv("GALAXY_TOOLS_RECOMMEND_COLLECTION")
            qdrant_url = os.getenv("galaxy_QDRANT_CLIENT")

            if not qdrant_url or not collection:
                logger.error("Missing env vars for RAG fallback")
                return {"text": "Configuration error: unable to process request."}

            if not self.embedding_model:
                logger.error("Embedding model not initialised")
                return {"text": "Configuration error: embedding model not available."}

            client = QdrantClient(url=qdrant_url, port=6333, grpc_port=6334, prefer_grpc=False)
            embedded = self.embedding_model(query)
            results = client.search(
                collection_name=collection,
                query_vector=embedded,
                with_payload=True,
                score_threshold=0.5,
                limit=5,
            )
            response = RETRIEVE_PROMPT.format(query=query, retrieved_content=results)
            return {"text": response}

        except Exception as e:
            logger.error(f"RAG fallback also failed: {e}")
            traceback.print_exc()
            return {"text": "Error: unable to process request. Please try again later."}