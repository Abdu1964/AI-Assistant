
import asyncio
import logging
import traceback
from app.Galaxy_integration.galaxy_content_clean import HTMLProcessor
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
import os

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

    def get_galaxy_info(self, query, user_id, token,files=None):
        """Main entry point: returns text only for Flask"""
        logger.info(f"get_galaxy_info called with query='{query}', user_id='{user_id}', files={files}")
        try:
            if files and query:
                return self._handle_files(query=query, user_id=user_id, files=files)
            else:
                logger.info("No files provided, routing to MCP handler")
                return self._handle_mcp_sync(query,token)
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
            processor = HTMLProcessor(self.qdrant_client,self.llm)
            for f in files:
                logger.info(f"this are the url id to be searched {f}")
                stored = self.qdrant_client.retrieve_similar_content(
                    collection_name=self.collection_name,
                    query=query,
                    content_ids=[f],
                    filter=True
                )
                logger.info(f"stored values are {stored}")
                if not stored:
                    new_chunks = processor.store_embedded(url=f, collection_name=self.collection_name)
                    logger.info(f"Document has been stored succesfully {new_chunks}")
            # Retrieve relevant chunks
            similar_results = self.qdrant_client.retrieve_similar_content(
            collection_name=self.collection_name,
            query=query,
            content_ids=files,  # pass list of URLs
            top_k=5)

            if similar_results:
                context_text = "\n\n".join([chunk.get("text", "") for chunk in similar_results])
                llm_prompt = f"""
                You are an expert AI assistant. You are given the following context extracted from documents:

                {context_text}

        #         User's query: "{query}"

        #         Please provide a clear, professional, and concise answer to the user's query based solely on the context above. 
        #         - If the answer is not directly available, politely inform the user.
        #         - Do not hallucinate information.
        #         - Keep the response clear and concise.
        #         """
                response_text = self.llm.generate(llm_prompt)
            else:
                response_text = "I could not find any relevant information in the provided documents to answer your query."

            return {"text": response_text}

        except Exception as e:
            logger.error(f"Galaxy file analyzer failed: {e}")
            traceback.print_exc()
            return {"text": f"Error analyzing files: {e}"}

    async def _handle_mcp(self, query,token):
        """Handle MCP-based queries asynchronously and return only text"""
        logger.info(f"_handle_mcp called with query='{query}' and passing auth of the user {token}")
        try:

            client = MultiServerMCPClient({
                "galaxyTools": {
                    "transport": "streamable_http",
                    "url": GALAXY_MCP_SERVER,
                    "headers": {"Authorization":f"Bearer {token}"}
                }
            })

            tools = await client.get_tools()
            agent = create_react_agent("openai:gpt-4o", tools)
            response = await agent.ainvoke({"messages": query})

            logger.info(f"here is the answer from mcp {response}")

            # Step 2: ask LLM to extract clean answer from the combined response
            llm_prompt = f"""
            DONT WRITE ANY ANSWER ON YOUR OWN
            Extract only the final answer text from the following response. 
            Remove any metadata, formatting, or extra messages. get the answer only.
            for the user query {query}
            clean all the answers with clear response please
            if the answer is an apology for not being able to answer, 
            just say "I could not find enough information to answer your question."
            Response: {response}
            """
            clean_text = self.llm.generate(llm_prompt)

            return {"text": clean_text}

        except Exception as e:
            logger.error(f"MCP interaction failed: {e}")
            traceback.print_exc()
            return {"text": f"Error during MCP interaction: {e}"}

    def _handle_mcp_sync(self, query,token):
        """Sync wrapper for Flask + eventlet: safely returns text"""
        logger.info(f"_handle_mcp_sync called with query='{query}'")
        try:
            try:
                loop = asyncio.get_running_loop()
                future = asyncio.run_coroutine_threadsafe(self._handle_mcp(query,token), loop)
                return future.result()
            except RuntimeError:
                return asyncio.run(self._handle_mcp(query,token))
        except Exception as e:
            logger.error(f"MCP sync wrapper failed: {e}")
            traceback.print_exc()
            return {"text": f"Error in MCP sync wrapper: {e}"}


