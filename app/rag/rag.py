from app.prompts.rag_prompts import RETRIEVE_PROMPT
from app.storage.memory_layer import MemoryManager
from app.storage.history_manager import HistoryManager
import traceback
import os
import logging
import uuid
from datetime import datetime
import fitz
from app.rag.utils.content_processor import ContentProcessor
from app.rag.utils.content_analyzer import ContentAnalyzer
from app.storage.mongo_storage import mongo_db_manager
from app.rag.utils.web_search import SimpleWebSearch


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION", "SITE_INFORMATION")
USER_COLLECTION = os.getenv("USER_COLLECTION", "CHAT_MEMORY")
CONTENT_LIMIT = 10  # Total content limit (PDFs + web content)


class RAG:
    def __init__(self, llm, qdrant_client):
        """
        Initializes the RAG (Retrieval Augmented Generation) class.
        Uses the provided Qdrant client
        :param llm: An instance of the LLMInterface for generating responses.
        :param qdrant_client: The shared Qdrant client.
        """
        self.llm = llm
        self.client = qdrant_client
        self.content_processor = ContentProcessor()
        self.content_analyzer = ContentAnalyzer()
        logger.info(
            "RAG initialized with LLM and shared Qdrant client/embedding model."
        )

    def save_doc_to_rag(
        self,
        data,
        collection_name=None,
        is_content=False,
        pdf_path=None,
        file_name=None,
        user_id=None,
        content_id=None,
        is_web=False,
        web_content=None,
    ):
        """
        Unified method to save documents to RAG storage using the unified upsert_data method.

        :param data: The data to store (list of dicts for sample data, or None for content)
        :param collection_name: The collection name to store in
        :param is_content: Boolean indicating if this is content data (PDF/web)
        :param pdf_path: Path to PDF file (only for PDF data)
        :param file_name: Name of the file (only for PDF data)
        :param user_id: User ID (only for content data)
        :param is_web: Boolean indicating if this is web content
        :param web_content: Web content data (only for web data)
        :param content_id: Content ID (for content data)
        """
        if is_content and not is_web:
            if pdf_path and file_name and user_id and content_id:
                chunks = self.content_processor.process_pdf(pdf_path)
                metadata = {
                    "content_id": content_id,
                    "filename": file_name,
                    "user_id": user_id,
                    "content_type": "pdf",
                }
                return self.client.upsert_data(
                    collection_name=collection_name,
                    data=None,
                    is_content=True,
                    chunks=chunks,
                    metadata=metadata,
                )
            else:
                logger.error("Missing required parameters for PDF processing")
                return None
        elif is_web:
            if web_content and user_id and content_id:
                result = self.content_processor.process_web_content(
                    web_content.get("metadata", {}).get("url", "")
                )
                if not result:
                    logger.error("Failed to process web content")
                    return None

                metadata = {
                    "content_id": content_id,
                    "url": web_content.get("metadata", {}).get("url", ""),
                    "title": web_content.get("metadata", {}).get("title", ""),
                    "user_id": user_id,
                    "content_type": "web",
                }
                return self.client.upsert_data(
                    collection_name=collection_name,
                    data=None,
                    is_content=True,
                    chunks=result["chunks"],
                    metadata=metadata,
                )
            else:
                logger.error("Missing required parameters for web content processing")
                return None
        else:
            # Handle sample/general data using unified upsert_data method
            if isinstance(data, list) and all(isinstance(d, dict) for d in data):
                # Pass the list of dicts directly to upsert_data
                return self.client.upsert_data(
                    collection_name=collection_name,
                    data=data,
                    is_content=False,
                )
            else:
                logger.error("Invalid data format for sample data storage")
                return None

    def save_retrievable_docs(self, file, user_id):
        try:
            return_response = {"text": None, "resource": {}}

            # Check for duplicate files
            pdf_files = mongo_db_manager.get_user_content_files(user_id, "pdf")
            if any(f.get("filename") == file.filename for f in pdf_files):
                return_response["text"] = "PDF already exists."
                return_response["resource"]["filename"] = file.filename
                return return_response

            # Check quota
            if mongo_db_manager.get_content_count(user_id) >= CONTENT_LIMIT:
                return_response["text"] = (
                    "Your quota is full. Maximum 10 content items allowed."
                )
                return_response["resource"]["count"] = (
                    mongo_db_manager.get_content_count(user_id)
                )
                return return_response

            content_id = str(uuid.uuid4())
            upload_folder = "storage/pdfs"
            os.makedirs(upload_folder, exist_ok=True)
            pdf_path = os.path.join(upload_folder, f"{content_id}.pdf")
            file.save(pdf_path)

            # Get number of pages
            try:
                with fitz.open(pdf_path) as doc:
                    num_pages = doc.page_count
            except Exception:
                num_pages = None

            # Get upload time
            upload_time = datetime.now()

            # Get file size in MB
            try:
                file_size_bytes = os.path.getsize(pdf_path)
                file_size = round(file_size_bytes / (1024 * 1024), 2)
            except Exception:
                file_size = None

            full_text = self.content_processor.extract_text_from_pdf(pdf_path)
            analysis = self.content_analyzer.analyze_content(full_text, "pdf")

            file_analysis = {
                "content_id": content_id,
                "filename": file.filename,
                "keywords": analysis.get("keywords", ""),
                "topics": analysis.get("topics", ""),
                "summary": analysis.get("summary", ""),
                "suggested_questions": analysis.get("suggested_questions", ""),
                "num_pages": num_pages,
                "file_size": f"{file_size} MB",
                "upload_time": upload_time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Store in Qdrant using custom chunking logic
            self.save_doc_to_rag(
                data=None,
                collection_name=user_id,
                is_content=True,
                pdf_path=pdf_path,
                file_name=file.filename,
                user_id=user_id,
                content_id=content_id,
            )

            # Add PDF metadata to the database using unified table
            mongo_db_manager.add_content_file(
                user_id=user_id,
                content_id=content_id,
                content_type="pdf",
                filename=file.filename,
                num_pages=num_pages,
                file_size=file_size,
                upload_time=upload_time,
                summary=analysis.get("summary"),
                keywords=str(analysis.get("keywords", [])),
                topics=str(analysis.get("topics", [])),
                suggested_questions=str(analysis.get("suggested_questions", [])),
            )

            # Add memory for the upload
            MemoryManager(self.llm).add_memory(f"pdf file : {file.filename}", user_id)

            # Add a history entry for the PDF upload
            HistoryManager().create_history(
                user_id=user_id,
                user_message=f"Uploaded PDF: {file.filename}",
                assistant_answer=f"PDF '{file.filename}' uploaded successfully.",
            )

            return_response["text"] = "PDF uploaded successfully."
            return_response["resource"] = file_analysis
            return return_response
        except Exception as e:
            logger.error(f"Error in save_retrievable_docs: {e}")
            import traceback

            traceback.print_exc()
            return {"text": f"Error uploading PDF: {str(e)}"}

    def save_web_content(self, url, user_id):
        try:
            return_response = {"text": None, "resource": {}}

            # Validate URL
            is_valid, error_msg = self.content_processor.validate_url(url)
            if not is_valid:
                return_response["text"] = f"Invalid URL: {error_msg}"
                return return_response

            # Check for duplicate URLs
            content_files = mongo_db_manager.get_user_content_files(user_id, "web")
            if any(f.get("url") == url for f in content_files):
                return_response["text"] = "URL already exists."
                return_response["resource"]["url"] = url
                return return_response

            # Check quota
            if mongo_db_manager.get_content_count(user_id) >= CONTENT_LIMIT:
                return_response["text"] = (
                    "Your quota is full. Maximum 10 content items allowed."
                )
                return_response["resource"]["count"] = (
                    mongo_db_manager.get_content_count(user_id)
                )
                return return_response

            content_id = str(uuid.uuid4())
            upload_time = datetime.now()

            web_content = self.content_processor.extract_text_from_url(url)
            if not web_content:
                return_response["text"] = "Failed to extract content from URL."
                return return_response

            cleaned_text = self.content_processor.clean_text_content(
                web_content["text"]
            )
            analysis = self.content_analyzer.analyze_content(cleaned_text, "web")

            web_analysis = {
                "content_id": content_id,
                "url": url,
                "title": web_content["metadata"].get("title", "No title") or "No title",
                "author": web_content["metadata"].get("author", "Unknown") or "Unknown",
                "keywords": analysis.get("keywords", ""),
                "topics": analysis.get("topics", ""),
                "summary": analysis.get("summary", ""),
                "suggested_questions": analysis.get("suggested_questions", ""),
                "upload_time": upload_time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Store in Qdrant using custom chunking logic
            self.save_doc_to_rag(
                data=None,
                collection_name=user_id,
                is_web=True,
                web_content=web_content,
                user_id=user_id,
                content_id=content_id,
            )

            # Add web content metadata to the database
            mongo_db_manager.add_content_file(
                user_id=user_id,
                content_id=content_id,
                content_type="web",
                url=url,
                title=web_content["metadata"].get("title") or None,
                author=web_content["metadata"].get("author") or None,
                publish_date=None,
                file_size=None,
                upload_time=upload_time,
                summary=analysis.get("summary"),
                keywords=str(analysis.get("keywords", [])),
                topics=str(analysis.get("topics", [])),
                suggested_questions=str(analysis.get("suggested_questions", [])),
            )

            # Add memory for the upload
            MemoryManager(self.llm).add_memory(f"web content : {url}", user_id)

            # Add a history entry for the web content upload
            HistoryManager().create_history(
                user_id=user_id,
                user_message=f"Added web content: {url}",
                assistant_answer=f"Web content from '{url}' added successfully.",
            )

            return_response["text"] = "Web content added successfully."
            return_response["resource"] = web_analysis
            return return_response
        except Exception as e:
            logger.error(f"Error in save_web_content: {e}")
            import traceback

            traceback.print_exc()
            return {"text": f"Error adding web content: {str(e)}"}

    def query(
        self,
        query_str: str,
        user_id=None,
        filter=None,
        content_ids=None,
    ):
        """
        Unified query method for retrieving similar content from Qdrant.
        :param query_str: The query string to process.
        :param user_id: The ID of the user making the query.
        :param content_ids: Optional list of content IDs to filter user content.
        :return: List of relevant results.
        """
        try:
            if filter:
                # User content collection, optionally filtered by content_ids
                return self.client.retrieve_similar_content(
                    collection_name=user_id,
                    query=query_str,
                    user_id=user_id,
                    content_ids=content_ids,
                    top_k=10,
                    filter=True,
                )
            else:
                # General collection
                return self.client.retrieve_similar_content(
                    collection_name=VECTOR_COLLECTION,
                    query=query_str,
                    top_k=10,
                    filter=False,
                )
        except Exception as e:
            logger.error(f"An error occurred during query processing: {e}")
            traceback.print_exc()
            return []

    def get_result_from_rag(self, query_str: str, user_id: str, content_ids=None):
        """
        Retrieves the result for a query by calling the query method
        and generating a response based on the retrieved content.
        :param query_str: The query string to process.
        :param user_id: The ID of the user making the request.
        :param content_ids: Optional list of content IDs to filter user content.
        :return: The result from the LLM generated based on the query and retrieved content.
        """
        try:
            logger.info("Generating result for the query.")
            result1 = self.query(query_str=query_str, user_id=user_id)
            result2 = self.query(
                query_str=query_str,
                user_id=user_id,
                filter=True,
                content_ids=content_ids,
            )
            # Combine both results (general + user content)
            combined_results = []
            if isinstance(result1, list):
                combined_results.extend(result1)
            if isinstance(result2, list):
                combined_results.extend(result2)
            if not combined_results:
                logger.error("No query result to process.")
                return None

            urls = SimpleWebSearch().get_context_urls(query_str, num_results=3)
            urls_line = ", ".join(urls) if urls else "None"
            retrieved_blob = (
                f"{combined_results}\n\nWeb context URLs (not scraped): {urls_line}"
            )

            prompt = RETRIEVE_PROMPT.format(
                query=query_str, retrieved_content=retrieved_blob
            )
            result = self.llm.generate(prompt)
            logger.info("Result generated successfully.")
            response = {"text": result, "resource": {"type": "RAG", "id": None}}
            return response
        except Exception as e:
            logger.error(f"An error occurred while generating the result: {e}")
            traceback.print_exc()
            return None
