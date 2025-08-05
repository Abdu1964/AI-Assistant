import os
import json
from datetime import datetime
from pymongo import MongoClient
import uuid
import logging

logger = logging.getLogger(__name__)


class MongoManager:
    def __init__(self):
        self.client = None
        self.db = None
        self.user_info_collection = None
        self.content_files_collection = None
        self._connect()
        self._create_indexes()

    def _connect(self):
        # Initialize MongoDB connection
        try:
            mongo_uri = os.getenv("MONGO_URL")
            database_name = os.getenv("MONGO_DATABASE", "ai_assistant")

            self.client = MongoClient(mongo_uri)
            self.db = self.client[database_name]
            self.user_info_collection = self.db["user_information"]
            self.content_files_collection = self.db["user_content_files"]

            logger.info(f"MongoDB connection established to {database_name}")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise

    def _create_indexes(self):
        # Create necessary indexes for performance
        try:
            # User information indexes
            self.user_info_collection.create_index("user_id")
            self.user_info_collection.create_index("time")
            self.user_info_collection.create_index("question_id")

            # Content files indexes
            self.content_files_collection.create_index("user_id")
            self.content_files_collection.create_index("content_id", unique=True)
            self.content_files_collection.create_index("content_type")
            self.content_files_collection.create_index("upload_time")

            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.error(f"Error creating MongoDB indexes: {e}")

    # User Information Methods
    def create_user_information(
        self,
        user_id: str,
        user_question: str,
        memory: dict = None,
        context: dict = None,
    ):
        try:
            # Clean old records (keep only 3 most recent)
            self._clean_old_user_records(user_id)

            user_info = {
                "user_id": user_id,
                "question_id": str(uuid.uuid4()),
                "user_question": user_question,
                "memory": json.dumps(memory) if memory else None,
                "context": json.dumps(context) if context else None,
                "time": datetime.utcnow(),
                "assistant_answer": None,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }

            result = self.user_info_collection.insert_one(user_info)
            user_info["_id"] = result.inserted_id
            return user_info

        except Exception as e:
            logger.error(f"Error creating user information: {e}")
            raise

    def get_user_information(self, user_id: str, limit: int = 10):
        try:
            cursor = (
                self.user_info_collection.find({"user_id": user_id})
                .sort("time", 1)
                .limit(limit)
            )

            return list(cursor)
        except Exception as e:
            logger.error(f"Error retrieving user information: {e}")
            return []

    def _clean_old_user_records(self, user_id: str):
        # Keep only 3 most recent records per user
        try:
            # Get all records for user, sorted by time descending
            all_records = list(
                self.user_info_collection.find({"user_id": user_id}).sort("time", -1)
            )

            # Delete records beyond the 3rd one (keep only the 3 most recent)
            if len(all_records) > 3:
                records_to_delete = all_records[3:]
                for record in records_to_delete:
                    self.user_info_collection.delete_one({"_id": record["_id"]})
                logger.info(
                    f"Cleaned {len(records_to_delete)} old records for user {user_id}"
                )

        except Exception as e:
            logger.error(f"Error cleaning old user records: {e}")

    def update_user_information(
        self, question_id: str, memory: dict = None, context: dict = None
    ):
        # Update user information by question_id
        try:
            update_data = {"updated_at": datetime.utcnow()}
            if memory is not None:
                update_data["memory"] = json.dumps(memory)
            if context is not None:
                update_data["context"] = json.dumps(context)

            result = self.user_info_collection.update_one(
                {"question_id": question_id}, {"$set": update_data}
            )

            if result.modified_count > 0:
                return self.user_info_collection.find_one({"question_id": question_id})
            return None

        except Exception as e:
            logger.error(f"Error updating user information: {e}")
            raise

    # Content File Methods
    def add_content_file(
        self,
        user_id: str,
        content_id: str,
        content_type: str = "pdf",
        filename: str = None,
        num_pages: int = None,
        url: str = None,
        title: str = None,
        author: str = None,
        publish_date: datetime = None,
        file_size: float = None,
        upload_time: datetime = None,
        summary: str = None,
        keywords: str = None,
        topics: str = None,
        suggested_questions: str = None,
    ):
        try:
            content_file = {
                "user_id": user_id,
                "content_id": content_id,
                "content_type": content_type,
                "filename": filename,
                "num_pages": num_pages,
                "url": url,
                "title": title,
                "author": author,
                "publish_date": publish_date,
                "file_size": file_size,
                "upload_time": upload_time or datetime.utcnow(),
                "summary": summary,
                "keywords": keywords,
                "topics": topics,
                "suggested_questions": suggested_questions,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }

            result = self.content_files_collection.insert_one(content_file)
            content_file["_id"] = result.inserted_id
            return content_file

        except Exception as e:
            logger.error(f"Error adding content file: {e}")
            raise

    def get_user_content_files(self, user_id: str, content_type: str = None):
        try:
            query = {"user_id": user_id}
            if content_type:
                query["content_type"] = content_type

            cursor = self.content_files_collection.find(query).sort("upload_time", 1)
            return list(cursor)

        except Exception as e:
            logger.error(f"Error retrieving content files: {e}")
            return []

    def get_content_file_by_id(self, user_id: str, content_id: str):
        try:
            return self.content_files_collection.find_one(
                {"user_id": user_id, "content_id": content_id}
            )
        except Exception as e:
            logger.error(f"Error retrieving content file by ID: {e}")
            return None

    def get_content_count(self, user_id: str, content_type: str = None):
        try:
            query = {"user_id": user_id}
            if content_type:
                query["content_type"] = content_type

            return self.content_files_collection.count_documents(query)

        except Exception as e:
            logger.error(f"Error counting content files: {e}")
            return 0

    def update_content_file(
        self,
        user_id: str,
        content_id: str,
        summary: str = None,
        keywords: str = None,
        topics: str = None,
        suggested_questions: str = None,
    ):
        try:
            update_data = {"updated_at": datetime.utcnow()}
            if summary is not None:
                update_data["summary"] = summary
            if keywords is not None:
                update_data["keywords"] = keywords
            if topics is not None:
                update_data["topics"] = topics
            if suggested_questions is not None:
                update_data["suggested_questions"] = suggested_questions

            result = self.content_files_collection.update_one(
                {"user_id": user_id, "content_id": content_id}, {"$set": update_data}
            )

            if result.modified_count > 0:
                return self.get_content_file_by_id(user_id, content_id)
            return None

        except Exception as e:
            logger.error(f"Error updating content file: {e}")
            raise

    def delete_content_file(self, user_id: str, content_id: str):
        try:
            result = self.content_files_collection.delete_one(
                {"user_id": user_id, "content_id": content_id}
            )
            return result.deleted_count > 0

        except Exception as e:
            logger.error(f"Error deleting content file: {e}")
            return False

    def get_context_and_memory(self, user_id: str):
        try:
            user_information = self.get_user_information(user_id)
            result = []

            for record in user_information:
                # Parse question
                question = record.get("user_question", "")

                # Parse context
                context = None
                if record.get("context"):
                    try:
                        context_data = json.loads(record["context"])
                        if "content" in context_data:
                            content = context_data["content"]
                            if isinstance(content, str):
                                context = json.loads(content)
                            else:
                                context = content
                        else:
                            context = context_data
                    except json.JSONDecodeError:
                        context = None

                # Parse memory
                memory = None
                if record.get("memory"):
                    try:
                        memory_data = json.loads(record["memory"])
                        if "content" in memory_data:
                            content = memory_data["content"]
                            if isinstance(content, str):
                                memory = json.loads(content)
                            else:
                                memory = content
                        else:
                            memory = memory_data
                    except json.JSONDecodeError:
                        memory = None

                # If memory is empty or None, use empty string
                if memory in [None, []]:
                    memory = ""

                # Add to result list
                result.append(
                    {
                        "QUESTION": {"question": question, "context": context},
                        "MEMORIES": memory,
                    }
                )

            return result

        except Exception as e:
            logger.error(f"Error getting context and memory: {e}")
            return []

    async def save_user_information(self, advanced_llm, query, user_id, context=None):
        try:
            user_info = self.create_user_information(
                user_id=user_id, user_question=query, memory="", context=context
            )
            logger.info(
                f"Saved user information with question_id: {user_info['question_id']}"
            )
            return user_info
        except Exception as e:
            logger.error(f"Error saving user information: {e}")
            return None


mongo_db_manager = MongoManager()
