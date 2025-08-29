from datetime import datetime
from app.storage.mongo_storage import mongo_db_manager
import uuid


class HistoryManager:
    def create_history(
        self, user_id, user_message, assistant_answer, graph_id_referenced=None
    ):
        # Generate a unique query_id
        query_id = str(uuid.uuid4())

        # Create a new UserInformation record
        mongo_db_manager.create_user_information(
            user_id=user_id,
            user_question=user_message,
            memory=None,
            context=None,
            graph_id_referenced=graph_id_referenced,
        )

        # Update the assistant_answer and question_id for the latest record
        latest_record = mongo_db_manager.user_info_collection.find_one(
            {"user_id": user_id}, sort=[("time", -1)]
        )

        if latest_record:
            mongo_db_manager.user_info_collection.update_one(
                {"_id": latest_record["_id"]},
                {
                    "$set": {
                        "assistant_answer": assistant_answer,
                        "question_id": query_id,
                        "updated_at": datetime.utcnow(),
                    }
                },
            )

        return query_id

    def retrieve_user_history(self, user_id, limit=5):
        records = mongo_db_manager.get_user_information(user_id, limit=limit)
        user_id_str = str(user_id)
        history = []
        for record in records:
            history.append(
                {
                    "query_id": record.get("question_id"),
                    "user": record.get("user_question"),
                    "assistant answer": record.get("assistant_answer"),
                    "graph_id_referenced": record.get("graph_id_referenced"),
                    "time": (
                        record.get("time").isoformat() if record.get("time") else None
                    ),
                }
            )
        return {user_id_str: history}

    def get_entry_by_query_id(self, user_id, query_id):
        record = mongo_db_manager.user_info_collection.find_one(
            {"user_id": user_id, "question_id": query_id}
        )

        if record:
            return {
                "query_id": record.get("question_id"),
                "user": record.get("user_question"),
                "assistant answer": record.get("assistant_answer"),
                "graph_id_referenced": record.get("graph_id_referenced"),
                "time": record.get("time").isoformat() if record.get("time") else None,
            }
        return None

    def clear_user_history(self, user_id):
        mongo_db_manager.user_info_collection.delete_many({"user_id": user_id})
