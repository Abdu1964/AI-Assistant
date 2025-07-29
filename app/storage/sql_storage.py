import os
import json
from datetime import datetime
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    DateTime,
    Text,
    Float,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import uuid
from app.storage.memory_layer import MemoryManager

# SQLite database configuration
DATABASE_DIR = os.getenv("DATABASE_DIR", "./data")
DATABASE_FILE = os.getenv("DATABASE_FILE", "assistant.db")

# Ensure data directory exists
os.makedirs(DATABASE_DIR, exist_ok=True)

DATABASE_URL = f"sqlite:///{os.path.join(DATABASE_DIR, DATABASE_FILE)}"

# Create database engine with SQLite optimizations
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Allow multiple threads
    echo=False,  # Set to True for SQL debugging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UserInformation(Base):
    __tablename__ = "user_information"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)  # Your user identifier
    question_id = Column(String, default=lambda: str(uuid.uuid4()))
    user_question = Column(Text, nullable=False)
    time = Column(DateTime, default=datetime.utcnow)
    memory = Column(Text)  # JSON string for memory data
    context = Column(Text)  # JSON string for context data
    assistant_answer = Column(Text, nullable=True)


class UserPDFFile(Base):
    __tablename__ = "user_pdf_file"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    pdf_id = Column(String, unique=True, nullable=False)
    filename = Column(String, nullable=False)
    num_pages = Column(Integer)
    file_size = Column(Float)
    upload_time = Column(DateTime)
    summary = Column(Text)


# Create tables
def create_tables():
    print("Creating tables...")
    Base.metadata.create_all(bind=engine)


# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class DatabaseManager:
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal
        # Create tables on initialization
        create_tables()

    def get_session(self):
        return self.SessionLocal()

    def create_user_information(
        self,
        user_id: str,
        user_question: str,
        memory: dict = None,
        context: dict = None,
    ):
        """Create a new user information record and keep only the 3 most recent messages per user."""
        db = self.get_session()
        try:
            # 1. Create the new record first
            user_info = UserInformation(
                user_id=user_id,
                user_question=user_question,
                memory=json.dumps(memory) if memory else None,
                context=json.dumps(context) if context else None,
            )
            db.add(user_info)
            db.flush()  # This assigns the ID without committing

            # Store the values we need before cleanup
            result_data = {
                "id": user_info.id,
                "user_id": user_info.user_id,
                "question_id": user_info.question_id,
                "user_question": user_info.user_question,
                "time": user_info.time,
                "memory": user_info.memory,
                "context": user_info.context,
            }

            # 2. Retrieve all messages for this user, ordered by descending time
            messages = (
                db.query(UserInformation)
                .filter(UserInformation.user_id == user_id)
                .order_by(UserInformation.time.desc())
                .all()
            )

            # 3. If more than 3 messages, delete the oldest ones
            if len(messages) > 3:
                for msg in messages[3:]:
                    db.delete(msg)

            db.commit()

            # Create a detached object with the stored data
            result_obj = UserInformation()
            for key, value in result_data.items():
                setattr(result_obj, key, value)

            return result_obj

        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    def get_user_information(self, user_id: str, limit: int = 10):
        """Retrieve user information records"""
        db = self.get_session()
        try:
            return (
                db.query(UserInformation)
                .filter(UserInformation.user_id == user_id)
                .order_by(UserInformation.time.asc())
                .limit(limit)
                .all()
            )
        finally:
            db.close()

    def get_context_and_memory(self, user_id: str):
        """Extract user questions with context, plus memory for a user, combined in single list"""
        user_information = self.get_user_information(user_id)

        result = []

        for record in user_information:
            # Parse question
            question = record.user_question

            # Parse context
            context = None
            if record.context:
                try:
                    context_data = json.loads(record.context)
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
            if record.memory:
                try:
                    memory_data = json.loads(record.memory)
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

    def update_user_information(
        self, question_id: str, memory: dict = None, context: dict = None
    ):
        """
        Update user information by question_id
        """
        db = self.get_session()
        try:
            user_info = (
                db.query(UserInformation)
                .filter(UserInformation.question_id == question_id)
                .first()
            )

            if user_info:
                if memory is not None:
                    user_info.memory = json.dumps(memory)
                if context is not None:
                    user_info.context = json.dumps(context)

                db.commit()
                return user_info
            return None
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    async def save_user_information(self, advanced_llm, query, user_id, context=None):
        try:
            # memory_manager = MemoryManager(advanced_llm)
            # memory = memory_manager.add_memory(query, user_id)
            # memory_value = memory[0]['memory'] if memory and len(memory) > 0 else None
            user_info = self.create_user_information(
                user_id=user_id, user_question=query, memory="", context=context
            )
            print(
                f"Saved user information with question_id: {user_info.question_id}, {user_info.user_question} {user_info.memory} {user_info.context}"
            )
            return user_info
        except Exception as e:
            print(f"Error saving user information: {e}")
            return None

    # PDF File CRUD Methods
    def add_pdf_file(
        self, user_id, pdf_id, filename, num_pages, file_size, upload_time, summary=None
    ):
        db = self.get_session()
        try:
            pdf_file = UserPDFFile(
                user_id=user_id,
                pdf_id=pdf_id,
                filename=filename,
                num_pages=num_pages,
                file_size=file_size,
                upload_time=upload_time,
                summary=summary,
            )
            db.add(pdf_file)
            db.commit()
            db.refresh(pdf_file)
            return pdf_file
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    def get_user_pdfs(self, user_id):
        db = self.get_session()
        try:
            return (
                db.query(UserPDFFile)
                .filter(UserPDFFile.user_id == user_id)
                .order_by(UserPDFFile.upload_time.asc())
                .all()
            )
        finally:
            db.close()

    def get_pdf_count(self, user_id):
        db = self.get_session()
        try:
            return db.query(UserPDFFile).filter(UserPDFFile.user_id == user_id).count()
        finally:
            db.close()

    def delete_pdf_file(self, user_id, pdf_id):
        db = self.get_session()
        try:
            pdf_file = (
                db.query(UserPDFFile)
                .filter(UserPDFFile.user_id == user_id, UserPDFFile.pdf_id == pdf_id)
                .first()
            )
            if pdf_file:
                db.delete(pdf_file)
                db.commit()
                return True
            return False
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()


db_manager = DatabaseManager()
