
from app.socket_manager import emit_to_user
import requests
import os

GALAXY_MESSAGE_URL= os.getenv("GALAXY_MESSAGE_URL")
class GalaxyHandler:
    def __init__(self, llm) -> None:
        self.llm = llm 

    def get_galaxy_info(self, query, user_id):
        payload = {
            "message": query
        }
        emit_to_user(user=user_id,message='processing your query on Galaxy in real time.')
        response = requests.post(GALAXY_MESSAGE_URL, json=payload)
        if response.status_code == 200:
            response = {"text":response.json().get("response"),"resource":{"type":"galaxy",id:None}}
            return response
        else:
            return {"text": f"I apologize, but I wasn't able to generate what you requested. Could you please rephrase your question."}
