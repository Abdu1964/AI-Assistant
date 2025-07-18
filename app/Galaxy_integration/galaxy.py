
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
        emit_to_user(user=user_id,message='Galaxy is processing your query in real time.')
        response = requests.post(GALAXY_MESSAGE_URL, json=payload)
        if response.status_code == 200:
            return response.json().get("response")
        else:
            return f"Error: {response.status_code} - {response.text}"
