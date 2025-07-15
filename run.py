import eventlet
eventlet.monkey_patch()

from app import create_app
from dotenv import load_dotenv
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()
port = int(os.getenv('FLASK_PORT', 5003))
app, socketio = create_app()

if __name__ == '__main__':
    logger.info(f"Starting application on port {port}")
    socketio.run(app, host='0.0.0.0', port=port, debug=False, use_reloader=False)
