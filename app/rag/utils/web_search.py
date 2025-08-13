import logging
import time
from typing import List
from googlesearch import search

logger = logging.getLogger(__name__)


class SimpleWebSearch:
    def __init__(self) -> None:
        pass

    def get_context_urls(self, query: str, num_results: int = 3) -> List[str]:
        if search is None:
            logger.warning("googlesearch not available; returning empty URL list")
            return []
        try:
            urls: List[str] = []
            for url in search(query, num_results=num_results):
                urls.append(url)
                time.sleep(1)
            return urls
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []
