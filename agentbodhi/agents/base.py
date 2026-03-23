import hashlib
from typing import Any, Optional

from ..core.utils import extract_json


class ResearchAgent:
    def __init__(self, client: Any, model: str, tavily_client: Any = None):
        self.client = client
        self.model = model
        self.tavily = tavily_client
        self.cache = {}

    def execute(self, *args, **kwargs):
        raise NotImplementedError

    def _cache_key(self, *args) -> str:
        return hashlib.md5(str(args).encode()).hexdigest()

    def _get_cached(self, key: str) -> Optional[Any]:
        return self.cache.get(key)

    def _set_cache(self, key: str, value: Any):
        self.cache[key] = value

    def _extract_json(self, text: str) -> str:
        return extract_json(text)

