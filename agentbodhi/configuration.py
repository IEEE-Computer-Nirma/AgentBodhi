"""API key helpers for Agent Bodhi."""

from typing import Optional, Tuple

try:
    import config as user_config
except ImportError:  # pragma: no cover
    user_config = None


class ConfigManager:
    @staticmethod
    def get_api_keys() -> Tuple[Optional[str], Optional[str]]:
        if not user_config:
            return None, None

        gemini_key = getattr(user_config, 'GEMINI_API_KEY', None)
        tavily_key = getattr(user_config, 'TAVILY_API_KEY', None)
        return gemini_key, tavily_key

    @staticmethod
    def validate_keys(gemini_key: Optional[str], tavily_key: Optional[str]) -> Tuple[bool, str]:
        if not gemini_key or gemini_key == "YOUR_GEMINI_API_KEY":
            return False, "Invalid Gemini API key"
        if not tavily_key or tavily_key == "YOUR_TAVILY_API_KEY":
            return False, "Invalid Tavily API key"
        return True, "Keys validated"


