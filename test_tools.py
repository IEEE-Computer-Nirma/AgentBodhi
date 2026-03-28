from google import genai
from google.genai import types
from agentbodhi.configuration import ConfigManager
import inspect
gemini_key, _ = ConfigManager.get_api_keys()

def search_web(query: str) -> str:
    """Searches the web for general knowledge."""
    return "result"

client = genai.Client(api_key=gemini_key)
chat = client.chats.create(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(tools=[search_web])
)
resp = chat.send_message("Please search the web for the latest news on dogs.")
print("Function Calls:")
print(resp.function_calls)
