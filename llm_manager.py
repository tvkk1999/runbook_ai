# llm_manager.py with Ollama - llama3.1:8b

import requests
from typing import Optional
from config import OLLAMA_API_URL, DEFAULT_MODEL_NAME, LLM_PARAMS


class LocalLLMManager:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, api_url: str = OLLAMA_API_URL):
        self.model_name = model_name
        self.api_url = api_url.rstrip("/")

    def ensure_ollama_running(self) -> bool:
        """
        Check if Ollama local service is running by querying available models/tags.
        """
        try:
            response = requests.get(f"{self.api_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def pull_model(self, model_name: Optional[str] = None) -> bool:
        """
        Pull/download the specified model if not already present.
        """
        model = model_name or self.model_name
        try:
            response = requests.post(
                f"{self.api_url}/api/pull",
                json={"name": model},
                timeout=30
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def generate_response(self, prompt: str, context: str = "") -> str:
        """
        Send prompt with context to Ollama model and get generated completion.
        The prompt instructs the model to base answer only on provided context,
        respond concisely, and indicate lack of info if applicable.
        """
        full_prompt = f"""
Context from documents:
{context}

Question: {prompt}

Instructions:
- Answer ONLY using the above context.
- If information is not present, say "I don't have enough information".
- Be concise and accurate.
- Cite relevant sections when possible.

Answer:
"""

        try:
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": LLM_PARAMS.get("temperature", 0.3),
                    "top_p": LLM_PARAMS.get("top_p", 0.9),
                    "max_tokens": LLM_PARAMS.get("max_tokens", 512)
                }
            }
            response = requests.post(
                f"{self.api_url}/api/generate",
                json=payload,
                timeout=60
            )
            if response.status_code == 200:
                json_resp = response.json()
                return json_resp.get("response", "Error: No response field in API output")
            else:
                return f"Error: Ollama returned status {response.status_code}"
        except requests.RequestException as e:
            return f"Error: Unable to connect to Ollama server - {str(e)}"
