import json
import aiohttp
import requests

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


class CustomGeminiAIClient(AIClient):
    """
    Custom HTTP client for Google Gemini API.

    This implementation uses raw HTTP requests (requests/aiohttp) instead of
    the official SDK, demonstrating how to interact with Gemini's API directly
    and handle its Server-Sent Events (SSE) streaming format.
    """

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response using raw HTTP POST request.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The AI's response message.

        Raises:
            ValueError: If the API response contains no candidates.
            Exception: If the HTTP request fails (non-200 status code).

        Note:
            The URL is constructed by appending ':generateContent' to the model endpoint.
            Uses 'x-goog-api-key' header for authentication.
            Response candidates contain content parts that are concatenated.
        """
        headers = {"x-goog-api-key": self._api_key, "Content-Type": "application/json"}
        contents = [
            {"role": "user" if msg.role == Role.USER else "model",
             "parts": [{"text": msg.content}]}
            for msg in messages
        ]
        body = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": self._system_prompt}]},
        }
        url = f"{self._endpoint}/{self._model_name}:generateContent"
        resp = requests.post(url, headers=headers, json=body)
        content = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        print(content)
        return Message(role=Role.ASSISTANT, content=content)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response using raw HTTP with Server-Sent Events (SSE).

        The response is streamed using Gemini's SSE format, with text chunks
        printed immediately as they arrive.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The complete AI response message after all chunks are received.

        Note:
            The URL is constructed with ':streamGenerateContent?alt=sse' endpoint.
            Uses Server-Sent Events (SSE) format where each line starts with "data: ".
            Each SSE chunk contains candidates with content parts.
            Each text chunk is printed to stdout as it arrives.
        """
        headers = {"x-goog-api-key": self._api_key, "Content-Type": "application/json"}
        contents = [
            {"role": "user" if msg.role == Role.USER else "model",
             "parts": [{"text": msg.content}]}
            for msg in messages
        ]
        body = {
            "contents": contents,
            "systemInstruction": {"parts": [{"text": self._system_prompt}]},
        }
        url = f"{self._endpoint}/{self._model_name}:streamGenerateContent?alt=sse"
        full_content = ""
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=body) as resp:
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    try:
                        chunk = json.loads(data)
                        text = chunk["candidates"][0]["content"]["parts"][0]["text"]
                        print(text, end="", flush=True)
                        full_content += text
                    except (json.JSONDecodeError, KeyError):
                        continue
        print()
        return Message(role=Role.ASSISTANT, content=full_content)