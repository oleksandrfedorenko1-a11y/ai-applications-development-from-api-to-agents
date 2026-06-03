import json
import aiohttp
import requests

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


class CustomAnthropicAIClient(AIClient):
    """
    Custom HTTP client for Anthropic's Claude API.

    This implementation uses raw HTTP requests (requests/aiohttp) instead of
    the official SDK, demonstrating how to interact with Claude's API directly
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
            ValueError: If the API response contains no content blocks.
            Exception: If the HTTP request fails (non-200 status code).

        Note:
            Requires 'x-api-key' header and 'anthropic-version' header.
            Claude's API returns content as an array of content blocks.
            The response is printed to stdout before being returned.
        """
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        body = {
            "model": self._model_name,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "system": self._system_prompt,
            "messages": [msg.to_dict() for msg in messages],
        }
        resp = requests.post(self._endpoint, headers=headers, json=body)
        content = resp.json()["content"][0]["text"]
        print(content)
        return Message(role=Role.ASSISTANT, content=content)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response using raw HTTP with Server-Sent Events (SSE).

        The response is streamed using Anthropic's SSE format, with text deltas
        printed immediately as they arrive.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The complete AI response message after all deltas are received.

        Note:
            Uses Server-Sent Events (SSE) format where each line starts with "data: ".
            Listens for 'content_block_delta' events with 'text_delta' type.
            Stops processing when 'message_stop' event is received.
            Each delta is printed to stdout as it arrives.
        """
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        body = {
            "model": self._model_name,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "system": self._system_prompt,
            "messages": [msg.to_dict() for msg in messages],
            "stream": True,
        }
        full_content = ""
        async with aiohttp.ClientSession() as session:
            async with session.post(self._endpoint, headers=headers, json=body) as resp:
                async for line in resp.content:
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    try:
                        event_data = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if event_data.get("type") == "content_block_delta":
                        delta = event_data.get("delta", {}).get("text", "")
                        if delta:
                            print(delta, end="", flush=True)
                            full_content += delta
                    elif event_data.get("type") == "message_stop":
                        break
        print()
        return Message(role=Role.ASSISTANT, content=full_content)

