from anthropic import Anthropic, AsyncAnthropic

from commons.models.message import Message
from commons.models.role import Role
from t1_llm_api.base_client import AIClient


class AnthropicAIClient(AIClient):
    """
    Client for Anthropic's Claude API using the official SDK.

    This implementation uses the official Anthropic Python library to interact
    with Claude models, providing both synchronous and streaming response capabilities.

    Attributes:
        _client (Anthropic): Synchronous Anthropic client instance.
        _async_client (AsyncAnthropic): Asynchronous Anthropic client instance.
        Inherits all other attributes from AIClient.
    """

    def __init__(self, endpoint: str, model_name: str, api_key: str, system_prompt: str):
        """
        Initialize the Anthropic client with SDK.

        Args:
            endpoint (str): The Anthropic API endpoint (for compatibility, not used by SDK).
            model_name (str): The Claude model to use (e.g., 'claude-3-opus', 'claude-sonnet-4-5').
            api_key (str): The Anthropic API key for authentication.
            system_prompt (str): The system instruction to guide Claude's behavior.
        """
        super().__init__(endpoint, model_name, api_key, system_prompt)
        self._client = Anthropic(api_key=self._api_key)
        self._async_client = AsyncAnthropic(api_key=self._api_key)

    def response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a synchronous response from Anthropic's Claude API.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The AI's response message.

        Note:
            Claude's API uses a separate 'system' parameter for system instructions.
            Response content blocks are concatenated into a single text response.
            The response is printed to stdout before being returned.
        """
        resp = self._client.messages.create(
            model=self._model_name,
            max_tokens=kwargs.get("max_tokens", 1024),
            system=self._system_prompt,
            messages=[msg.to_dict() for msg in messages],
        )
        content = resp.content[0].text
        print(content)
        return Message(role=Role.ASSISTANT, content=content)

    async def stream_response(self, messages: list[Message], **kwargs) -> Message:
        """
        Get a streaming response from Anthropic's Claude API.

        The response is streamed using event-based streaming, with text deltas
        printed immediately as they arrive.

        Args:
            messages (list[Message]): The conversation history.
            **kwargs: Additional parameters like max_tokens (default: 1024).

        Returns:
            Message: The complete AI response message after all deltas are received.

        Note:
            Listens for 'content_block_delta' events with text deltas.
            Each delta is printed to stdout as it arrives for real-time display.
        """
        full_content = ""
        async with self._async_client.messages.stream(
            model=self._model_name,
            max_tokens=kwargs.get("max_tokens", 1024),
            system=self._system_prompt,
            messages=[msg.to_dict() for msg in messages],
        ) as stream:
            async for text in stream.text_stream:
                print(text, end="", flush=True)
                full_content += text
        print()
        return Message(role=Role.ASSISTANT, content=full_content)
