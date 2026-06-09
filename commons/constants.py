"""
Configuration constants for AI service integrations.

This module centralizes all API endpoints, API keys, and default configuration
values used across different AI service providers (OpenAI, Anthropic, Gemini).

All API keys are loaded from environment variables for security.
"""

import os

# Default system prompt used across all AI services
DEFAULT_SYSTEM_PROMPT = "You are an assistant who answers concisely and informatively."

# OpenAI API configuration
# Read from OPENAI_BASE_URL if set (OpenAI SDK convention includes /v1; strip it so tasks can
# construct full paths as OPENAI_HOST + "/v1/...")
_openai_base_url = os.getenv('OPENAI_BASE_URL', '')
OPENAI_HOST = _openai_base_url.rstrip('/').removesuffix('/v1') if _openai_base_url else "https://api.openai.com"
OPENAI_CHAT_COMPLETIONS_ENDPOINT = f"{OPENAI_HOST}/v1/chat/completions"
OPENAI_RESPONSES_ENDPOINT = f"{OPENAI_HOST}/v1/responses"
OPENAI_EMBEDDINGS_ENDPOINT = f"{OPENAI_HOST}/v1/embeddings"
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')

# Anthropic API configuration
ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY', '')

# Google Gemini API configuration
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')

# User Service API configuration
USER_SERVICE_ENDPOINT = "http://localhost:8041"