"""
LLM client: wraps the Anthropic API for answer generation.

This layer handles all concerns specific to the API call itself:
  - Authentication (API key from settings)
  - Retry logic (network blips shouldn't fail the whole request)
  - Token limits and temperature
  - Isolating the rest of the codebase from the Anthropic SDK

WHY RETRY LOGIC:
  External API calls fail occasionally — network timeouts, rate limits, or
  transient 500s from the provider. A single retry with exponential backoff
  recovers from most of these without impacting the user. We use tenacity
  for this because it's declarative, readable, and handles the common cases.

WHY TEMPERATURE = 0:
  For grounded Q&A we want deterministic, factual responses — not creative ones.
  Temperature 0 makes the model pick the most likely token at each step, which
  produces more consistent answers when given the same context.
"""

import logging

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Thin wrapper around the Anthropic Messages API.

    Initialized once at startup and shared across all requests.
    The underlying anthropic.Anthropic client manages its own connection pool.
    """

    def __init__(self):
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        logger.info("LLMClient initialized — model=%s max_tokens=%d", settings.llm_model, settings.llm_max_tokens)

    @retry(
        retry=retry_if_exception_type((anthropic.APIConnectionError, anthropic.APITimeoutError)),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def complete(self, system_prompt: str, user_message: str) -> str:
        """
        Send a prompt to Claude and return the text response.

        Retries up to 3 times on connection/timeout errors with exponential
        backoff (1s, 2s, 4s). Rate limit errors (429) are not retried here —
        those should bubble up so the caller can surface them to the user.

        Args:
            system_prompt: instructions that frame how Claude should behave
            user_message:  the actual question + context to answer

        Returns:
            Claude's response as a plain string
        """
        response = self._client.messages.create(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text
