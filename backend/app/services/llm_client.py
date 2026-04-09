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
from typing import AsyncIterator

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Thin wrapper around the Anthropic Messages API.

    Exposes two generation modes:
      complete()        — waits for the full response (used by /query/)
      astream_complete() — async generator yielding text tokens as they arrive
                           (used by /query/stream for real-time UI updates)

    Both share the same model/token/temperature settings. The async client is
    separate from the sync one because the Anthropic SDK requires different
    client instances for sync vs async operation.
    """

    def __init__(self):
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._async_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
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

    async def astream_complete(self, system_prompt: str, user_message: str) -> AsyncIterator[str]:
        """
        Stream a response from Claude, yielding text tokens as they arrive.

        Uses the async Anthropic client so the event loop isn't blocked while
        waiting for each token. The caller gets the first token in ~200ms instead
        of waiting 1-2s for the full response — a significant UX improvement.

        Not wrapped with @retry because streaming is stateful: if a retry fires
        mid-stream the partial response has already been sent to the client.
        Connection errors before the first token are extremely rare.

        Args:
            system_prompt: instructions that frame how Claude should behave
            user_message:  the actual question + context to answer

        Yields:
            text tokens (strings) as they arrive from the API
        """
        async with self._async_client.messages.stream(
            model=settings.llm_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            async for text in stream.text_stream:
                yield text
