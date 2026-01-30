"""Gemini LLM for text generation, summarization, and chat."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Literal, Optional

import google.generativeai as genai


@dataclass
class TokenUsage:
    """Token usage statistics."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class SafetyRating:
    """Safety rating for generated content."""

    category: str
    probability: str
    blocked: bool = False


@dataclass
class GenerationResult:
    """Result of text generation."""

    text: str
    usage: TokenUsage
    finish_reason: str  # "STOP", "MAX_TOKENS", "SAFETY", "RECITATION", "OTHER"
    safety_ratings: list[SafetyRating] = field(default_factory=list)


@dataclass
class Message:
    """A chat message."""

    role: Literal["user", "assistant", "system"]
    content: str


@dataclass
class LLMGenerationConfig:
    """Configuration for LLM text generation."""

    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 40
    max_output_tokens: int = 4096
    stop_sequences: list[str] = field(default_factory=list)


class GeminiLLM:
    """Gemini LLM for summarization and chat."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        generation_config: Optional[LLMGenerationConfig] = None,
    ) -> None:
        """
        Initialize Gemini LLM client.

        Args:
            api_key: Google AI Studio API key
            model: Gemini model to use
            generation_config: Default generation configuration
        """
        self._api_key = api_key
        self._model_name = model
        self._generation_config = generation_config or LLMGenerationConfig()

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _build_generation_config(
        self,
        override: Optional[LLMGenerationConfig] = None,
    ) -> dict:
        """Build generation config dict for API call."""
        config = override or self._generation_config
        result = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "max_output_tokens": config.max_output_tokens,
        }
        if config.stop_sequences:
            result["stop_sequences"] = config.stop_sequences
        return result

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert Message objects to Gemini format."""
        converted = []
        for msg in messages:
            role = "user" if msg.role == "user" else "model"
            converted.append({"role": role, "parts": [msg.content]})
        return converted

    def _extract_usage(self, response) -> TokenUsage:
        """Extract token usage from response."""
        try:
            usage_metadata = response.usage_metadata
            return TokenUsage(
                prompt_tokens=getattr(usage_metadata, "prompt_token_count", 0),
                completion_tokens=getattr(usage_metadata, "candidates_token_count", 0),
                total_tokens=getattr(usage_metadata, "total_token_count", 0),
            )
        except Exception:
            return TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    def _extract_safety_ratings(self, response) -> list[SafetyRating]:
        """Extract safety ratings from response."""
        ratings = []
        try:
            if response.candidates and response.candidates[0].safety_ratings:
                for rating in response.candidates[0].safety_ratings:
                    ratings.append(
                        SafetyRating(
                            category=str(rating.category),
                            probability=str(rating.probability),
                            blocked=getattr(rating, "blocked", False),
                        )
                    )
        except Exception:
            pass
        return ratings

    def _get_finish_reason(self, response) -> str:
        """Get finish reason from response."""
        try:
            if response.candidates:
                reason = response.candidates[0].finish_reason
                return str(reason.name) if hasattr(reason, "name") else str(reason)
        except Exception:
            pass
        return "OTHER"

    def _generate_sync(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        generation_config: Optional[LLMGenerationConfig] = None,
    ) -> GenerationResult:
        """Synchronous generation for use with executor."""
        config = self._build_generation_config(generation_config)

        # Build content with optional system instruction
        if system_prompt:
            model = genai.GenerativeModel(
                self._model_name,
                system_instruction=system_prompt,
            )
        else:
            model = self._model

        response = model.generate_content(
            prompt,
            generation_config=config,
        )

        return GenerationResult(
            text=response.text,
            usage=self._extract_usage(response),
            finish_reason=self._get_finish_reason(response),
            safety_ratings=self._extract_safety_ratings(response),
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        generation_config: Optional[LLMGenerationConfig] = None,
    ) -> GenerationResult:
        """
        Generate text response.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            generation_config: Override default generation config

        Returns:
            GenerationResult with text and metadata
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._generate_sync,
            prompt,
            system_prompt,
            generation_config,
        )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        generation_config: Optional[LLMGenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """
        Generate text response with streaming.

        Args:
            prompt: User prompt
            system_prompt: Optional system instruction
            generation_config: Override default generation config

        Yields:
            Text chunks as they are generated
        """
        config = self._build_generation_config(generation_config)

        if system_prompt:
            model = genai.GenerativeModel(
                self._model_name,
                system_instruction=system_prompt,
            )
        else:
            model = self._model

        response = model.generate_content(
            prompt,
            generation_config=config,
            stream=True,
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

    def _chat_sync(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        generation_config: Optional[LLMGenerationConfig] = None,
    ) -> GenerationResult:
        """Synchronous chat for use with executor."""
        config = self._build_generation_config(generation_config)

        # Create model with system instruction
        if system_prompt:
            model = genai.GenerativeModel(
                self._model_name,
                system_instruction=system_prompt,
            )
        else:
            model = self._model

        # Convert messages to Gemini format
        history = []
        for msg in messages[:-1]:  # All but last message
            history.append(
                {
                    "role": "user" if msg.role == "user" else "model",
                    "parts": [msg.content],
                }
            )

        chat = model.start_chat(history=history)

        # Send the last message
        response = chat.send_message(
            messages[-1].content,
            generation_config=config,
        )

        return GenerationResult(
            text=response.text,
            usage=self._extract_usage(response),
            finish_reason=self._get_finish_reason(response),
            safety_ratings=self._extract_safety_ratings(response),
        )

    async def chat(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        generation_config: Optional[LLMGenerationConfig] = None,
    ) -> GenerationResult:
        """
        Multi-turn chat.

        Args:
            messages: List of chat messages
            system_prompt: Optional system instruction
            generation_config: Override default generation config

        Returns:
            GenerationResult with assistant response
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._chat_sync,
            messages,
            system_prompt,
            generation_config,
        )

    async def chat_stream(
        self,
        messages: list[Message],
        system_prompt: Optional[str] = None,
        generation_config: Optional[LLMGenerationConfig] = None,
    ) -> AsyncIterator[str]:
        """
        Multi-turn chat with streaming.

        Args:
            messages: List of chat messages
            system_prompt: Optional system instruction
            generation_config: Override default generation config

        Yields:
            Text chunks as they are generated
        """
        config = self._build_generation_config(generation_config)

        if system_prompt:
            model = genai.GenerativeModel(
                self._model_name,
                system_instruction=system_prompt,
            )
        else:
            model = self._model

        # Convert messages to Gemini format
        history = []
        for msg in messages[:-1]:
            history.append(
                {
                    "role": "user" if msg.role == "user" else "model",
                    "parts": [msg.content],
                }
            )

        chat = model.start_chat(history=history)

        response = chat.send_message(
            messages[-1].content,
            generation_config=config,
            stream=True,
        )

        for chunk in response:
            if chunk.text:
                yield chunk.text

    async def summarize_captions(
        self,
        captions: list[str],
        prompt_template: str,
        generation_config: Optional[LLMGenerationConfig] = None,
    ) -> str:
        """
        Summarize a batch of captions.

        Args:
            captions: List of caption strings to summarize
            prompt_template: Template for summarization prompt (should contain {captions})
            generation_config: Override default generation config

        Returns:
            Summarized text
        """
        # Format captions into prompt
        captions_text = "\n".join([f"- {caption}" for caption in captions])

        prompt = prompt_template.format(captions=captions_text)

        result = await self.generate(
            prompt=prompt,
            generation_config=generation_config,
        )

        return result.text

    async def aggregate_summaries(
        self,
        summaries: list[str],
        aggregation_prompt: str,
        generation_config: Optional[LLMGenerationConfig] = None,
    ) -> str:
        """
        Aggregate multiple summaries into a final summary.

        Args:
            summaries: List of intermediate summaries
            aggregation_prompt: Prompt for aggregation (should contain {summaries})
            generation_config: Override default generation config

        Returns:
            Final aggregated summary
        """
        summaries_text = "\n\n".join(
            [f"Summary {i + 1}:\n{s}" for i, s in enumerate(summaries)]
        )

        prompt = aggregation_prompt.format(summaries=summaries_text)

        result = await self.generate(
            prompt=prompt,
            generation_config=generation_config,
        )

        return result.text

    async def check_notification(
        self,
        caption: str,
        events: list[str],
        notification_prompt: str,
        generation_config: Optional[LLMGenerationConfig] = None,
    ) -> tuple[bool, list[str], str]:
        """
        Check if caption matches notification events.

        Args:
            caption: Caption to check
            events: List of events to detect
            notification_prompt: Prompt for notification check

        Returns:
            Tuple of (should_notify, detected_events, explanation)
        """
        events_list = ", ".join(events)
        prompt = notification_prompt.format(
            caption=caption,
            events=events_list,
        )

        result = await self.generate(
            prompt=prompt,
            generation_config=generation_config,
        )

        # Parse response to extract detected events
        detected_events = []
        should_notify = False

        if "DETECTED:" in result.text.upper():
            should_notify = True
            # Extract event names from response
            for event in events:
                if event.lower() in result.text.lower():
                    detected_events.append(event)

        return should_notify, detected_events, result.text

    @staticmethod
    def get_model_info() -> tuple[str, str, str]:
        """Get model information (id, api_type, owned_by)."""
        return "gemini-2.0-flash", "google-ai", "Google"
