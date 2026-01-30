"""Gemini VLM for video understanding and captioning."""

from __future__ import annotations

import asyncio
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

import google.generativeai as genai
from google.generativeai.types import HarmBlockThreshold, HarmCategory


@dataclass
class VideoEvent:
    """A single event detected in the video."""

    start_time: str  # HH:MM:SS format
    end_time: str  # HH:MM:SS format
    description: str  # Event description
    confidence: Optional[float] = None  # Optional confidence score


@dataclass
class TokenUsage:
    """Token usage statistics for a generation request."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass
class SafetyRating:
    """Safety rating for generated content."""

    category: str  # e.g., "HARM_CATEGORY_HARASSMENT"
    probability: str  # e.g., "NEGLIGIBLE", "LOW", "MEDIUM", "HIGH"
    blocked: bool = False


@dataclass
class VideoAnalysisResult:
    """Result of video analysis."""

    captions: str  # Raw caption text
    parsed_events: list[VideoEvent]  # Structured events
    usage: TokenUsage  # Token usage
    safety_ratings: list[SafetyRating]  # Safety ratings
    reasoning_description: Optional[str] = None  # Chain-of-thought reasoning if enabled
    raw_response: Optional[str] = None  # Full raw response for debugging


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    temperature: float = 0.2
    top_p: float = 0.8
    top_k: int = 40
    max_output_tokens: int = 2048
    stop_sequences: list[str] = field(default_factory=list)


@dataclass
class SafetySettings:
    """Safety filter settings."""

    harassment: str = "BLOCK_ONLY_HIGH"
    hate_speech: str = "BLOCK_ONLY_HIGH"
    sexually_explicit: str = "BLOCK_ONLY_HIGH"
    dangerous_content: str = "BLOCK_ONLY_HIGH"


class VLMError(Exception):
    """Base exception for VLM errors."""

    pass


class SafetyBlockedError(VLMError):
    """Content was blocked by safety filters."""

    pass


class ContextLengthExceededError(VLMError):
    """Input exceeded maximum context length."""

    pass


class GenerationError(VLMError):
    """Generation failed."""

    pass


class GeminiVLM:
    """Gemini-based video understanding using native video upload."""

    # Default prompts
    DEFAULT_CAPTION_PROMPT = """Analyze this video segment and provide detailed captions of all events.
For each distinct event or action, provide:
- Start and end timestamps in format <HH:MM:SS>
- Detailed description of what is happening
- Any relevant objects, people, or text visible
Focus on actions, interactions, and notable occurrences."""

    # Timestamp patterns for parsing
    TIMESTAMP_PATTERNS = [
        # <HH:MM:SS> <HH:MM:SS> description
        r"<(\d{2}:\d{2}:\d{2})>\s*<(\d{2}:\d{2}:\d{2})>\s*(.+)",
        # HH:MM:SS - HH:MM:SS: description
        r"(\d{2}:\d{2}:\d{2})\s*-\s*(\d{2}:\d{2}:\d{2}):\s*(.+)",
        # [HH:MM:SS-HH:MM:SS] description
        r"\[(\d{2}:\d{2}:\d{2})-(\d{2}:\d{2}:\d{2})\]\s*(.+)",
        # HH:MM:SS.mmm format
        r"<(\d{2}:\d{2}:\d{2}\.\d+)>\s*<(\d{2}:\d{2}:\d{2}\.\d+)>\s*(.+)",
        # <HH:MM:SS> description (single timestamp, use same for start/end)
        r"<(\d{2}:\d{2}:\d{2})>\s*(.+)",
    ]

    # Safety setting mapping
    SAFETY_MAPPING = {
        "BLOCK_NONE": HarmBlockThreshold.BLOCK_NONE,
        "BLOCK_ONLY_HIGH": HarmBlockThreshold.BLOCK_ONLY_HIGH,
        "BLOCK_MEDIUM_AND_ABOVE": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        "BLOCK_LOW_AND_ABOVE": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
    }

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        generation_config: Optional[GenerationConfig] = None,
        safety_settings: Optional[SafetySettings] = None,
    ) -> None:
        """
        Initialize Gemini VLM client.

        Args:
            api_key: Google AI Studio API key
            model: Gemini model to use (default: gemini-2.0-flash)
            generation_config: Default generation configuration
            safety_settings: Default safety filter settings
        """
        self._api_key = api_key
        self._model_name = model
        self._generation_config = generation_config or GenerationConfig()
        self._safety_settings = safety_settings or SafetySettings()

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _build_generation_config(
        self,
        override: Optional[GenerationConfig] = None,
    ) -> dict:
        """Build generation config dict for API call."""
        config = override or self._generation_config
        return {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "max_output_tokens": config.max_output_tokens,
            "stop_sequences": config.stop_sequences if config.stop_sequences else None,
        }

    def _build_safety_settings(self) -> list[dict]:
        """Build safety settings list for API call."""
        return [
            {
                "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                "threshold": self.SAFETY_MAPPING.get(
                    self._safety_settings.harassment, HarmBlockThreshold.BLOCK_ONLY_HIGH
                ),
            },
            {
                "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": self.SAFETY_MAPPING.get(
                    self._safety_settings.hate_speech, HarmBlockThreshold.BLOCK_ONLY_HIGH
                ),
            },
            {
                "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": self.SAFETY_MAPPING.get(
                    self._safety_settings.sexually_explicit, HarmBlockThreshold.BLOCK_ONLY_HIGH
                ),
            },
            {
                "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": self.SAFETY_MAPPING.get(
                    self._safety_settings.dangerous_content, HarmBlockThreshold.BLOCK_ONLY_HIGH
                ),
            },
        ]

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

    def parse_events(self, caption_text: str) -> list[VideoEvent]:
        """
        Parse timestamped events from caption text.

        Expected formats:
        - <00:00:05> <00:00:10> Person walks across the room
        - 00:00:05 - 00:00:10: Person walks across the room
        - [00:00:05-00:00:10] Person walks across the room

        Args:
            caption_text: Raw caption text from model

        Returns:
            List of parsed VideoEvent objects
        """
        events = []
        for line in caption_text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            for pattern in self.TIMESTAMP_PATTERNS:
                match = re.match(pattern, line)
                if match:
                    groups = match.groups()
                    if len(groups) == 3:
                        events.append(
                            VideoEvent(
                                start_time=groups[0],
                                end_time=groups[1],
                                description=groups[2].strip(),
                            )
                        )
                    elif len(groups) == 2:
                        # Single timestamp pattern
                        events.append(
                            VideoEvent(
                                start_time=groups[0],
                                end_time=groups[0],
                                description=groups[1].strip(),
                            )
                        )
                    break

        return events

    def _analyze_sync(
        self,
        file_uri: str,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None,
    ) -> VideoAnalysisResult:
        """Synchronous video analysis for use with executor."""
        try:
            # Get the uploaded file reference
            file_name = file_uri.split("/")[-1] if "/" in file_uri else file_uri
            video_file = genai.get_file(file_name)

            # Build the prompt with video
            contents = [video_file, prompt]

            # Generate response
            response = self._model.generate_content(
                contents,
                generation_config=self._build_generation_config(generation_config),
                safety_settings=self._build_safety_settings(),
            )

            # Check for blocked content
            if not response.candidates:
                raise SafetyBlockedError("Content was blocked by safety filters")

            # Extract text
            text = response.text

            return VideoAnalysisResult(
                captions=text,
                parsed_events=self.parse_events(text),
                usage=self._extract_usage(response),
                safety_ratings=self._extract_safety_ratings(response),
                raw_response=text,
            )

        except SafetyBlockedError:
            raise
        except Exception as e:
            error_str = str(e).lower()
            if "safety" in error_str or "blocked" in error_str:
                raise SafetyBlockedError(f"Content blocked: {e}") from e
            if "context" in error_str or "length" in error_str or "token" in error_str:
                raise ContextLengthExceededError(f"Context length exceeded: {e}") from e
            raise GenerationError(f"Generation failed: {e}") from e

    async def analyze_video(
        self,
        file_uri: str,
        prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> VideoAnalysisResult:
        """
        Analyze video using Gemini's native video understanding.

        Args:
            file_uri: Gemini File API URI (e.g., "files/abc123")
            prompt: Analysis prompt (uses default if not provided)
            generation_config: Override default generation config

        Returns:
            VideoAnalysisResult with captions and metadata
        """
        prompt = prompt or self.DEFAULT_CAPTION_PROMPT

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            self._analyze_sync,
            file_uri,
            prompt,
            generation_config,
        )

    def _format_context(self, context: list[dict]) -> str:
        """Format context from previous chunks."""
        formatted = []
        for ctx in context:
            formatted.append(
                f"Chunk {ctx.get('chunk_idx', '?')} "
                f"({ctx.get('start_time', '?')} - {ctx.get('end_time', '?')}):\n"
                f"{ctx.get('captions', '')}"
            )
        return "\n\n".join(formatted)

    async def analyze_video_with_context(
        self,
        file_uri: str,
        prompt: str,
        context: list[dict],
        generation_config: Optional[GenerationConfig] = None,
    ) -> VideoAnalysisResult:
        """
        Analyze video with context from previous chunks.

        Args:
            file_uri: Gemini File API URI
            prompt: Analysis prompt
            context: List of previous chunks' captions
                     Format: [{"chunk_idx": 0, "start_time": "00:00:00",
                              "end_time": "00:01:00", "captions": "..."}]
            generation_config: Override default generation config

        Returns:
            VideoAnalysisResult with captions and metadata
        """
        context_prompt = f"""Previous video segments analysis:
{self._format_context(context)}

Now analyze the current video segment. Continue from where the previous segment ended.
Maintain consistency with previously identified objects and people.

{prompt}"""

        return await self.analyze_video(
            file_uri=file_uri,
            prompt=context_prompt,
            generation_config=generation_config,
        )

    async def analyze_video_batch(
        self,
        file_uris: list[str],
        prompt: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
    ) -> list[VideoAnalysisResult]:
        """
        Analyze multiple videos concurrently.

        Args:
            file_uris: List of Gemini File API URIs
            prompt: Analysis prompt
            generation_config: Override default generation config

        Returns:
            List of VideoAnalysisResult for each video
        """
        tasks = [
            self.analyze_video(uri, prompt, generation_config)
            for uri in file_uris
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    @staticmethod
    def get_model_info() -> tuple[str, str, str]:
        """Get model information (id, api_type, owned_by)."""
        return "gemini-2.0-flash", "google-ai", "Google"
