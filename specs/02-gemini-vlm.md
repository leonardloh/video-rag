# Gemini VLM Specification

## Overview

The `GeminiVLM` class provides video understanding capabilities using Gemini 3.0 Pro's native video analysis. This replaces the local VLM models (Cosmos Reason1, NVILA, VILA 1.5) used in the original VSS engine.

## Gap Analysis

### Original Implementation
- `src/vss-engine/src/models/cosmos_reason1/cosmos_reason1_model.py` - Local Cosmos Reason1 model
- `src/vss-engine/src/models/nvila/nvila_model.py` - Local NVILA model
- `src/vss-engine/src/vlm_pipeline/vlm_pipeline.py` - `VlmProcess` orchestrates VLM inference
- Frame-based processing with timestamp overlay
- TensorRT/VLLM for local inference

### PoC Requirement
- Use Gemini 3.0 Pro for native video understanding
- Send uploaded video file URI with analysis prompt
- Parse timestamped captions from response
- Support context from previous chunks for continuity

## Component Location

```
./src/models/gemini/gemini_vlm.py
```

## Dependencies

```python
# From requirements.txt
google-generativeai>=0.8.0
```

## Data Classes

```python
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VideoEvent:
    """A single event detected in the video."""
    start_time: str              # HH:MM:SS format
    end_time: str                # HH:MM:SS format
    description: str             # Event description
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
    category: str                # e.g., "HARM_CATEGORY_HARASSMENT"
    probability: str             # e.g., "NEGLIGIBLE", "LOW", "MEDIUM", "HIGH"
    blocked: bool = False


@dataclass
class VideoAnalysisResult:
    """Result of video analysis."""
    captions: str                           # Raw caption text
    parsed_events: list[VideoEvent]         # Structured events
    usage: TokenUsage                       # Token usage
    safety_ratings: list[SafetyRating]      # Safety ratings
    reasoning_description: Optional[str] = None  # Chain-of-thought reasoning if enabled
    raw_response: Optional[str] = None      # Full raw response for debugging


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
```

## Class Interface

```python
import re
from typing import Optional

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold


class GeminiVLM:
    """Gemini-based video understanding using native video upload."""

    # Default prompts
    DEFAULT_CAPTION_PROMPT = """Analyze this video segment and provide detailed captions of all events.
For each distinct event or action, provide:
- Start and end timestamps in format <HH:MM:SS>
- Detailed description of what is happening
- Any relevant objects, people, or text visible
Focus on actions, interactions, and notable occurrences."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        generation_config: Optional[GenerationConfig] = None,
        safety_settings: Optional[SafetySettings] = None,
    ):
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
        pass

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
        pass

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
        pass

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
        pass

    def _build_generation_config(
        self,
        override: Optional[GenerationConfig] = None,
    ) -> dict:
        """Build generation config dict for API call."""
        pass

    def _build_safety_settings(self) -> list[dict]:
        """Build safety settings list for API call."""
        pass

    def _extract_usage(self, response) -> TokenUsage:
        """Extract token usage from response."""
        pass

    def _extract_safety_ratings(self, response) -> list[SafetyRating]:
        """Extract safety ratings from response."""
        pass

    @staticmethod
    def get_model_info() -> tuple[str, str, str]:
        """Get model information (id, api_type, owned_by)."""
        return "gemini-2.0-flash", "google-ai", "Google"
```

## Implementation Notes

### Video Analysis Flow

1. **Prepare the request**
   ```python
   # Get the uploaded file reference
   video_file = genai.get_file(file_uri)

   # Build the prompt with video
   contents = [
       video_file,
       prompt or self.DEFAULT_CAPTION_PROMPT,
   ]
   ```

2. **Generate response**
   ```python
   response = await self._model.generate_content_async(
       contents,
       generation_config=self._build_generation_config(),
       safety_settings=self._build_safety_settings(),
   )
   ```

3. **Parse and return results**
   ```python
   return VideoAnalysisResult(
       captions=response.text,
       parsed_events=self.parse_events(response.text),
       usage=self._extract_usage(response),
       safety_ratings=self._extract_safety_ratings(response),
   )
   ```

### Context-Aware Analysis

For multi-chunk videos, provide context from previous chunks:

```python
context_prompt = f"""Previous video segments analysis:
{self._format_context(context)}

Now analyze the current video segment. Continue from where the previous segment ended.
Maintain consistency with previously identified objects and people.

{prompt}"""
```

### Event Parsing

Support multiple timestamp formats:

```python
TIMESTAMP_PATTERNS = [
    # <HH:MM:SS> <HH:MM:SS> description
    r'<(\d{2}:\d{2}:\d{2})>\s*<(\d{2}:\d{2}:\d{2})>\s*(.+)',
    # HH:MM:SS - HH:MM:SS: description
    r'(\d{2}:\d{2}:\d{2})\s*-\s*(\d{2}:\d{2}:\d{2}):\s*(.+)',
    # [HH:MM:SS-HH:MM:SS] description
    r'\[(\d{2}:\d{2}:\d{2})-(\d{2}:\d{2}:\d{2})\]\s*(.+)',
    # HH:MM:SS.mmm format
    r'<(\d{2}:\d{2}:\d{2}\.\d+)>\s*<(\d{2}:\d{2}:\d{2}\.\d+)>\s*(.+)',
]

def parse_events(self, caption_text: str) -> list[VideoEvent]:
    events = []
    for line in caption_text.strip().split('\n'):
        for pattern in self.TIMESTAMP_PATTERNS:
            match = re.match(pattern, line.strip())
            if match:
                events.append(VideoEvent(
                    start_time=match.group(1),
                    end_time=match.group(2),
                    description=match.group(3).strip(),
                ))
                break
    return events
```

### Error Handling

```python
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
```

## Configuration

```yaml
# config/config.yaml
gemini:
  vlm:
    model: "gemini-2.0-flash"
    generation_config:
      temperature: 0.2
      top_p: 0.8
      top_k: 40
      max_output_tokens: 2048
    safety_settings:
      harassment: BLOCK_ONLY_HIGH
      hate_speech: BLOCK_ONLY_HIGH
      sexually_explicit: BLOCK_ONLY_HIGH
      dangerous_content: BLOCK_ONLY_HIGH
```

## Environment Variables

```bash
GEMINI_API_KEY=your_api_key_here
GEMINI_MODEL=gemini-2.0-flash
GEMINI_VLM_TEMPERATURE=0.2
GEMINI_VLM_MAX_TOKENS=2048
```

## Prompts

### Caption Prompt (config/prompts/caption.txt)

```
Analyze this video segment and provide detailed captions of all events.
For each distinct event or action, provide:
- Start and end timestamps in format <HH:MM:SS>
- Detailed description of what is happening
- Any relevant objects, people, or text visible
Focus on actions, interactions, and notable occurrences.
```

### Warehouse-Specific Prompt

```
Write a concise and clear dense caption for the provided warehouse video,
focusing on irregular or hazardous events such as:
- boxes falling
- workers not wearing PPE
- workers falling
- workers taking photographs
- workers chitchatting
- forklift stuck

Start and end each sentence with a timestamp in format <HH:MM:SS>.
```

## Integration with VLM Pipeline

```python
# ./src/vlm_pipeline/gemini_video_analyzer.py

class GeminiVideoAnalyzer:
    """Orchestrates video analysis using Gemini VLM."""

    def __init__(
        self,
        file_manager: GeminiFileManager,
        vlm: GeminiVLM,
    ):
        self._file_manager = file_manager
        self._vlm = vlm
        self._context: list[dict] = []

    async def analyze_chunk(
        self,
        chunk: ChunkInfo,
        prompt: str,
        use_context: bool = True,
    ) -> VideoAnalysisResult:
        """Analyze a single video chunk."""
        # Upload chunk to Gemini
        file_result = await self._file_manager.upload_and_wait(chunk.file)

        try:
            # Analyze with or without context
            if use_context and self._context:
                result = await self._vlm.analyze_video_with_context(
                    file_uri=file_result.uri,
                    prompt=prompt,
                    context=self._context,
                )
            else:
                result = await self._vlm.analyze_video(
                    file_uri=file_result.uri,
                    prompt=prompt,
                )

            # Update context for next chunk
            self._context.append({
                "chunk_idx": chunk.chunkIdx,
                "start_time": chunk.start_ntp,
                "end_time": chunk.end_ntp,
                "captions": result.captions,
            })

            return result
        finally:
            # Clean up uploaded file
            await self._file_manager.delete_file(file_result.uri)

    def reset_context(self):
        """Reset context for new video."""
        self._context = []
```

## Testing

```python
# tests/test_gemini_vlm.py

import pytest
from poc.src.models.gemini.gemini_vlm import GeminiVLM, VideoEvent


class TestGeminiVLM:
    def test_parse_events_angle_brackets(self):
        """Test parsing events with angle bracket format."""
        vlm = GeminiVLM(api_key="test")
        text = "<00:00:05> <00:00:10> Person walks across room"
        events = vlm.parse_events(text)
        assert len(events) == 1
        assert events[0].start_time == "00:00:05"
        assert events[0].end_time == "00:00:10"

    def test_parse_events_dash_format(self):
        """Test parsing events with dash format."""
        vlm = GeminiVLM(api_key="test")
        text = "00:00:05 - 00:00:10: Person walks across room"
        events = vlm.parse_events(text)
        assert len(events) == 1

    async def test_analyze_video(self):
        """Test video analysis."""
        pass

    async def test_analyze_with_context(self):
        """Test context-aware analysis."""
        pass
```

## Comparison with Original

| Feature | Original (Cosmos Reason1) | PoC (Gemini VLM) |
|---------|---------------------------|------------------|
| Input | Extracted frames with timestamp overlay | Native video file |
| Processing | Local GPU (TRT/VLLM) | Cloud API |
| Context | Frame-level embeddings | Text-based context |
| Timestamps | Overlay on frames | Extracted from video |
| Batching | Local batch processing | Concurrent API calls |
| Cost | GPU compute | API usage |
