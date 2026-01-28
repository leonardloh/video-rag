# Gemini LLM Specification

## Overview

The `GeminiLLM` class provides text generation capabilities for RAG operations including summarization, chat, and notifications. This replaces the NVIDIA NIM-based Llama 3.1 70B model used in the original VSS engine.

## Gap Analysis

### Original Implementation
- `src/vss-engine/config/config.yaml` - Defines `chat_llm`, `summarization_llm`, `notification_llm` tools
- Uses OpenAI-compatible API with NVIDIA NIMs
- Model: `meta/llama-3.1-70b-instruct`
- Base URL: `https://integrate.api.nvidia.com/v1`

### PoC Requirement
- Use Gemini 3.0 Pro for all LLM tasks
- Support summarization, chat, and notification generation
- Maintain compatibility with CA-RAG context manager interface

## Component Location

```
./src/models/gemini/gemini_llm.py
```

## Dependencies

```python
# From requirements.txt
google-generativeai>=0.8.0
langchain-google-genai>=1.0.0  # Optional: for LangChain integration
```

## Data Classes

```python
from dataclasses import dataclass, field
from typing import Optional, Literal


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
```

## Class Interface

```python
from typing import Optional, AsyncIterator

import google.generativeai as genai


class GeminiLLM:
    """Gemini LLM for summarization and chat."""

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        generation_config: Optional[LLMGenerationConfig] = None,
    ):
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
        pass

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
        pass

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
        pass

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
        pass

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
            prompt_template: Template for summarization prompt
            generation_config: Override default generation config

        Returns:
            Summarized text
        """
        pass

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
            aggregation_prompt: Prompt for aggregation
            generation_config: Override default generation config

        Returns:
            Final aggregated summary
        """
        pass

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
        pass

    def _build_generation_config(
        self,
        override: Optional[LLMGenerationConfig] = None,
    ) -> dict:
        """Build generation config dict for API call."""
        pass

    def _convert_messages(self, messages: list[Message]) -> list[dict]:
        """Convert Message objects to Gemini format."""
        pass

    @staticmethod
    def get_model_info() -> tuple[str, str, str]:
        """Get model information (id, api_type, owned_by)."""
        return "gemini-2.0-flash", "google-ai", "Google"
```

## Implementation Notes

### Single-Turn Generation

```python
async def generate(
    self,
    prompt: str,
    system_prompt: Optional[str] = None,
    generation_config: Optional[LLMGenerationConfig] = None,
) -> GenerationResult:
    config = self._build_generation_config(generation_config)

    # Build content with optional system instruction
    if system_prompt:
        model = genai.GenerativeModel(
            self._model_name,
            system_instruction=system_prompt,
        )
    else:
        model = self._model

    response = await model.generate_content_async(
        prompt,
        generation_config=config,
    )

    return GenerationResult(
        text=response.text,
        usage=self._extract_usage(response),
        finish_reason=self._get_finish_reason(response),
        safety_ratings=self._extract_safety_ratings(response),
    )
```

### Multi-Turn Chat

```python
async def chat(
    self,
    messages: list[Message],
    system_prompt: Optional[str] = None,
    generation_config: Optional[LLMGenerationConfig] = None,
) -> GenerationResult:
    config = self._build_generation_config(generation_config)

    # Create chat session
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
        history.append({
            "role": "user" if msg.role == "user" else "model",
            "parts": [msg.content],
        })

    chat = model.start_chat(history=history)

    # Send the last message
    response = await chat.send_message_async(
        messages[-1].content,
        generation_config=config,
    )

    return GenerationResult(
        text=response.text,
        usage=self._extract_usage(response),
        finish_reason=self._get_finish_reason(response),
        safety_ratings=self._extract_safety_ratings(response),
    )
```

### Batch Summarization

```python
async def summarize_captions(
    self,
    captions: list[str],
    prompt_template: str,
    generation_config: Optional[LLMGenerationConfig] = None,
) -> str:
    # Format captions into prompt
    captions_text = "\n".join([
        f"- {caption}" for caption in captions
    ])

    prompt = prompt_template.format(captions=captions_text)

    result = await self.generate(
        prompt=prompt,
        generation_config=generation_config,
    )

    return result.text
```

### Notification Check

```python
async def check_notification(
    self,
    caption: str,
    events: list[str],
    notification_prompt: str,
    generation_config: Optional[LLMGenerationConfig] = None,
) -> tuple[bool, list[str], str]:
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
    # Expected format: "DETECTED: event1, event2" or "NO EVENTS DETECTED"
    detected_events = []
    should_notify = False

    if "DETECTED:" in result.text.upper():
        should_notify = True
        # Extract event names from response
        for event in events:
            if event.lower() in result.text.lower():
                detected_events.append(event)

    return should_notify, detected_events, result.text
```

## Configuration

```yaml
# config/config.yaml
gemini:
  llm:
    model: "gemini-2.0-flash"
    generation_config:
      temperature: 0.3
      top_p: 0.9
      max_output_tokens: 4096
```

## Environment Variables

```bash
GEMINI_API_KEY=your_api_key_here
GEMINI_LLM_MODEL=gemini-2.0-flash
GEMINI_LLM_TEMPERATURE=0.3
GEMINI_LLM_MAX_TOKENS=4096
```

## Prompts

### Summarization Prompt (config/prompts/summarization.txt)

```
Given the following timestamped captions from a video, create a structured summary.
Group related events together and identify:
- Key activities and their timeframes
- Notable incidents or anomalies
- Patterns or recurring events
Format as bullet points with time ranges.

Captions:
{captions}
```

### Chat System Prompt (config/prompts/chat.txt)

```
You are a video analysis assistant. You have access to detailed captions
and object detection data from a video. Answer questions accurately based
on the provided context. If information is not available in the context,
say so clearly. Always reference specific timestamps when relevant.
```

### Notification Prompt

```
Analyze the following video caption and determine if any of these events occurred:
Events to detect: {events}

Caption: {caption}

If any events are detected, respond with:
DETECTED: [list of detected events]
Explanation: [brief explanation]

If no events are detected, respond with:
NO EVENTS DETECTED
```

## Integration with RAG Pipeline

```python
# ./src/rag/summarization.py

class Summarizer:
    """Batch summarization using Gemini LLM."""

    def __init__(self, llm: GeminiLLM, batch_size: int = 6):
        self._llm = llm
        self._batch_size = batch_size

    async def summarize_video(
        self,
        captions: list[str],
        caption_summarization_prompt: str,
        summary_aggregation_prompt: str,
    ) -> str:
        """Summarize all captions from a video."""
        # Batch summarization
        batch_summaries = []
        for i in range(0, len(captions), self._batch_size):
            batch = captions[i:i + self._batch_size]
            summary = await self._llm.summarize_captions(
                captions=batch,
                prompt_template=caption_summarization_prompt,
            )
            batch_summaries.append(summary)

        # Aggregate summaries
        if len(batch_summaries) == 1:
            return batch_summaries[0]

        final_summary = await self._llm.aggregate_summaries(
            summaries=batch_summaries,
            aggregation_prompt=summary_aggregation_prompt,
        )

        return final_summary
```

## LangChain Integration (Optional)

```python
# ./src/models/gemini/gemini_langchain.py

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class GeminiLangChainLLM:
    """LangChain-compatible Gemini LLM wrapper."""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        self._llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
        )

    async def invoke(self, messages: list) -> str:
        response = await self._llm.ainvoke(messages)
        return response.content
```

## Testing

```python
# tests/test_gemini_llm.py

import pytest
from poc.src.models.gemini.gemini_llm import GeminiLLM, Message


class TestGeminiLLM:
    async def test_generate(self):
        """Test single-turn generation."""
        pass

    async def test_chat(self):
        """Test multi-turn chat."""
        pass

    async def test_summarize_captions(self):
        """Test caption summarization."""
        pass

    async def test_check_notification(self):
        """Test notification event detection."""
        pass

    async def test_streaming(self):
        """Test streaming generation."""
        pass
```

## Comparison with Original

| Feature | Original (Llama 3.1 70B) | PoC (Gemini 3.0 Pro) |
|---------|--------------------------|----------------------|
| Provider | NVIDIA NIM | Google AI Studio |
| API | OpenAI-compatible | Gemini SDK |
| Context | 128K tokens | 1M+ tokens |
| Streaming | Yes | Yes |
| Cost | NVIDIA API credits | Google API credits |
| Latency | ~1-3s | ~1-3s |
