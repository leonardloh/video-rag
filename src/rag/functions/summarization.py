"""Batch summarization function for RAG pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .base import FunctionConfig, FunctionResult, FunctionStatus, RAGFunction

if TYPE_CHECKING:
    from ...models.gemini.gemini_llm import GeminiLLM

logger = logging.getLogger(__name__)


# Default prompts
DEFAULT_SUMMARIZATION_PROMPT = """Given the following timestamped captions from a video, create a structured summary.
Group related events together and identify:
- Key activities and their timeframes
- Notable incidents or anomalies
- Patterns or recurring events
Format as bullet points with time ranges.

Captions:
{captions}
"""

DEFAULT_AGGREGATION_PROMPT = """Aggregate the following video summaries into a final comprehensive report.
Cluster the information by category:
- Safety Issues: Any hazards, incidents, or safety concerns
- Operations: Normal operational activities
- Notable Events: Unusual or significant occurrences
- Patterns: Recurring behaviors or trends

Provide timestamps where relevant.

Summaries:
{summaries}
"""


@dataclass
class SummarizationInput:
    """Input for summarization function."""

    captions: list[str]
    metadata: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SummarizationOutput:
    """Output from summarization function."""

    summary: str
    batch_summaries: list[str] = field(default_factory=list)
    batch_count: int = 0


class SummarizationFunction(RAGFunction[SummarizationInput, SummarizationOutput]):
    """Batch summarization function for video captions.

    This function:
    1. Groups captions into batches
    2. Summarizes each batch using the LLM
    3. Aggregates batch summaries into a final summary
    """

    def __init__(
        self,
        llm: GeminiLLM,
        config: FunctionConfig | None = None,
        batch_size: int = 6,
        summarization_prompt: str | None = None,
        aggregation_prompt: str | None = None,
    ) -> None:
        """
        Initialize summarization function.

        Args:
            llm: Gemini LLM for text generation
            config: Function configuration
            batch_size: Number of captions per batch
            summarization_prompt: Prompt for batch summarization
            aggregation_prompt: Prompt for summary aggregation
        """
        super().__init__(config or FunctionConfig(name="summarization"))
        self._llm = llm
        self._batch_size = batch_size
        self._summarization_prompt = summarization_prompt or DEFAULT_SUMMARIZATION_PROMPT
        self._aggregation_prompt = aggregation_prompt or DEFAULT_AGGREGATION_PROMPT

    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return self._batch_size

    def configure(self, **kwargs: Any) -> None:
        """
        Update function configuration.

        Args:
            **kwargs: Configuration parameters
        """
        super().configure(**kwargs)

        if "batch_size" in kwargs:
            self._batch_size = kwargs["batch_size"]
        if "summarization_prompt" in kwargs:
            self._summarization_prompt = kwargs["summarization_prompt"]
        if "aggregation_prompt" in kwargs:
            self._aggregation_prompt = kwargs["aggregation_prompt"]

    async def execute(
        self,
        input_data: SummarizationInput,
        **kwargs: Any,
    ) -> FunctionResult[SummarizationOutput]:
        """
        Execute batch summarization.

        Args:
            input_data: Captions to summarize
            **kwargs: Additional parameters (batch_size, prompts)

        Returns:
            FunctionResult with summary
        """
        self._set_status(FunctionStatus.RUNNING)

        captions = input_data.captions
        metadata = input_data.metadata

        if not captions:
            self._set_status(FunctionStatus.COMPLETED)
            return FunctionResult.ok(
                SummarizationOutput(
                    summary="No captions available for summarization.",
                    batch_summaries=[],
                    batch_count=0,
                )
            )

        # Get parameters from kwargs or use defaults
        batch_size = kwargs.get("batch_size", self._batch_size)
        summarization_prompt = kwargs.get("summarization_prompt", self._summarization_prompt)
        aggregation_prompt = kwargs.get("aggregation_prompt", self._aggregation_prompt)

        try:
            # Summarize batches
            batch_summaries = await self._summarize_batches(
                captions=captions,
                metadata=metadata,
                batch_size=batch_size,
                prompt=summarization_prompt,
            )

            # Aggregate if multiple batches
            if len(batch_summaries) == 1:
                final_summary = batch_summaries[0]
            else:
                final_summary = await self._aggregate_summaries(
                    summaries=batch_summaries,
                    prompt=aggregation_prompt,
                )

            self._set_status(FunctionStatus.COMPLETED)
            return FunctionResult.ok(
                SummarizationOutput(
                    summary=final_summary,
                    batch_summaries=batch_summaries,
                    batch_count=len(batch_summaries),
                ),
                caption_count=len(captions),
            )

        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            self._set_status(FunctionStatus.FAILED)
            return FunctionResult.fail(str(e))

    async def _summarize_batches(
        self,
        captions: list[str],
        metadata: list[dict[str, Any]],
        batch_size: int,
        prompt: str,
    ) -> list[str]:
        """
        Summarize captions in batches.

        Args:
            captions: List of caption texts
            metadata: List of metadata dicts
            batch_size: Captions per batch
            prompt: Summarization prompt template

        Returns:
            List of batch summaries
        """
        batch_summaries = []

        for i in range(0, len(captions), batch_size):
            batch_captions = captions[i : i + batch_size]
            batch_metadata = metadata[i : i + batch_size] if metadata else []

            # Format captions with timestamps if available
            formatted_captions = self._format_captions(batch_captions, batch_metadata)

            # Generate summary
            summary = await self._summarize_batch(formatted_captions, prompt)
            batch_summaries.append(summary)

        return batch_summaries

    def _format_captions(
        self,
        captions: list[str],
        metadata: list[dict[str, Any]],
    ) -> str:
        """
        Format captions with timestamps.

        Args:
            captions: Caption texts
            metadata: Caption metadata

        Returns:
            Formatted caption string
        """
        formatted_parts = []

        for i, caption in enumerate(captions):
            if i < len(metadata) and metadata[i]:
                start_time = metadata[i].get("start_time", "")
                end_time = metadata[i].get("end_time", "")
                if start_time and end_time:
                    formatted_parts.append(f"[{start_time} - {end_time}]\n{caption}")
                else:
                    formatted_parts.append(caption)
            else:
                formatted_parts.append(caption)

        return "\n\n".join(formatted_parts)

    async def _summarize_batch(self, captions_text: str, prompt: str) -> str:
        """
        Summarize a single batch of captions.

        Args:
            captions_text: Formatted caption text
            prompt: Prompt template

        Returns:
            Summary text
        """
        formatted_prompt = prompt.format(captions=captions_text)

        result = await self._llm.generate(formatted_prompt)
        return result.text

    async def _aggregate_summaries(
        self,
        summaries: list[str],
        prompt: str,
    ) -> str:
        """
        Aggregate multiple batch summaries.

        Args:
            summaries: List of batch summaries
            prompt: Aggregation prompt template

        Returns:
            Final aggregated summary
        """
        summaries_text = "\n\n---\n\n".join(
            f"Batch {i + 1}:\n{summary}" for i, summary in enumerate(summaries)
        )

        formatted_prompt = prompt.format(summaries=summaries_text)

        result = await self._llm.generate(formatted_prompt)
        return result.text

    async def reset(self) -> None:
        """Reset function state."""
        await super().reset()
