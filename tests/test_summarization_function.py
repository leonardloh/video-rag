"""Unit tests for SummarizationFunction RAG wrapper."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.rag.functions import (
    FunctionConfig,
    FunctionStatus,
    SummarizationFunction,
    SummarizationInput,
    SummarizationOutput,
)


class TestSummarizationInput:
    """Tests for SummarizationInput dataclass."""

    def test_create_input(self) -> None:
        """Test creating SummarizationInput."""
        input_data = SummarizationInput(
            captions=["Caption 1", "Caption 2", "Caption 3"],
            metadata=[
                {"start_time": "00:00:00", "end_time": "00:01:00"},
                {"start_time": "00:01:00", "end_time": "00:02:00"},
                {"start_time": "00:02:00", "end_time": "00:03:00"},
            ],
        )

        assert len(input_data.captions) == 3
        assert len(input_data.metadata) == 3

    def test_default_values(self) -> None:
        """Test default values."""
        input_data = SummarizationInput(captions=["Caption 1"])

        assert input_data.metadata == []


class TestSummarizationOutput:
    """Tests for SummarizationOutput dataclass."""

    def test_create_output(self) -> None:
        """Test creating SummarizationOutput."""
        output = SummarizationOutput(
            summary="Final summary of the video.",
            batch_summaries=["Batch 1 summary", "Batch 2 summary"],
            batch_count=2,
        )

        assert output.summary == "Final summary of the video."
        assert len(output.batch_summaries) == 2
        assert output.batch_count == 2

    def test_default_values(self) -> None:
        """Test default values."""
        output = SummarizationOutput(summary="Summary")

        assert output.batch_summaries == []
        assert output.batch_count == 0


class TestSummarizationFunction:
    """Tests for SummarizationFunction class."""

    @pytest.fixture
    def mock_llm(self) -> AsyncMock:
        """Create mock LLM."""
        mock = AsyncMock()
        mock.generate = AsyncMock(return_value=MagicMock(text="Generated summary"))
        return mock

    @pytest.fixture
    def function(self, mock_llm: AsyncMock) -> SummarizationFunction:
        """Create SummarizationFunction instance."""
        return SummarizationFunction(llm=mock_llm)

    def test_init(self, mock_llm: AsyncMock) -> None:
        """Test SummarizationFunction initialization."""
        func = SummarizationFunction(llm=mock_llm)

        assert func.name == "summarization"
        assert func.batch_size == 6

    def test_init_with_config(self, mock_llm: AsyncMock) -> None:
        """Test initialization with custom config."""
        config = FunctionConfig(name="custom_summarization", enabled=True)
        func = SummarizationFunction(
            llm=mock_llm,
            config=config,
            batch_size=10,
            summarization_prompt="Custom prompt: {captions}",
            aggregation_prompt="Custom aggregation: {summaries}",
        )

        assert func.name == "custom_summarization"
        assert func.batch_size == 10

    def test_configure(self, function: SummarizationFunction) -> None:
        """Test configure method."""
        function.configure(
            batch_size=12,
            summarization_prompt="New prompt: {captions}",
            aggregation_prompt="New aggregation: {summaries}",
        )

        assert function.batch_size == 12
        assert "New prompt" in function._summarization_prompt
        assert "New aggregation" in function._aggregation_prompt

    @pytest.mark.asyncio
    async def test_execute_empty_captions(
        self,
        function: SummarizationFunction,
    ) -> None:
        """Test executing with empty captions."""
        input_data = SummarizationInput(captions=[])

        result = await function.execute(input_data)

        assert result.success is True
        assert result.output is not None
        assert "No captions available" in result.output.summary
        assert result.output.batch_count == 0
        assert function.status == FunctionStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_single_batch(
        self,
        function: SummarizationFunction,
        mock_llm: AsyncMock,
    ) -> None:
        """Test executing with captions that fit in single batch."""
        mock_llm.generate.return_value = MagicMock(text="Summary of batch 1")

        input_data = SummarizationInput(
            captions=["Caption 1", "Caption 2", "Caption 3"],
            metadata=[
                {"start_time": "00:00:00", "end_time": "00:01:00"},
                {"start_time": "00:01:00", "end_time": "00:02:00"},
                {"start_time": "00:02:00", "end_time": "00:03:00"},
            ],
        )

        result = await function.execute(input_data)

        assert result.success is True
        assert result.output is not None
        assert result.output.summary == "Summary of batch 1"
        assert result.output.batch_count == 1
        # Only one call for single batch (no aggregation needed)
        assert mock_llm.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_execute_multiple_batches(
        self,
        mock_llm: AsyncMock,
    ) -> None:
        """Test executing with captions that span multiple batches."""
        # Create function with small batch size
        function = SummarizationFunction(llm=mock_llm, batch_size=2)

        # Mock responses for batch summaries and aggregation
        mock_llm.generate.side_effect = [
            MagicMock(text="Summary of batch 1"),
            MagicMock(text="Summary of batch 2"),
            MagicMock(text="Final aggregated summary"),
        ]

        input_data = SummarizationInput(
            captions=["Caption 1", "Caption 2", "Caption 3", "Caption 4"],
            metadata=[
                {"start_time": "00:00:00", "end_time": "00:01:00"},
                {"start_time": "00:01:00", "end_time": "00:02:00"},
                {"start_time": "00:02:00", "end_time": "00:03:00"},
                {"start_time": "00:03:00", "end_time": "00:04:00"},
            ],
        )

        result = await function.execute(input_data)

        assert result.success is True
        assert result.output is not None
        assert result.output.summary == "Final aggregated summary"
        assert result.output.batch_count == 2
        assert len(result.output.batch_summaries) == 2
        # 2 batch summaries + 1 aggregation
        assert mock_llm.generate.call_count == 3

    @pytest.mark.asyncio
    async def test_execute_with_kwargs_override(
        self,
        function: SummarizationFunction,
        mock_llm: AsyncMock,
    ) -> None:
        """Test executing with kwargs overriding defaults."""
        mock_llm.generate.return_value = MagicMock(text="Custom summary")

        input_data = SummarizationInput(
            captions=["Caption 1", "Caption 2"],
        )

        result = await function.execute(
            input_data,
            batch_size=10,
            summarization_prompt="Custom: {captions}",
        )

        assert result.success is True
        # Verify custom prompt was used
        call_args = mock_llm.generate.call_args[0][0]
        assert "Custom:" in call_args

    @pytest.mark.asyncio
    async def test_execute_failure(
        self,
        function: SummarizationFunction,
        mock_llm: AsyncMock,
    ) -> None:
        """Test handling execution failure."""
        mock_llm.generate.side_effect = Exception("LLM error")

        input_data = SummarizationInput(captions=["Caption 1"])

        result = await function.execute(input_data)

        assert result.success is False
        assert result.error is not None
        assert "LLM error" in result.error
        assert function.status == FunctionStatus.FAILED

    @pytest.mark.asyncio
    async def test_reset(self, function: SummarizationFunction) -> None:
        """Test reset method."""
        function._set_status(FunctionStatus.COMPLETED)

        await function.reset()

        assert function.status == FunctionStatus.IDLE

    def test_properties(self, function: SummarizationFunction) -> None:
        """Test property accessors."""
        assert function.batch_size == 6

    @pytest.mark.asyncio
    async def test_callable(
        self,
        function: SummarizationFunction,
        mock_llm: AsyncMock,
    ) -> None:
        """Test function is callable."""
        mock_llm.generate.return_value = MagicMock(text="Summary")

        input_data = SummarizationInput(captions=["Caption 1"])

        # Call function directly
        result = await function(input_data)

        assert result.success is True


class TestSummarizationFunctionFormatting:
    """Tests for caption formatting methods."""

    @pytest.fixture
    def mock_llm(self) -> AsyncMock:
        """Create mock LLM."""
        return AsyncMock()

    @pytest.fixture
    def function(self, mock_llm: AsyncMock) -> SummarizationFunction:
        """Create SummarizationFunction instance."""
        return SummarizationFunction(llm=mock_llm)

    def test_format_captions_with_timestamps(
        self,
        function: SummarizationFunction,
    ) -> None:
        """Test formatting captions with timestamps."""
        captions = ["Caption 1", "Caption 2"]
        metadata = [
            {"start_time": "00:00:00", "end_time": "00:01:00"},
            {"start_time": "00:01:00", "end_time": "00:02:00"},
        ]

        result = function._format_captions(captions, metadata)

        assert "[00:00:00 - 00:01:00]" in result
        assert "[00:01:00 - 00:02:00]" in result
        assert "Caption 1" in result
        assert "Caption 2" in result

    def test_format_captions_without_timestamps(
        self,
        function: SummarizationFunction,
    ) -> None:
        """Test formatting captions without timestamps."""
        captions = ["Caption 1", "Caption 2"]
        metadata: list[dict] = []

        result = function._format_captions(captions, metadata)

        assert "Caption 1" in result
        assert "Caption 2" in result
        # No timestamps should be present
        assert "[" not in result or "]" not in result

    def test_format_captions_partial_metadata(
        self,
        function: SummarizationFunction,
    ) -> None:
        """Test formatting captions with partial metadata."""
        captions = ["Caption 1", "Caption 2", "Caption 3"]
        metadata = [
            {"start_time": "00:00:00", "end_time": "00:01:00"},
            # Missing metadata for caption 2 and 3
        ]

        result = function._format_captions(captions, metadata)

        assert "[00:00:00 - 00:01:00]" in result
        assert "Caption 1" in result
        assert "Caption 2" in result
        assert "Caption 3" in result


class TestSummarizationFunctionBatching:
    """Tests for batch summarization logic."""

    @pytest.fixture
    def mock_llm(self) -> AsyncMock:
        """Create mock LLM."""
        mock = AsyncMock()
        mock.generate = AsyncMock(return_value=MagicMock(text="Batch summary"))
        return mock

    @pytest.mark.asyncio
    async def test_summarize_batches_single(self, mock_llm: AsyncMock) -> None:
        """Test summarizing single batch."""
        function = SummarizationFunction(llm=mock_llm, batch_size=10)

        summaries = await function._summarize_batches(
            captions=["Caption 1", "Caption 2"],
            metadata=[],
            batch_size=10,
            prompt="Summarize: {captions}",
        )

        assert len(summaries) == 1
        assert mock_llm.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_summarize_batches_multiple(self, mock_llm: AsyncMock) -> None:
        """Test summarizing multiple batches."""
        function = SummarizationFunction(llm=mock_llm, batch_size=2)

        mock_llm.generate.side_effect = [
            MagicMock(text="Batch 1 summary"),
            MagicMock(text="Batch 2 summary"),
            MagicMock(text="Batch 3 summary"),
        ]

        summaries = await function._summarize_batches(
            captions=["C1", "C2", "C3", "C4", "C5"],
            metadata=[],
            batch_size=2,
            prompt="Summarize: {captions}",
        )

        assert len(summaries) == 3
        assert mock_llm.generate.call_count == 3

    @pytest.mark.asyncio
    async def test_aggregate_summaries(self, mock_llm: AsyncMock) -> None:
        """Test aggregating summaries."""
        function = SummarizationFunction(llm=mock_llm)

        mock_llm.generate.return_value = MagicMock(text="Aggregated summary")

        result = await function._aggregate_summaries(
            summaries=["Summary 1", "Summary 2"],
            prompt="Aggregate: {summaries}",
        )

        assert result == "Aggregated summary"
        # Verify prompt contains batch labels
        call_args = mock_llm.generate.call_args[0][0]
        assert "Batch 1" in call_args
        assert "Batch 2" in call_args


class TestSummarizationFunctionMetadata:
    """Tests for result metadata."""

    @pytest.fixture
    def mock_llm(self) -> AsyncMock:
        """Create mock LLM."""
        mock = AsyncMock()
        mock.generate = AsyncMock(return_value=MagicMock(text="Summary"))
        return mock

    @pytest.mark.asyncio
    async def test_result_metadata(self, mock_llm: AsyncMock) -> None:
        """Test that result includes metadata."""
        function = SummarizationFunction(llm=mock_llm)

        input_data = SummarizationInput(
            captions=["Caption 1", "Caption 2", "Caption 3"],
        )

        result = await function.execute(input_data)

        assert result.success is True
        assert "caption_count" in result.metadata
        assert result.metadata["caption_count"] == 3
