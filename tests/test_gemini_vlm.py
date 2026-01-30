"""Tests for Gemini VLM."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.models.gemini.gemini_vlm import (
    ContextLengthExceededError,
    GenerationConfig,
    GenerationError,
    GeminiVLM,
    SafetyBlockedError,
    SafetyRating,
    SafetySettings,
    TokenUsage,
    VideoAnalysisResult,
    VideoEvent,
    VLMError,
)


class TestGeminiVLM:
    """Tests for GeminiVLM class."""

    @pytest.fixture
    def vlm(self) -> GeminiVLM:
        """Create a VLM instance."""
        with patch("google.generativeai.configure"):
            with patch("google.generativeai.GenerativeModel"):
                return GeminiVLM(api_key="test_key")

    def test_init(self, vlm: GeminiVLM) -> None:
        """Test initialization."""
        assert vlm._api_key == "test_key"
        assert vlm._model_name == "gemini-2.0-flash"

    def test_init_with_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = GenerationConfig(temperature=0.5, max_output_tokens=1024)
        safety = SafetySettings(harassment="BLOCK_NONE")

        with patch("google.generativeai.configure"):
            with patch("google.generativeai.GenerativeModel"):
                vlm = GeminiVLM(
                    api_key="test_key",
                    model="gemini-pro",
                    generation_config=config,
                    safety_settings=safety,
                )

        assert vlm._model_name == "gemini-pro"
        assert vlm._generation_config.temperature == 0.5
        assert vlm._safety_settings.harassment == "BLOCK_NONE"

    def test_parse_events_angle_brackets(self, vlm: GeminiVLM) -> None:
        """Test parsing events with angle bracket format."""
        text = "<00:00:05> <00:00:10> Person walks across room"
        events = vlm.parse_events(text)

        assert len(events) == 1
        assert events[0].start_time == "00:00:05"
        assert events[0].end_time == "00:00:10"
        assert events[0].description == "Person walks across room"

    def test_parse_events_dash_format(self, vlm: GeminiVLM) -> None:
        """Test parsing events with dash format."""
        text = "00:00:05 - 00:00:10: Person walks across room"
        events = vlm.parse_events(text)

        assert len(events) == 1
        assert events[0].start_time == "00:00:05"
        assert events[0].end_time == "00:00:10"

    def test_parse_events_bracket_format(self, vlm: GeminiVLM) -> None:
        """Test parsing events with bracket format."""
        text = "[00:00:05-00:00:10] Person walks across room"
        events = vlm.parse_events(text)

        assert len(events) == 1
        assert events[0].start_time == "00:00:05"
        assert events[0].end_time == "00:00:10"

    def test_parse_events_multiple(self, vlm: GeminiVLM) -> None:
        """Test parsing multiple events."""
        text = """<00:00:05> <00:00:10> Person walks across room
<00:00:15> <00:00:20> Car drives by
<00:00:25> <00:00:30> Dog runs in park"""
        events = vlm.parse_events(text)

        assert len(events) == 3
        assert events[0].description == "Person walks across room"
        assert events[1].description == "Car drives by"
        assert events[2].description == "Dog runs in park"

    def test_parse_events_empty(self, vlm: GeminiVLM) -> None:
        """Test parsing empty text."""
        events = vlm.parse_events("")
        assert len(events) == 0

    def test_parse_events_no_timestamps(self, vlm: GeminiVLM) -> None:
        """Test parsing text without timestamps."""
        text = "Person walks across room"
        events = vlm.parse_events(text)
        assert len(events) == 0

    def test_parse_events_milliseconds(self, vlm: GeminiVLM) -> None:
        """Test parsing events with milliseconds."""
        text = "<00:00:05.123> <00:00:10.456> Person walks across room"
        events = vlm.parse_events(text)

        assert len(events) == 1
        assert events[0].start_time == "00:00:05.123"
        assert events[0].end_time == "00:00:10.456"

    def test_build_generation_config(self, vlm: GeminiVLM) -> None:
        """Test building generation config."""
        config = vlm._build_generation_config()

        assert "temperature" in config
        assert "top_p" in config
        assert "top_k" in config
        assert "max_output_tokens" in config

    def test_build_generation_config_override(self, vlm: GeminiVLM) -> None:
        """Test building generation config with override."""
        override = GenerationConfig(temperature=0.9, max_output_tokens=512)
        config = vlm._build_generation_config(override)

        assert config["temperature"] == 0.9
        assert config["max_output_tokens"] == 512

    def test_build_safety_settings(self, vlm: GeminiVLM) -> None:
        """Test building safety settings."""
        settings = vlm._build_safety_settings()

        assert len(settings) == 4
        for setting in settings:
            assert "category" in setting
            assert "threshold" in setting

    @pytest.mark.asyncio
    async def test_analyze_video(self, vlm: GeminiVLM) -> None:
        """Test video analysis."""
        mock_response = MagicMock()
        mock_response.text = "<00:00:05> <00:00:10> Person walks"
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].safety_ratings = []
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.total_token_count = 150

        mock_file = MagicMock()

        with patch("google.generativeai.get_file", return_value=mock_file):
            vlm._model.generate_content = MagicMock(return_value=mock_response)
            result = await vlm.analyze_video("files/test123")

        assert result.captions == "<00:00:05> <00:00:10> Person walks"
        assert len(result.parsed_events) == 1
        assert result.usage.total_tokens == 150

    @pytest.mark.asyncio
    async def test_analyze_video_with_context(self, vlm: GeminiVLM) -> None:
        """Test video analysis with context."""
        mock_response = MagicMock()
        mock_response.text = "<00:00:05> <00:00:10> Person continues walking"
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].safety_ratings = []
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.total_token_count = 150

        mock_file = MagicMock()

        context = [
            {
                "chunk_idx": 0,
                "start_time": "00:00:00",
                "end_time": "00:01:00",
                "captions": "Person enters room",
            }
        ]

        with patch("google.generativeai.get_file", return_value=mock_file):
            vlm._model.generate_content = MagicMock(return_value=mock_response)
            result = await vlm.analyze_video_with_context(
                "files/test123", "Analyze video", context
            )

        assert "continues walking" in result.captions


class TestVideoEvent:
    """Tests for VideoEvent dataclass."""

    def test_create_event(self) -> None:
        """Test creating a video event."""
        event = VideoEvent(
            start_time="00:00:05",
            end_time="00:00:10",
            description="Person walks",
            confidence=0.95,
        )

        assert event.start_time == "00:00:05"
        assert event.end_time == "00:00:10"
        assert event.description == "Person walks"
        assert event.confidence == 0.95


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_create_usage(self) -> None:
        """Test creating token usage."""
        usage = TokenUsage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
        )

        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150


class TestGenerationConfig:
    """Tests for GenerationConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default generation config."""
        config = GenerationConfig()

        assert config.temperature == 0.2
        assert config.top_p == 0.8
        assert config.top_k == 40
        assert config.max_output_tokens == 2048
        assert config.stop_sequences == []

    def test_custom_config(self) -> None:
        """Test custom generation config."""
        config = GenerationConfig(
            temperature=0.9,
            max_output_tokens=512,
            stop_sequences=["END"],
        )

        assert config.temperature == 0.9
        assert config.max_output_tokens == 512
        assert config.stop_sequences == ["END"]


class TestExceptions:
    """Tests for VLM exceptions."""

    def test_vlm_error(self) -> None:
        """Test VLMError."""
        error = VLMError("Test error")
        assert str(error) == "Test error"
        assert isinstance(error, Exception)

    def test_safety_blocked_error(self) -> None:
        """Test SafetyBlockedError."""
        error = SafetyBlockedError("Content blocked")
        assert str(error) == "Content blocked"
        assert isinstance(error, VLMError)

    def test_context_length_exceeded_error(self) -> None:
        """Test ContextLengthExceededError."""
        error = ContextLengthExceededError("Too long")
        assert str(error) == "Too long"
        assert isinstance(error, VLMError)

    def test_generation_error(self) -> None:
        """Test GenerationError."""
        error = GenerationError("Generation failed")
        assert str(error) == "Generation failed"
        assert isinstance(error, VLMError)
