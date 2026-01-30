"""Tests for Gemini LLM."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.models.gemini.gemini_llm import (
    GenerationResult,
    GeminiLLM,
    LLMGenerationConfig,
    Message,
    SafetyRating,
    TokenUsage,
)


class TestGeminiLLM:
    """Tests for GeminiLLM class."""

    @pytest.fixture
    def llm(self) -> GeminiLLM:
        """Create an LLM instance."""
        with patch("google.generativeai.configure"):
            with patch("google.generativeai.GenerativeModel"):
                return GeminiLLM(api_key="test_key")

    def test_init(self, llm: GeminiLLM) -> None:
        """Test initialization."""
        assert llm._api_key == "test_key"
        assert llm._model_name == "gemini-2.0-flash"

    def test_init_with_custom_config(self) -> None:
        """Test initialization with custom config."""
        config = LLMGenerationConfig(temperature=0.5, max_output_tokens=1024)

        with patch("google.generativeai.configure"):
            with patch("google.generativeai.GenerativeModel"):
                llm = GeminiLLM(
                    api_key="test_key",
                    model="gemini-pro",
                    generation_config=config,
                )

        assert llm._model_name == "gemini-pro"
        assert llm._generation_config.temperature == 0.5

    def test_build_generation_config(self, llm: GeminiLLM) -> None:
        """Test building generation config."""
        config = llm._build_generation_config()

        assert "temperature" in config
        assert "top_p" in config
        assert "top_k" in config
        assert "max_output_tokens" in config

    def test_build_generation_config_override(self, llm: GeminiLLM) -> None:
        """Test building generation config with override."""
        override = LLMGenerationConfig(temperature=0.9, max_output_tokens=512)
        config = llm._build_generation_config(override)

        assert config["temperature"] == 0.9
        assert config["max_output_tokens"] == 512

    def test_convert_messages(self, llm: GeminiLLM) -> None:
        """Test converting messages to Gemini format."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there"),
        ]

        converted = llm._convert_messages(messages)

        assert len(converted) == 2
        assert converted[0]["role"] == "user"
        assert converted[0]["parts"] == ["Hello"]
        assert converted[1]["role"] == "model"
        assert converted[1]["parts"] == ["Hi there"]

    @pytest.mark.asyncio
    async def test_generate(self, llm: GeminiLLM) -> None:
        """Test text generation."""
        mock_response = MagicMock()
        mock_response.text = "Generated response"
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].safety_ratings = []
        mock_response.candidates[0].finish_reason = MagicMock(name="STOP")
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 20
        mock_response.usage_metadata.total_token_count = 30

        llm._model.generate_content = MagicMock(return_value=mock_response)

        result = await llm.generate("Test prompt")

        assert result.text == "Generated response"
        assert result.usage.total_tokens == 30

    @pytest.mark.asyncio
    async def test_generate_with_system_prompt(self, llm: GeminiLLM) -> None:
        """Test generation with system prompt."""
        mock_response = MagicMock()
        mock_response.text = "Response with system"
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].safety_ratings = []
        mock_response.candidates[0].finish_reason = MagicMock(name="STOP")
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 20
        mock_response.usage_metadata.total_token_count = 30

        with patch("google.generativeai.GenerativeModel") as mock_model_class:
            mock_model = MagicMock()
            mock_model.generate_content = MagicMock(return_value=mock_response)
            mock_model_class.return_value = mock_model

            result = await llm.generate("Test prompt", system_prompt="Be helpful")

        assert result.text == "Response with system"

    @pytest.mark.asyncio
    async def test_chat(self, llm: GeminiLLM) -> None:
        """Test multi-turn chat."""
        mock_response = MagicMock()
        mock_response.text = "Chat response"
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].safety_ratings = []
        mock_response.candidates[0].finish_reason = MagicMock(name="STOP")
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 20
        mock_response.usage_metadata.total_token_count = 30

        mock_chat = MagicMock()
        mock_chat.send_message = MagicMock(return_value=mock_response)
        llm._model.start_chat = MagicMock(return_value=mock_chat)

        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi"),
            Message(role="user", content="How are you?"),
        ]

        result = await llm.chat(messages)

        assert result.text == "Chat response"

    @pytest.mark.asyncio
    async def test_summarize_captions(self, llm: GeminiLLM) -> None:
        """Test caption summarization."""
        mock_response = MagicMock()
        mock_response.text = "Summary of captions"
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].safety_ratings = []
        mock_response.candidates[0].finish_reason = MagicMock(name="STOP")
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 20
        mock_response.usage_metadata.total_token_count = 30

        llm._model.generate_content = MagicMock(return_value=mock_response)

        captions = ["Caption 1", "Caption 2", "Caption 3"]
        template = "Summarize these captions:\n{captions}"

        result = await llm.summarize_captions(captions, template)

        assert result == "Summary of captions"

    @pytest.mark.asyncio
    async def test_aggregate_summaries(self, llm: GeminiLLM) -> None:
        """Test summary aggregation."""
        mock_response = MagicMock()
        mock_response.text = "Aggregated summary"
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].safety_ratings = []
        mock_response.candidates[0].finish_reason = MagicMock(name="STOP")
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 20
        mock_response.usage_metadata.total_token_count = 30

        llm._model.generate_content = MagicMock(return_value=mock_response)

        summaries = ["Summary 1", "Summary 2"]
        prompt = "Aggregate these summaries:\n{summaries}"

        result = await llm.aggregate_summaries(summaries, prompt)

        assert result == "Aggregated summary"

    @pytest.mark.asyncio
    async def test_check_notification_detected(self, llm: GeminiLLM) -> None:
        """Test notification check with detected events."""
        mock_response = MagicMock()
        mock_response.text = "DETECTED: falling, collision"
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].safety_ratings = []
        mock_response.candidates[0].finish_reason = MagicMock(name="STOP")
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 20
        mock_response.usage_metadata.total_token_count = 30

        llm._model.generate_content = MagicMock(return_value=mock_response)

        events = ["falling", "collision", "fire"]
        prompt = "Check for events: {events}\nCaption: {caption}"

        should_notify, detected, explanation = await llm.check_notification(
            "A person is falling", events, prompt
        )

        assert should_notify is True
        assert "falling" in detected

    @pytest.mark.asyncio
    async def test_check_notification_not_detected(self, llm: GeminiLLM) -> None:
        """Test notification check with no events detected."""
        mock_response = MagicMock()
        mock_response.text = "NO EVENTS DETECTED"
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].safety_ratings = []
        mock_response.candidates[0].finish_reason = MagicMock(name="STOP")
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 20
        mock_response.usage_metadata.total_token_count = 30

        llm._model.generate_content = MagicMock(return_value=mock_response)

        events = ["falling", "collision", "fire"]
        prompt = "Check for events: {events}\nCaption: {caption}"

        should_notify, detected, explanation = await llm.check_notification(
            "A person is walking", events, prompt
        )

        assert should_notify is False
        assert len(detected) == 0


class TestMessage:
    """Tests for Message dataclass."""

    def test_create_user_message(self) -> None:
        """Test creating a user message."""
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_create_assistant_message(self) -> None:
        """Test creating an assistant message."""
        msg = Message(role="assistant", content="Hi there")
        assert msg.role == "assistant"
        assert msg.content == "Hi there"

    def test_create_system_message(self) -> None:
        """Test creating a system message."""
        msg = Message(role="system", content="Be helpful")
        assert msg.role == "system"
        assert msg.content == "Be helpful"


class TestLLMGenerationConfig:
    """Tests for LLMGenerationConfig dataclass."""

    def test_default_config(self) -> None:
        """Test default LLM generation config."""
        config = LLMGenerationConfig()

        assert config.temperature == 0.3
        assert config.top_p == 0.9
        assert config.top_k == 40
        assert config.max_output_tokens == 4096
        assert config.stop_sequences == []

    def test_custom_config(self) -> None:
        """Test custom LLM generation config."""
        config = LLMGenerationConfig(
            temperature=0.7,
            max_output_tokens=2048,
            stop_sequences=["END", "STOP"],
        )

        assert config.temperature == 0.7
        assert config.max_output_tokens == 2048
        assert config.stop_sequences == ["END", "STOP"]


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_create_result(self) -> None:
        """Test creating a generation result."""
        usage = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        result = GenerationResult(
            text="Generated text",
            usage=usage,
            finish_reason="STOP",
        )

        assert result.text == "Generated text"
        assert result.usage.total_tokens == 30
        assert result.finish_reason == "STOP"
        assert result.safety_ratings == []
