"""Base classes for RAG functions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

# Type variables for generic function input/output
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


class FunctionStatus(Enum):
    """Status of a RAG function."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class FunctionConfig:
    """Configuration for a RAG function."""

    name: str
    enabled: bool = True
    params: dict[str, Any] = field(default_factory=dict)
    tools: dict[str, str] = field(default_factory=dict)  # tool_name -> tool_id mapping


@dataclass
class FunctionResult(Generic[OutputT]):
    """Result from a RAG function execution."""

    success: bool
    output: OutputT | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, output: OutputT, **metadata: Any) -> FunctionResult[OutputT]:
        """Create a successful result."""
        return cls(success=True, output=output, metadata=metadata)

    @classmethod
    def fail(cls, error: str, **metadata: Any) -> FunctionResult[OutputT]:
        """Create a failed result."""
        return cls(success=False, error=error, metadata=metadata)


class RAGFunction(ABC, Generic[InputT, OutputT]):
    """Abstract base class for RAG functions.

    RAG functions are modular components that can be registered with the
    ContextManager to perform specific operations like:
    - Graph ingestion
    - Graph retrieval
    - Batch summarization
    - Entity extraction
    """

    def __init__(self, config: FunctionConfig | None = None) -> None:
        """
        Initialize the RAG function.

        Args:
            config: Function configuration
        """
        self._config = config or FunctionConfig(name=self.__class__.__name__)
        self._status = FunctionStatus.IDLE

    @property
    def name(self) -> str:
        """Get function name."""
        return self._config.name

    @property
    def status(self) -> FunctionStatus:
        """Get current status."""
        return self._status

    @property
    def config(self) -> FunctionConfig:
        """Get function configuration."""
        return self._config

    @property
    def is_enabled(self) -> bool:
        """Check if function is enabled."""
        return self._config.enabled

    def configure(self, **kwargs: Any) -> None:
        """
        Update function configuration.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if key in ("name", "enabled"):
                setattr(self._config, key, value)
            else:
                self._config.params[key] = value

    @abstractmethod
    async def execute(self, input_data: InputT, **kwargs: Any) -> FunctionResult[OutputT]:
        """
        Execute the function.

        Args:
            input_data: Input data for the function
            **kwargs: Additional execution parameters

        Returns:
            FunctionResult with output or error
        """
        pass

    async def reset(self) -> None:
        """Reset function state."""
        self._status = FunctionStatus.IDLE

    def _set_status(self, status: FunctionStatus) -> None:
        """Set function status."""
        self._status = status

    async def __call__(self, input_data: InputT, **kwargs: Any) -> FunctionResult[OutputT]:
        """Allow calling function directly."""
        return await self.execute(input_data, **kwargs)


class CompositeFunction(RAGFunction[InputT, OutputT]):
    """A function that composes multiple RAG functions."""

    def __init__(
        self,
        functions: list[RAGFunction[Any, Any]],
        config: FunctionConfig | None = None,
    ) -> None:
        """
        Initialize composite function.

        Args:
            functions: List of functions to compose
            config: Function configuration
        """
        super().__init__(config)
        self._functions = functions

    @property
    def functions(self) -> list[RAGFunction[Any, Any]]:
        """Get composed functions."""
        return self._functions

    def add_function(self, function: RAGFunction[Any, Any]) -> None:
        """Add a function to the composition."""
        self._functions.append(function)

    def remove_function(self, name: str) -> bool:
        """Remove a function by name."""
        for i, func in enumerate(self._functions):
            if func.name == name:
                self._functions.pop(i)
                return True
        return False

    async def execute(self, input_data: InputT, **kwargs: Any) -> FunctionResult[OutputT]:
        """
        Execute all composed functions in sequence.

        Args:
            input_data: Input data
            **kwargs: Additional parameters

        Returns:
            Result from the last function
        """
        self._set_status(FunctionStatus.RUNNING)
        current_data: Any = input_data
        last_result: FunctionResult[Any] | None = None

        try:
            for func in self._functions:
                if not func.is_enabled:
                    continue

                result = await func.execute(current_data, **kwargs)
                last_result = result

                if not result.success:
                    self._set_status(FunctionStatus.FAILED)
                    return FunctionResult.fail(
                        f"Function '{func.name}' failed: {result.error}"
                    )

                # Pass output to next function
                current_data = result.output

            self._set_status(FunctionStatus.COMPLETED)

            if last_result is not None:
                return FunctionResult.ok(last_result.output)
            return FunctionResult.ok(current_data)

        except Exception as e:
            self._set_status(FunctionStatus.FAILED)
            return FunctionResult.fail(str(e))

    async def reset(self) -> None:
        """Reset all composed functions."""
        await super().reset()
        for func in self._functions:
            await func.reset()
