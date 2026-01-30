"""Utility functions for VSS PoC."""

from __future__ import annotations

import functools
import time
from typing import Callable, TypeVar

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

T = TypeVar("T")


def timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert HH:MM:SS or HH:MM:SS.mmm to seconds.

    Args:
        timestamp: Timestamp string

    Returns:
        Time in seconds
    """
    parts = timestamp.split(":")
    if len(parts) != 3:
        return 0.0

    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])

    return hours * 3600 + minutes * 60 + seconds


def seconds_to_timestamp(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS format.

    Args:
        seconds: Time in seconds

    Returns:
        Timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def seconds_to_timestamp_ms(seconds: float) -> str:
    """
    Convert seconds to HH:MM:SS.mmm format.

    Args:
        seconds: Time in seconds

    Returns:
        Timestamp string with milliseconds
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def seconds_to_pts(seconds: float) -> int:
    """
    Convert seconds to presentation timestamp (nanoseconds).

    Args:
        seconds: Time in seconds

    Returns:
        PTS in nanoseconds
    """
    return int(seconds * 1_000_000_000)


def pts_to_seconds(pts: int) -> float:
    """
    Convert presentation timestamp (nanoseconds) to seconds.

    Args:
        pts: PTS in nanoseconds

    Returns:
        Time in seconds
    """
    return pts / 1_000_000_000


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def retry_with_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
) -> Callable:
    """
    Decorator for retry with exponential backoff.

    Args:
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay in seconds
        exponential_base: Base for exponential backoff
        retryable_exceptions: Tuple of exceptions to retry

    Returns:
        Decorated function
    """
    return retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(
            multiplier=initial_delay,
            max=max_delay,
            exp_base=exponential_base,
        ),
        retry=retry_if_exception_type(retryable_exceptions),
    )


def timed(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to measure function execution time.

    Args:
        func: Function to decorate

    Returns:
        Decorated function that logs execution time
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.3f}s")
        return result

    return wrapper


async def timed_async(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to measure async function execution time.

    Args:
        func: Async function to decorate

    Returns:
        Decorated function that logs execution time
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        start = time.perf_counter()
        result = await func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.3f}s")
        return result

    return wrapper


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Rough estimate: ~4 characters per token.

    Args:
        text: Text to estimate

    Returns:
        Estimated token count
    """
    return len(text) // 4


def chunk_list(lst: list, chunk_size: int) -> list[list]:
    """
    Split a list into chunks.

    Args:
        lst: List to split
        chunk_size: Size of each chunk

    Returns:
        List of chunks
    """
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]
