"""Limitació de peticions (token bucket compartit entre fils) i reintents amb backoff."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import TypeVar

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from finance.providers.base import TransientProviderError

log = logging.getLogger(__name__)
T = TypeVar("T")


class RateLimiter:
    """Token bucket: `rate` peticions/segon de mitjana amb ràfegues fins a `burst`."""

    def __init__(self, rate: float, burst: int | None = None):
        if rate <= 0:
            raise ValueError("rate ha de ser > 0")
        self.rate = rate
        self.capacity = float(burst if burst is not None else max(1, int(rate)))
        self._tokens = self.capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, n: float = 1.0) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
                self._last = now
                if self._tokens >= n:
                    self._tokens -= n
                    return
                wait = (n - self._tokens) / self.rate
            time.sleep(wait)


def with_retries(
    fn: Callable[..., T],
    *,
    attempts: int = 5,
    max_wait: float = 60.0,
) -> Callable[..., T]:
    """Reintenta només errors transitoris, amb backoff exponencial aleatori."""
    return retry(
        retry=retry_if_exception_type(TransientProviderError),
        stop=stop_after_attempt(attempts),
        wait=wait_random_exponential(multiplier=2, max=max_wait),
        before_sleep=before_sleep_log(log, logging.WARNING),
        reraise=True,
    )(fn)
