"""
logger.py
---------
Structured logging and LLM cost tracking for the AEP Customer Intelligence Platform.

Features:
  - JSON-structured log output (suitable for CloudWatch Logs Insights queries)
  - Automatic request-id injection into log records
  - LLM cost accumulator with per-session and total-run summaries
  - Pipeline step timing decorator
  - Sensitive field masking (API keys, PII patterns)

Usage:
  from observability.logger import get_logger, CostTracker, timed_step

  log = get_logger(__name__)
  log.info("Pipeline started", extra={"step": "data_quality", "records": 3000})

  with CostTracker() as tracker:
      tracker.record_llm_call("gpt-4o-mini", prompt_tokens=120, completion_tokens=80)
  print(tracker.summary())
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable

from dotenv import load_dotenv

load_dotenv()

LOG_LEVEL  = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FORMAT = os.getenv("LOG_FORMAT", "json")   # "json" | "text"
SERVICE    = os.getenv("SERVICE_NAME", "aep-customer-intelligence-platform")

# ── Sensitive field masking ────────────────────────────────────────────────────
# Patterns for values that should never appear in logs
_MASK_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"sk-[A-Za-z0-9_-]{20,}"),            "[OPENAI_KEY_REDACTED]"),
    (re.compile(r"AKIA[A-Z0-9]{16}"),                 "[AWS_KEY_REDACTED]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),            "[SSN_REDACTED]"),
    (re.compile(r"\b[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}\b"), "[EMAIL_REDACTED]"),
]


def _mask_sensitive(text: str) -> str:
    for pattern, replacement in _MASK_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


# ── JSON Log Formatter ────────────────────────────────────────────────────────

class JsonFormatter(logging.Formatter):
    """
    Emit each log record as a single-line JSON object.
    Compatible with AWS CloudWatch Logs Insights.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp":  self.formatTime(record, self.datefmt),
            "level":      record.levelname,
            "logger":     record.name,
            "message":    _mask_sensitive(record.getMessage()),
            "service":    SERVICE,
        }

        # Attach extra fields passed via `extra=` kwarg
        for key, val in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and not key.startswith("_"):
                if key not in log_obj:
                    log_obj[key] = val

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, default=str)


# ── Text Formatter (dev-friendly) ─────────────────────────────────────────────

class TextFormatter(logging.Formatter):
    _FMT = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    _DATE = "%Y-%m-%d %H:%M:%S"

    def __init__(self):
        super().__init__(fmt=self._FMT, datefmt=self._DATE)

    def format(self, record: logging.LogRecord) -> str:
        record.msg = _mask_sensitive(str(record.msg))
        return super().format(record)


# ── Logger Factory ────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger.  Calling this multiple times with the same
    name is safe — handlers are only added once.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured

    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    handler = logging.StreamHandler()
    if LOG_FORMAT == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(TextFormatter())

    logger.addHandler(handler)
    logger.propagate = False
    return logger


# ── LLM Cost Tracker ──────────────────────────────────────────────────────────

# Pricing per 1M tokens (USD) — update as model prices change
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o":         {"input": 5.00,  "output": 15.00},
    "gpt-4o-mini":    {"input": 0.15,  "output": 0.60},
    "gpt-4-turbo":    {"input": 10.00, "output": 30.00},
    "gpt-3.5-turbo":  {"input": 0.50,  "output": 1.50},
}
_DEFAULT_PRICING = {"input": 0.50, "output": 1.50}  # fallback


@dataclass
class LLMCallRecord:
    model:             str
    prompt_tokens:     int
    completion_tokens: int
    cost_usd:          float
    task:              str = "unknown"
    elapsed_seconds:   float = 0.0


@dataclass
class CostTracker:
    """
    Accumulates token usage and cost across multiple LLM calls.

    Usage (as context manager):
        with CostTracker() as tracker:
            tracker.record_llm_call("gpt-4o-mini", 150, 80, task="summarization")
        print(tracker.summary())

    Usage (standalone):
        tracker = CostTracker()
        tracker.record_llm_call(...)
        print(tracker.summary())
    """
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    _calls: list[LLMCallRecord] = field(default_factory=list, init=False, repr=False)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(c.prompt_tokens for c in self._calls)

    @property
    def total_completion_tokens(self) -> int:
        return sum(c.completion_tokens for c in self._calls)

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def total_cost_usd(self) -> float:
        return sum(c.cost_usd for c in self._calls)

    @property
    def call_count(self) -> int:
        return len(self._calls)

    def record_llm_call(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        task: str = "unknown",
        elapsed_seconds: float = 0.0,
    ) -> float:
        """
        Record one LLM API call. Returns the cost in USD for this call.
        """
        pricing = _MODEL_PRICING.get(model.lower(), _DEFAULT_PRICING)
        cost = (
            prompt_tokens     / 1_000_000 * pricing["input"]
            + completion_tokens / 1_000_000 * pricing["output"]
        )
        self._calls.append(LLMCallRecord(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            task=task,
            elapsed_seconds=elapsed_seconds,
        ))
        return cost

    def summary(self) -> dict[str, Any]:
        """Return a summary dict suitable for logging or display."""
        by_model: dict[str, dict] = {}
        for call in self._calls:
            entry = by_model.setdefault(call.model, {
                "calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0
            })
            entry["calls"]             += 1
            entry["prompt_tokens"]     += call.prompt_tokens
            entry["completion_tokens"] += call.completion_tokens
            entry["cost_usd"]          += call.cost_usd

        return {
            "session_id":          self.session_id,
            "total_calls":         self.call_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens":        self.total_tokens,
            "total_cost_usd":      round(self.total_cost_usd, 6),
            "by_model":            {
                m: {**v, "cost_usd": round(v["cost_usd"], 6)}
                for m, v in by_model.items()
            },
        }

    def log_summary(self, logger: logging.Logger | None = None) -> None:
        """Log the cost summary at INFO level."""
        lg = logger or get_logger(__name__)
        lg.info("LLM Cost Summary", extra=self.summary())

    # Context manager support
    def __enter__(self) -> "CostTracker":
        return self

    def __exit__(self, *_) -> None:
        self.log_summary()


# ── Timing Decorator ──────────────────────────────────────────────────────────

def timed_step(step_name: str | None = None):
    """
    Decorator that logs the execution time of a pipeline step.

    @timed_step("data_quality")
    def run_checks(): ...
    """
    def decorator(func: Callable) -> Callable:
        name = step_name or func.__name__

        @wraps(func)
        def wrapper(*args, **kwargs):
            log = get_logger(func.__module__)
            start = time.monotonic()
            log.info("Step started", extra={"step": name})
            try:
                result = func(*args, **kwargs)
                elapsed = time.monotonic() - start
                log.info("Step completed", extra={"step": name, "elapsed_seconds": round(elapsed, 3)})
                return result
            except Exception as exc:
                elapsed = time.monotonic() - start
                log.error(
                    "Step failed",
                    extra={"step": name, "elapsed_seconds": round(elapsed, 3), "error": str(exc)},
                )
                raise

        return wrapper
    return decorator


# ── Context Manager for Timed Blocks ─────────────────────────────────────────

@contextmanager
def timed_block(name: str, logger: logging.Logger | None = None):
    """
    Context manager for timing an arbitrary block of code.

    with timed_block("glue_transform"):
        run_spark_job()
    """
    lg = logger or get_logger(__name__)
    start = time.monotonic()
    lg.info("Block started", extra={"block": name})
    try:
        yield
        elapsed = time.monotonic() - start
        lg.info("Block completed", extra={"block": name, "elapsed_seconds": round(elapsed, 3)})
    except Exception as exc:
        elapsed = time.monotonic() - start
        lg.error(
            "Block failed",
            extra={"block": name, "elapsed_seconds": round(elapsed, 3), "error": str(exc)},
        )
        raise
