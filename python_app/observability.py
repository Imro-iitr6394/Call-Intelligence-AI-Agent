"""Optional, privacy-aware observability integrations.

The application remains usable without either external observability service. When
enabled, Logfire receives application spans and LangSmith receives sanitized model
runs. Raw audio, transcripts, prompts, outputs, and secrets are never attached to
these traces by this module.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import logging
import os
from time import perf_counter
from typing import Any, Callable, Iterator, TypeVar

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is optional at import time
    load_dotenv = None


LOGGER = logging.getLogger("astra.observability")
T = TypeVar("T")
_LOGFIRE_MODULE: Any | None = None
_LOGFIRE_CONFIGURED = False


def _load_environment() -> None:
    if load_dotenv is not None:
        load_dotenv()


def _is_true(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class ObservabilitySettings:
    """Non-secret observability configuration used by the application."""

    enabled: bool
    environment: str
    logfire_configured: bool
    langsmith_configured: bool
    langsmith_project: str


def get_settings() -> ObservabilitySettings:
    """Read observability settings without returning secret values."""

    _load_environment()
    enabled = _is_true(os.getenv("OBSERVABILITY_ENABLED", "false"))
    return ObservabilitySettings(
        enabled=enabled,
        environment=os.getenv("OBSERVABILITY_ENV", "development"),
        logfire_configured=bool(os.getenv("LOGFIRE_TOKEN")),
        langsmith_configured=bool(os.getenv("LANGSMITH_API_KEY"))
        and _is_true(os.getenv("LANGSMITH_TRACING", "false")),
        langsmith_project=os.getenv("LANGSMITH_PROJECT", "astra-call-intelligence"),
    )


def configure_observability() -> ObservabilitySettings:
    """Configure Logfire once and return a safe status object.

    Configuration failures are intentionally non-fatal: observability must never
    prevent call ingestion, transcript normalization, or local persistence.
    """

    global _LOGFIRE_CONFIGURED, _LOGFIRE_MODULE

    settings = get_settings()
    if not settings.enabled or _LOGFIRE_CONFIGURED:
        return settings

    try:
        import logfire

        config: dict[str, Any] = {
            "service_name": "astra-call-intelligence",
            "environment": settings.environment,
            "send_to_logfire": "if-token-present",
        }
        if os.getenv("LOGFIRE_TOKEN"):
            config["token"] = os.getenv("LOGFIRE_TOKEN")
        logfire.configure(**config)
        _LOGFIRE_MODULE = logfire
        _LOGFIRE_CONFIGURED = True
    except Exception as error:  # pragma: no cover - depends on external SDK state
        LOGGER.warning("Logfire setup skipped: %s", type(error).__name__)

    return settings


def _safe_attributes(attributes: dict[str, Any] | None) -> dict[str, Any]:
    """Keep trace attributes structured while excluding sensitive content."""

    if not attributes:
        return {}

    blocked_exact = {
        "api_key",
        "audio",
        "content",
        "credential",
        "file_name",
        "input",
        "output",
        "password",
        "prompt",
        "response",
        "secret",
        "source_bytes",
        "text",
        "token",
        "transcript",
    }
    blocked_suffixes = (
        "_api_key",
        "_bytes",
        "_content",
        "_file_name",
        "_password",
        "_prompt",
        "_response",
        "_text",
        "_token",
    )
    safe: dict[str, Any] = {}
    for key, value in attributes.items():
        normalized_key = str(key).lower()
        if normalized_key in blocked_exact or normalized_key.endswith(blocked_suffixes):
            continue
        if value is None or isinstance(value, (bool, int, float, str)):
            safe[str(key)] = value
        elif isinstance(value, (list, tuple, set)):
            safe[f"{key}_count"] = len(value)
        elif isinstance(value, dict):
            safe[f"{key}_count"] = len(value)
    return safe


@contextmanager
def observe(name: str, **attributes: Any) -> Iterator[None]:
    """Create a best-effort application span with safe metadata."""

    settings = configure_observability()
    safe = _safe_attributes(attributes)
    start = perf_counter()
    span_context = nullcontext()

    if settings.enabled and _LOGFIRE_MODULE is not None:
        try:
            span_context = _LOGFIRE_MODULE.span(name, **safe)
        except Exception as error:  # pragma: no cover - depends on external SDK state
            LOGGER.warning("Logfire span skipped: %s", type(error).__name__)

    try:
        with span_context:
            yield
    except Exception as error:
        LOGGER.error(
            "operation_failed name=%s error_type=%s duration_ms=%d",
            name,
            type(error).__name__,
            round((perf_counter() - start) * 1000),
        )
        raise
    else:
        LOGGER.info(
            "operation_completed name=%s duration_ms=%d",
            name,
            round((perf_counter() - start) * 1000),
        )


def trace_model_call(
    name: str,
    operation: Callable[[], T],
    *,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> T:
    """Trace a future model call without exporting its input or output content."""

    settings = configure_observability()
    if not (settings.enabled and settings.langsmith_configured):
        return operation()

    try:
        from langsmith import traceable

        traced_operation = traceable(
            name=name,
            run_type="llm",
            metadata=_safe_attributes(metadata),
            tags=tags or [],
            process_inputs=lambda _inputs: {},
            process_outputs=lambda _outputs: {},
        )(operation)
        return traced_operation()
    except Exception as error:  # pragma: no cover - depends on external SDK state
        LOGGER.warning("LangSmith trace skipped: %s", type(error).__name__)
        return operation()


def status_text() -> str:
    """Return a UI-safe one-line provider status without exposing secrets."""

    settings = get_settings()
    if not settings.enabled:
        return "disabled"
    providers = []
    if settings.logfire_configured:
        providers.append("Logfire")
    if settings.langsmith_configured:
        providers.append("LangSmith")
    return ", ".join(providers) if providers else "local fallback"
