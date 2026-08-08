"""Best-effort Langfuse business events for the workflow control plane."""

from __future__ import annotations

import hashlib
import os
from typing import Any


def record_langfuse_event(
    name: str,
    *,
    input: dict[str, Any],
    metadata: dict[str, Any] | None = None,
) -> None:
    """Record a small, content-safe event without affecting workflow execution."""
    if not (
        os.getenv("LANGFUSE_PUBLIC_KEY")
        and os.getenv("LANGFUSE_SECRET_KEY")
        and os.getenv("LANGFUSE_BASE_URL")
    ):
        return

    try:
        from langfuse import get_client

        get_client().create_event(name=name, input=input, metadata=metadata or {})
    except Exception:
        # Observability is diagnostic only; SQLite checkpoints remain authoritative.
        return


def record_model_call_event(
    *,
    phase: str,
    role: str,
    stage: str | None,
    episode_number: int | None,
    sequence: int | None,
    outcome: str,
    call_id: str,
    model: str,
    repair_round: int | None = None,
    error_code: str | None = None,
    finish_reason: str | None = None,
    duration_seconds: float | None = None,
) -> None:
    """Record queryable, content-safe lineage for one outbound model call.

    The local Langfuse deployment currently runs in ``events_only`` mode, where
    list responses expose event names but not event metadata. Keep the bounded
    dimensions in the name so a run can still be reconstructed from the list API.
    Prompts and model responses are intentionally excluded.
    """
    stage_label = _event_label(stage or "unknown")
    episode_label = f"episode-{episode_number}" if episode_number is not None else "no-episode"
    sequence_label = f"seq-{sequence}" if sequence is not None else "seq-none"
    repair_label = f"repair-{repair_round}" if repair_round is not None else "repair-none"
    name = ".".join(
        (
            "pengine",
            "model_call",
            "v1",
            _event_label(phase),
            _event_label(role),
            stage_label,
            episode_label,
            _event_label(outcome),
            sequence_label,
            repair_label,
        )
    )
    record_langfuse_event(
        name,
        input={
            "call_id": call_id,
            "model": model,
            "stage": stage,
            "episode_number": episode_number,
            "role": role,
            "sequence": sequence,
            "outcome": outcome,
            "repair_round": repair_round,
            "error_code": error_code,
            "finish_reason": finish_reason,
            "duration_seconds": duration_seconds,
        },
        metadata={"trace_version": "pengine-1", "lineage_version": "model-call-v1"},
    )


def _event_label(value: str) -> str:
    """Keep dimensions safe and stable when embedded in an event name."""
    normalized = "".join(char if char.isalnum() or char in "-_" else "_" for char in value)
    return normalized or "unknown"


def content_fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
