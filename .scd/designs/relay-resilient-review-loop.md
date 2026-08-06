# Design: relay-resilient multi-round review loop (#76)

status: draft
issue: 76
authority: Initiative #53 / Delivery #58

## Problem

A relay interruption (HTTP 408 timeout, connection error) inside
`_generate_consistent_story_artifact` propagates out of the review/repair loop,
through the supervisor graph, into the worker's except-block. The worker's
`_recover_relay_interruption` sets the run to `auto_resuming` and returns,
discarding all loop progress (`repair_rounds`, `previous_review`,
`previous_content`, the current candidate). On resume, the supervisor
re-delegates the stage from scratch because the in-flight stage has no durable
checkpoint. This resets the loop and wastes all prior review/repair work.

With unstable relays (the observed gpt-5.5 408 timeouts), this makes every
multi-round story-artifact stage non-convergent: the loop never reaches its
`_MAX_*_REPAIR_ROUNDS` cap because a relay timeout restarts it first.

## Constraint

The fix must not change the loop's correctness contract (bounded repair rounds,
`ContentReviewRejectedError` at the cap, canon-review issue closure semantics).
It must only change how a relay interruption *inside* the loop is handled, so
the loop resumes from its last known state instead of restarting.

## Option A: call-level retry inside the loop (recommended)

Wrap each `_invoke_semantic_reviewer`, `_invoke_story_review_backstop`, and
`_invoke_story_artifact_repair` call in a bounded relay-retry so a transient
relay failure retries the *same call* (with the relay's retry delay) instead of
propagating out and restarting the whole stage.

### Why
- Smallest blast radius: the loop's state (`repair_rounds`, `previous_review`)
  is preserved because the exception never leaves the loop.
- The relay's own `retryable_relay_interruption` classification already
  distinguishes retryable (408, connection) from fatal errors — reuse it.
- No persistence change: the loop is in-memory state, and call-level retry
  keeps it alive across transient failures.
- Matches the episode-script path's resilience model (episodes already tolerate
  relay interruptions at the call level via the worker's episode-error recovery).

### Sketch
```python
async def _invoke_with_relay_retry(self, coro_factory, *, stage, max_retries=2):
    delay = None
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory()
        except Exception as exc:
            interruption = retryable_relay_interruption(exc)
            if interruption is None or attempt >= max_retries:
                raise
            delay = interruption.retry_delay_seconds
            await asyncio.sleep(delay)
```
Each loop-internal subagent call is wrapped:
`_invoke_semantic_reviewer(...)`, `_invoke_story_review_backstop(...)`,
`_invoke_story_artifact_repair(...)`.

### Risk
- Adds latency (retries sleep before retrying). Bounded by `max_retries`.
- If the relay is *persistently* down (not transient), the retries exhaust and
  the interruption propagates as before — same behavior as today, no regression.

## Option B: persist loop state to a durable checkpoint

Write the loop's intermediate state (`current_content`, `repair_rounds`,
`previous_review`) to a durable store (SQLite or LangGraph checkpoint) after
each round, so the worker's resume can reconstruct the loop mid-flight.

### Why not
- Much larger change: new persistence schema, resume logic, state
  serialization for `CanonReviewerResult` and the candidate.
- The LangGraph checkpoint already persists the supervisor graph state, but the
  loop lives *inside* a single graph node (`awrap_tool_call`), so the
  checkpoint captures the pre-loop state, not the loop's intermediate state.
- Disproportionate complexity for a transient-retry problem.

## Recommendation

Option A (call-level retry). It is the smallest coherent fix that preserves
loop progress across transient relay failures, reuses the existing relay
classification, and degrades gracefully to today's behavior when the relay is
persistently unavailable.

## Acceptance

1. A relay interruption during a review or repair call retries the same call
   (bounded), and the loop continues from its current `repair_rounds`.
2. After `max_retries` exhausted relay failures, the interruption propagates
   and the worker recovers as today (stage restart) — no regression.
3. `ContentReviewRejectedError` still fires at `_MAX_*_REPAIR_ROUNDS`.
4. Unit test: a relay interruption injected mid-loop does not reset
   `repair_rounds` (with relay-retry, the loop survives).
5. Live e2e: a multi-round story-artifact stage survives a transient relay
   408 without restarting from `repair_rounds=0`.
