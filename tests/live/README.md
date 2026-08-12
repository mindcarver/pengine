# Real-model black-box E2E

This test starts a real Pengine server on a random loopback port, copies no
shared database state, and calls both configured model routes from the current
environment or repository `.env`. Generation and creative repair are fixed to
`claude-opus-5` over Anthropic Messages. Review may be `deepseek-v4-flash`,
`gpt-5.5`, `gpt-5.6-terra`, or `claude-opus-5`; Pengine selects the matching
DeepSeek, OpenAI, or Anthropic client. Both routes use the same
`PENGINE_RELAY_BASE_URL` and `PENGINE_RELAY_API_KEY`. Missing either
route fails closed; there is no single-model or cross-role fallback. The test is
deliberately skipped unless explicitly enabled because it makes billable model
requests and can take a long time.

Required model settings are:

```dotenv
PENGINE_RELAY_BASE_URL=https://your-relay.example/v1
PENGINE_RELAY_API_KEY=replace-with-your-key
PENGINE_GENERATION_MODEL_ID=claude-opus-5
PENGINE_REVIEW_MODEL_ID=deepseek-v4-flash
```

The review value above is an example matching `.env.example`, not the only
accepted review model. The selected relay route must support the matching tool
protocol and report that exact model identity.

`PENGINE_RELAY_ADAPTER`, `PENGINE_RELAY_MODEL_ID`, and
`PENGINE_RELAY_MAX_OUTPUT_TOKENS` are obsolete and ignored. Optional caps, when
needed, are `PENGINE_GENERATION_MAX_OUTPUT_TOKENS` and
`PENGINE_REVIEW_MAX_OUTPUT_TOKENS`.

Run the complete initial-creation flow with one command:

```bash
PENGINE_RUN_LIVE_E2E=1 uv run pytest -m live_model tests/live/test_real_model_e2e.py -vv -s
```

Run the isolated L3 A/B/C behavior probe with:

```bash
PENGINE_RUN_L3_ABC=1 uv run pytest -m live_model tests/live/test_l3_persona_e2e.py -vv -s
```

The three groups keep story, L0, Soul, L4, models, and parameters fixed while comparing the
confirmed full L3, a similarly sized neutral L3, and the historical one-line summary. The probe
checks single-line causal convergence, functional branches, L0/Canon authority, source privacy,
character independence, and that L3 never becomes a review gate.

Run the isolated L4 A/B/C authority probe with:

```bash
PENGINE_RUN_L4_ABC=1 uv run pytest -m live_model tests/live/test_l4_persona_e2e.py -vv -s
```

It copies the compiled persona to a temporary directory, records the real review call in a
temporary SQLite database, and proves three distinct decisions: an explicit hard-rule conflict
blocks, deviation from confirmed advice does not block, and a locked user parameter overrides
the Pengine default without being attributed to the creator.

Run the isolated Project authority probe with:

```bash
PENGINE_RUN_PROJECT_ABC=1 uv run pytest -m live_model tests/live/test_project_persona_e2e.py -vv -s
```

It sends the complete compiled Project exactly once to the generation route, keeps the full
Project out of the review route, records both real calls in a temporary SQLite database, and
checks locked user authority, absence of Project-created story facts, runtime privacy, and that
Project resemblance never becomes an independent review gate.

Every run writes durable evidence under `.artifacts/live-e2e/<timestamp>-<id>/`
by default, outside the shared `data/` tree. The directory contains safe
configuration metadata, the server log, persona and creation responses, poll and
stage timelines, the final creation resource, and the isolated
`data/pengine.sqlite3`. The API key is never written; metadata records only
whether one was present. At teardown the harness scans every evidence file,
including SQLite and WAL files, for the configured key and generic `sk-` tokens;
any match deletes the entire affected run directory before failing the test.
Relay, persona, timeout, and polling preflight validation happens before a run
directory is created, so preflight failures do not leave permanent `starting`
evidence behind. A successful run also requires audited calls for both roles,
with every returned model identity matching that role's configured model.

Optional controls:

- `PENGINE_LIVE_E2E_ARTIFACT_ROOT`: alternate evidence root.
- `PENGINE_LIVE_E2E_PERSONA_ID`: select a discovered persona (the first sorted
  persona is used otherwise).
- `PENGINE_LIVE_E2E_STORY` and `PENGINE_LIVE_E2E_REQUIREMENTS`: override the
  default Chinese test brief.
- `PENGINE_LIVE_E2E_TIMEOUT_SECONDS`: end-to-end deadline, default `7200`.
  The isolated server's run deadline is set to the same value so the worker
  cannot pause earlier than the black-box harness.
- `PENGINE_LIVE_E2E_POLL_SECONDS`: polling interval, default `2`.

Success requires a terminal `initial.state == "succeeded"`, all five content
artifacts to contain non-blank Simplified Chinese text according to Pengine's
production language detector, passed L0 and L4 gates, and
`revision.state == "available"`. The isolated SQLite database must also contain
a passing independent consistency review (plus a bounded repair count) for the
story outline, character biographies, and relationship logic checkpoints. Any
ordinary failed, paused, ended, or quality-rejected run keeps its evidence and
fails the test; credential-bearing evidence is the deliberate exception and is
deleted as described above.

The harness accepts arbitrary user requirements, but it does not certify the
current single-call episode-outline planner for 60–100 episode production. A
long-series run must be evaluated separately for structured-output size,
call-budget headroom, recovery behavior, and whole-series consistency.
