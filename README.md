# Pengine

Pengine V1 is a loopback-only backend for one internal operator. It uses
[Deep Agents](https://github.com/langchain-ai/deepagents) to turn a selected
writer persona, a story, and script requirements into one complete short-drama
delivery, then accepts one frozen revision request and returns a final complete
delivery.

There is no frontend, authentication, public deployment, or cross-creation
memory in V1. The approved product contract is
[Issue #1](https://github.com/mindcarver/pengine/issues/1); the machine contracts
are [`contracts/openapi.json`](contracts/openapi.json) and
[`contracts/persona-package.schema.json`](contracts/persona-package.schema.json).

## Requirements

- Python 3.12
- [`uv`](https://docs.astral.sh/uv/)
- An Anthropic Messages-compatible relay with a base URL, API key, and model ID
- At least one operator-supplied persona package

Install the locked dependencies:

```bash
uv sync --locked --all-groups
```

## Configure

Copy `.env.example` to `.env` and set the three relay fields:

```dotenv
PENGINE_PERSONA_ROOT=./personas
PENGINE_DATA_DIR=./data
PENGINE_HOST=127.0.0.1
PENGINE_PORT=8000
PENGINE_RELAY_BASE_URL=https://your-relay.example
PENGINE_RELAY_API_KEY=replace-with-your-key
PENGINE_RELAY_MODEL_ID=your-model-id
```

Only loopback hosts are accepted. The relay key is read at runtime and must not
be committed.

Each direct child of `PENGINE_PERSONA_ROOT` is one persona package containing
exactly:

```text
manifest.json
paradigm.md
project.md
l0.md
l1.md
l2.md
l3.md
l4.md
l5.md
l6.md
```

The manifest shape, fixed paths, hashes, and Markdown requirements are defined
by the persona-package contract. The example manifest is intentionally
non-production and is only suitable for schema validation. Pengine does not
bundle a production persona. `package_sha256` identifies the nine Markdown
contents; the API's `snapshot_sha256` also includes canonical manifest identity
metadata so different personas or versions cannot share a snapshot by mistake.

Restart the service after installing or replacing an active persona package.
Existing creations continue to use their immutable persona snapshot.

## Run

```bash
uv run pengine
```

The API listens on `http://127.0.0.1:8000` by default. OpenAPI documentation is
available locally at `http://127.0.0.1:8000/docs`.

## Three interactions

First, list the valid selectable personas:

```bash
curl --fail-with-body http://127.0.0.1:8000/personas
```

Second, queue a complete creation. Use a new caller-generated idempotency key:

```bash
curl --fail-with-body \
  -X POST http://127.0.0.1:8000/creations \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: creation-001' \
  -d '{
    "persona_id": "your-persona-id",
    "story": "一个离开故乡多年的人回乡处理旧屋。",
    "requirements": "创作一部完整短剧，具体标准遵循所选人格。"
  }'
```

Poll the returned `resource_url` until `initial.state` is `succeeded` or
`failed`:

```bash
curl --fail-with-body \
  http://127.0.0.1:8000/creations/REPLACE_WITH_CREATION_ID
```

Third, after initial success, submit the only revision. The first accepted
feedback is frozen:

```bash
curl --fail-with-body \
  -X POST \
  http://127.0.0.1:8000/creations/REPLACE_WITH_CREATION_ID/revision \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: revision-001' \
  -d '{
    "feedback": "主角最后的选择太顺利，请让代价更明确，同时保留原有情绪底色。"
  }'
```

Poll the same creation resource until `revision.state` is `succeeded` or
`failed`. A failed revision may be retried only with the exact frozen feedback.

## Local data and recovery

Pengine stores creation inputs, feedback, outputs, durable jobs, business
checkpoints, and LangGraph checkpoints in
`PENGINE_DATA_DIR/pengine.sqlite3`. Persona snapshots are stored under
`PENGINE_DATA_DIR/persona-snapshots`.

Data is retained until the operator backs up or removes it. Protect the data
directory and backups according to the sensitivity of the story content. If
the process stops, restart it with the same data directory; expired job leases
are requeued and the existing run thread and approved business checkpoints are
reused.

## Verify

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Automated tests use non-production persona fixtures and deterministic fake model
responses. They verify engineering behavior only. Real relay tool use and
structured output, plus persona-effect differences across two real persona
packages, remain separate operator UAT.
