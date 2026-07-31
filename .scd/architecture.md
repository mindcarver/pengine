---
managed_by: scd-architecture
status: ready
sources:
  - https://github.com/mindcarver/pengine/issues/1
  - https://github.com/mindcarver/pengine/issues/10
  - /Users/carver/Downloads/Telegram Desktop/短剧线_九文件与调用逻辑说明_v1_2.docx
contracts:
  - contracts/openapi.json
  - contracts/persona-package.schema.json
---

# System purpose and boundaries

Pengine V1 is a local, single-operator backend that turns one selected persona
package plus free-form story and script requirements into a complete short-drama
delivery. It permits one frozen revision request and reruns the full workflow
against the same immutable persona snapshot.

The system is a modular monolith. It binds only to loopback, serves one
same-origin single-page workbench plus a JSON HTTP API, persists durable state
in SQLite, and runs one embedded background worker. The worker drives a Deep
Agents/LangGraph creative runtime through one Anthropic Messages
API-compatible relay.

V1 does not include authentication, multi-user isolation, public deployment,
multiple model protocols, production persona authoring, automatic persona
learning, cross-creation writable memory, asynchronous or remote subagents,
host filesystem or shell access, task listing/export/deletion, streaming
transport, partial-result preview, or deployment automation.

## Domain model

### Persona package

A `PersonaPackage` is an operator-owned, versioned set of nine UTF-8 Markdown
content files plus one `manifest.json`. Its source files are read-only to the
application.

An accepted package is copied once into a content-addressed, immutable local
snapshot. A creation references the snapshot hash, not a mutable source path.
New package versions affect only new creations.

### Creation and workflow runs

A `Creation` owns:

- the original story and script requirements;
- the selected persona identifier, version, and immutable snapshot hash;
- one immutable initial `WorkflowRun`;
- zero or one frozen `RevisionFeedback`;
- zero or more revision-attempt `WorkflowRun` records, of which at most one may
  succeed;
- the durable initial and revised delivery packages.

A `WorkflowRun` is either `initial` or `revision`. Public run states are:

- `queued`
- `running`
- `auto_resuming`
- `paused`
- `ended`
- `succeeded`
- `failed`

Each `WorkflowRun` owns one immutable Deep Agents `thread_id`. An initial run
and every revision-attempt run use different thread identifiers. The thread
holds the supervisor's plan, synchronous subagent interactions, message state,
and virtual scratch files. It never substitutes for the domain run record.

The revision resource separately exposes:

- `unavailable` until the initial run succeeds;
- `available` before feedback is frozen;
- `queued`, `running`, `auto_resuming`, `paused`, `ended`, `failed`, or
  `succeeded` after feedback is frozen.

Internal stages are:

1. `loading_persona`
2. `selecting_l0_variant`
3. `generating_story_outline`
4. `generating_character_biographies`
5. `generating_relationship_logic`
6. `generating_episode_outline`
7. `generating_episode_scripts`
8. `accepting_l0`
9. `accepting_l4`
10. `assembling_delivery`

The workbench groups those internal stages into seven stable user stages:

1. determine the creative direction;
2. generate the story outline;
3. generate character biographies;
4. generate character relationships;
5. generate the episode outline;
6. generate episode scripts;
7. review the finished work.

The final user stage exposes L0 creative-core alignment and L4 craft/value
review as separate sub-statuses without collapsing their internal gates.

While generating episode scripts, a run resource may additionally expose the
durably recorded total, completed, and current episode numbers plus committed
per-episode drafts. The workbench presents those records as read-only
navigation, including after an ended or failed run; it does not infer or expose
text for an uncommitted episode.

The persona-bound `workflow_supervisor` advances these stages by delegating to
four named synchronous subagents:

- `story_architect` handles L0 selection, story outline, character biographies,
  and relationship logic through stage-specific structured tasks;
- `episode_planner` handles the episode outline;
- `script_writer` handles episode scripts;
- `quality_reviewer` produces structured L0/L4 evidence and revision-feedback
  coverage.

There are two distinct checkpoint meanings:

- LangGraph checkpoints preserve the in-progress Deep Agents thread after agent
  steps so the same run can resume after a process restart.
- Business checkpoints are approved stage outputs recorded by the creation
  service. Only these checkpoints may advance the domain stage or contribute to
  a delivery.

The supervisor and subagents may propose structured results and evidence. The
creation service validates them, enforces the three-attempt budget, and remains
the sole authority that approves a stage or marks a run succeeded.

### Delivery

`ContentPackage` contains exactly five creative outputs:

- story outline;
- character biographies;
- character-relationship logic;
- episode outline;
- episode scripts.

`DeliveryReport` is separate from the content package and contains:

- persona identifier, version, and snapshot hash;
- selected L0 variant and selection rationale;
- L0 gate result and evidence;
- L4 gate result and evidence;
- ownership statement;
- revision-feedback handling items for a revised delivery only.

No partial content package is exposed for a run that is queued, running,
auto-resuming, paused, ended, or failed. A succeeded initial package remains
readable while a revision is queued, running, auto-resuming, paused, ended, or
failed.

### Revision invariants

- A revision is allowed only after the initial run succeeds.
- The first accepted feedback payload is frozen.
- A queued or running revision rejects duplicate submissions.
- An ended revision is terminal and cannot be requeued.
- A failed revision may be requeued only with the exact frozen feedback.
- Requeuing a failed revision creates a new revision-attempt run and preserves
  the failed run for audit.
- A failed revision does not consume the revision entitlement.
- A succeeded revision permanently closes the revision command.
- A revision is a new full workflow run; it never mutates the initial delivery.
- Initial and revision runs use the same persona snapshot.

## Components and responsibilities

| Component | Owns | Does not own |
|---|---|---|
| Local workbench | Persona selection, creation/revision commands, 1.8-second status polling, one shared progress display, read-only per-episode draft navigation, paused-run controls | Stage inference, partial-result access, workflow execution |
| Local HTTP API | Runtime validation, idempotent commands, resource queries, paused-run continue/end commands, stable errors | Workflow execution, model retries, persona parsing |
| Creation service | Domain invariants and state-transition authority | HTTP serialization, vendor requests |
| Embedded worker | Durable job leases, stage-attempt guards, invocation/resume of Deep Agents threads, restart reconciliation | Creative planning, public request semantics |
| Deep Agents supervisor | Run-local planning, ordered specialist delegation, correction loops, structured candidate assembly | Domain state transitions, retry entitlement, final gate authority |
| Synchronous subagents | Stage-scoped creative work and structured review evidence | Job scheduling, direct database writes, persona mutation |
| Persona loader | Manifest/schema validation, immutable snapshots, read-only virtual context projection, bounded L5/L6 retrieval | Production persona content |
| LangGraph checkpointer | Thread messages, plans, subagent results, and `StateBackend` scratch checkpoints | Business checkpoints, public run state, revision entitlement |
| Relay client | Preconfigured `ChatAnthropic`, Anthropic-compatible request/response mapping, explicit timeouts, safe error mapping | Workflow retry policy, API keys at rest |
| SQLite repository | Authoritative creation, run, attempt, job, feedback, approved-stage, delivery, catalog, and LangGraph checkpoint tables | Persona source-file ownership |

The process uses one worker and processes one creation job at a time. Deep
Agents and LangGraph are embedded libraries, not a separately deployed service.
External brokers, distributed workers, LangSmith deployment, and caches are
outside V1.

## Runtime and data flow

### Persona discovery

1. The loader scans the configured local persona root.
2. It parses and validates `manifest.json`, fixed file names, hashes, UTF-8
   encoding, and required Markdown sections.
3. A valid package is catalogued by persona identifier and version.
4. `GET /personas` returns only currently valid, selectable packages.

The configured source root contains at most one current package directory per
`persona_id`. Replacing that directory with a higher operator-managed version
requires a service restart in V1. Older versions remain available only through
immutable snapshots already referenced by creations.

### Initial creation

1. `POST /creations` validates the request and required `Idempotency-Key`.
2. In one SQLite transaction, the service resolves the active persona version,
   creates or reuses its immutable content-addressed snapshot, persists the
   creation, queued initial run, and Deep Agents `thread_id`, then returns
   `202`.
3. The worker leases the job, compiles a read-only persona context for that
   immutable snapshot, and invokes the persona-bound `workflow_supervisor`.
4. The supervisor plans the run and delegates each logical stage to the named
   synchronous subagent responsible for that output.
5. Before a specialist invocation for a stage, the worker's guarded delegation
   boundary durably records the next stage-attempt number. A fourth invocation
   is rejected before any model request.
6. Subagents return schema-validated structured results. The creation service
   applies deterministic structural checks and commits an approved business
   checkpoint before the supervisor can rely on the output downstream.
7. `quality_reviewer` produces the structured evidence packs for the L0 and L4
   gates. The creation service evaluates the required gate contract and owns
   the pass/fail state transition.
8. The complete content package and delivery report commit atomically with the
   run's `succeeded` state.
9. Each guarded attempt records the current internal stage in SQLite. Resource
   queries map it and approved checkpoints to the seven user stages, completed
   stages, elapsed active time, final-review sub-status, and available actions.

### Revision

1. `POST /creations/{creation_id}/revision` verifies that the initial run
   succeeded and the revision state permits a command.
2. The first accepted request freezes its feedback and queues a full revision
   run with a new Deep Agents `thread_id` against the initial persona snapshot.
3. A failed revision can be requeued only when the submitted feedback hash
   matches the frozen payload; the retry is a new revision-attempt run with
   another new thread. A failed thread is never continued as a replacement
   feedback conversation.
4. A successful revision commits a new complete delivery and permanently closes
   revision.
5. Revision progress and timeout control use the same persisted contract and
   workbench component as the initial run; the successful initial delivery
   remains visible throughout.

### Restart recovery

- Jobs use a SQLite lease with an expiry.
- Startup moves expired leased jobs back to the queue.
- The worker resumes the run's existing Deep Agents `thread_id` through the
  SQLite-backed LangGraph checkpointer.
- A stage attempt is counted before its guarded specialist invocation. If a
  process stops during an external model call, that recorded attempt remains
  consumed; recovery cannot silently create an uncounted fourth call.
- SQLite business state is authoritative when it differs from agent thread
  state. Already approved stage outputs are re-injected through the run context
  instead of being regenerated.
- Missing or unreadable thread checkpoints produce a stable safe failure; they
  never cause the worker to treat unverified agent state as approved output.
- Exhausting three attempts for a stage fails that run with a stable stage and
  error identifier.
- The first worker wall-clock timeout in a user stage parks the active clock,
  requeues the same run and `thread_id`, and automatically resumes from approved
  business checkpoints. The unapproved in-flight stage is invoked again.
- A second wall-clock timeout in the same user stage parks the job in `paused`.
  Idempotent operator commands may requeue that same run or mark it `ended`.
  Graph recursion exhaustion, invalid structured output, quality-gate
  rejection, and relay/provider failures remain terminal and are never routed
  through timeout recovery.
- A paused or ended job is excluded from lease-expiry reconciliation. Refresh
  reconstructs its stage, completed checkpoints, frozen elapsed time, and
  available actions from SQLite.

## Persona-package loading contract

`manifest.json` is governed by `contracts/persona-package.schema.json`. The
manifest is metadata and is not one of the nine content files.

`package_sha256` is the SHA-256 of the canonical ordered concatenation of the
nine lowercase per-file SHA-256 values in this order:
`paradigm`, `project`, `l0`, `l1`, `l2`, `l3`, `l4`, `l5`, `l6`. The manifest
itself is excluded, avoiding a circular hash.

API `snapshot_sha256` is the domain-separated SHA-256 of `package_sha256` plus
the canonical complete manifest. It therefore addresses the immutable package
identity as well as its Markdown content: two personas or versions may reuse
identical Markdown without sharing or replacing a snapshot.

| Logical file | Fixed name | Minimum required structure | Runtime use | V1 write policy |
|---|---|---|---|---|
| Paradigm | `paradigm.md` | L0-L6 definitions; arbitration; L0 structure; translation; feedback, blank-space, gates, discipline, ownership | Rule provenance and conflict resolution | Read-only |
| Project instruction | `project.md` | Identity; L0 full text; L1-L6 summaries and statuses; four iron rules; arbitration; workflow; gates/feedback summary | Full boot context for every run | Read-only |
| L0 | `l0.md` | Variants; red lines; temperature; item ownership/status markers | Included through project context; gate source | Read-only |
| L1 | `l1.md` | Source profile plus summary | Summary through project context | Read-only |
| L2 | `l2.md` | Source profile plus summary | Summary through project context | Read-only |
| L3 | `l3.md` | Methods, cognition, and shortcomings plus summary | Summary through project context | Read-only |
| L4 | `l4.md` | L4-A values; L4-B five-stage craft rules and numeric parameters | Stage-scoped constraints and validation | Read-only |
| L5 | `l5.md` | Works and experience entries | Bounded on-demand style references | Read-only |
| L6 | `l6.md` | External craft entries | Bounded on-demand craft references | Read-only |

Each L0 status marker must distinguish creator-confirmed content from an
AI-structured pending item. Pending items cannot be compiled as confirmed
persona rules. Exact heading and marker validation belongs to the persona
loader; accepting arbitrary rich documents is outside V1.

L5 and L6 are split into a restart-scoped derived local chunk index. The
Markdown snapshot remains authoritative. Only ranked excerpts bounded by the
configured result count and character budgets enter the relay context; neither
file is projected wholesale into Agent state.

For each run, the loader seeds only the approved compiled context into the
Deep Agents `StateBackend` virtual tree:

- `/persona/` is read-only and contains the full project instruction, required
  L0 material, L1-L3 summaries, and stage-scoped L4 material;
- `/workspace/` is thread-scoped scratch shared by the supervisor and its
  synchronous subagents;
- L5/L6 content is available only through a bounded read-only retrieval tool,
  not as complete virtual files.

Permission rules allow the minimum required virtual reads and scratch writes,
then deny unmatched access. No Deep Agents backend resolves these paths to the
operator's host filesystem.

## Data ownership and lifecycle

SQLite is authoritative for:

- persona catalog metadata and snapshot references;
- creations and idempotency records;
- initial and revision workflow runs;
- stage attempts and approved checkpoints;
- durable jobs and leases;
- frozen revision feedback;
- content packages and delivery reports.

The same local SQLite database also contains LangGraph-owned checkpointer
tables managed by `AsyncSqliteSaver` from `langgraph-checkpoint-sqlite`. Those
tables are authoritative only for agent-thread execution state. They are not
queried to decide public creation status, revision eligibility, stage attempt
counts, or delivery validity.

Deep Agents uses `StateBackend`, so its virtual files persist only inside the
checkpointed thread. V1 does not configure `StoreBackend`, `CompositeBackend`
memory routes, or any other cross-thread writable memory. A later version may
add an explicitly namespaced `/memories/` route after memory sources, write
approval, curation, invalidation, and rollback are defined. Such memory must
remain separate from the immutable nine-file persona authority.

The configured persona root is operator-owned source data. The application
owns immutable, content-addressed snapshots under its local data directory.
Snapshots are shared by hash across creations and are never modified.

Creation data is retained locally without automatic expiry. V1 has no list,
export, or per-task deletion API. The operator may stop the service and back up
or remove the complete local data directory. Selective deletion and snapshot
garbage collection are deferred.

## Shared interface contracts

`contracts/openapi.json` is the canonical HTTP contract.
`contracts/persona-package.schema.json` is the canonical manifest contract.

The HTTP contract owns operation identifiers, request/response schemas,
idempotency, public states, and stable errors. This document owns rationale,
responsibility, persistence, recovery, and trust boundaries; it does not
duplicate the full field contract.

## Security and trust boundaries

- The API binds to `127.0.0.1` by default and has no authentication in V1.
- The relay receives the user's story, requirements, selected persona content,
  stage outputs needed for continuation, and validation prompts.
- `base_url`, `api_key`, and `model_id` are deployment configuration.
- One process-level `ChatAnthropic` instance is constructed from those values
  and supplied to the supervisor and all four subagents. The relay must support
  Anthropic Messages tool use. Supervisor and subagent schemas use LangChain
  `ToolStrategy` rather than provider-native structured-output extensions.
- The API key is read from the environment, never returned by the API, stored in
  SQLite, written to logs, or committed.
- Application logs contain identifiers, states, durations, attempt counts, safe
  error identifiers, and model identifiers only. They exclude creative content,
  complete prompts, raw provider responses, and secrets.
- Persona paths are resolved below the configured persona root; absolute paths
  and traversal outside that root are rejected.
- The default Deep Agents general-purpose subagent is disabled. Only the four
  named synchronous subagents are registered.
- `FilesystemBackend`, `LocalShellBackend`, sandbox `execute`, asynchronous or
  remote subagents, arbitrary MCP tools, agent-authored skills, and
  cross-creation writable memory are disabled in V1.
- Agent code has no tool that writes the persona source root, snapshots, or
  business tables. Structured stage output crosses back through the creation
  service for validation and persistence.
- Relay and parsing failures map to stable safe errors; vendor bodies, database
  messages, and stack traces do not cross the API boundary.

## Reliability and operations

- `ChatAnthropic.max_retries` is zero and no Deep Agents retry middleware is
  installed. The worker is the sole owner of the three-attempt budget.
- A finite LangGraph recursion limit and a worker-enforced wall-clock deadline
  bound each run attempt. Graph recursion exhaustion fails with its own stable
  error. A wall-clock timeout follows the one-auto-resume/then-pause policy and
  never starts an unrecorded model call.
- Provider-specific prompt caching and beta-only Anthropic features remain
  disabled until the configured relay smoke test proves compatibility.
- Each POST command requires an `Idempotency-Key`.
- The same key and request hash replays the original command response; its
  `resource_url` resolves the resource's current state.
- Reusing a key with a different request hash returns
  `idempotency_conflict`.
- SQLite uses transactions, foreign keys, WAL mode, and a busy timeout.
- LangGraph checkpointer tables and domain tables share the same database file
  but have separate schema ownership. Checkpointer writes never count as
  approved business checkpoints.
- A single worker avoids concurrent mutation of one creation and keeps V1's
  queue semantics deterministic.
- There is no ordinary-running cancellation, priority, SSE/WebSocket streaming,
  token streaming, partial result, percentage, ETA, or budget display in V1.
  Manual continue/end is available only after the second timeout pauses a run.

## Compatibility, migration, and rollback

This is a greenfield V1. The HTTP contract, manifest schema, and SQLite schema
each carry an explicit version.

The implementation must lock exact tested versions of `deepagents`,
`langchain-anthropic`, `langgraph`, and `langgraph-checkpoint-sqlite`. Upgrading
any of them is a deliberate compatibility change requiring the supervisor,
subagent structured-output, permissions, checkpoint-resume, and relay contract
tests to pass again. V1 does not track prerelease dependency versions.

Compatible additive HTTP changes may stay within V1. The progress fields and
run-control commands are additive. Breaking HTTP or persona format changes
require a new contract version. Persona source packages remain outside the
database so application rollback does not rewrite operator content.

SQLite schema version 2 adds `run_progress` and backfills existing runs without
rewriting creations, checkpoints, deliveries, or frozen revisions. Migrations
must remain forward and transactional where SQLite permits; production rollback
automation is outside this architecture-delivery slice.

Technology feasibility is grounded in the current official contracts:

- Deep Agents synchronous subagents accept individual structured response
  formats and return those results to the supervisor:
  <https://docs.langchain.com/oss/python/deepagents/subagents>
- `StateBackend` stores virtual files in checkpointed thread state, while
  `StoreBackend` is the separate cross-thread persistence mechanism:
  <https://docs.langchain.com/oss/python/deepagents/backends>
- `langgraph-checkpoint-sqlite` supplies `SqliteSaver` and
  `AsyncSqliteSaver` for local workflows:
  <https://docs.langchain.com/oss/python/langgraph/persistence>
- `ChatAnthropic` supports a preconfigured model client and structured output:
  <https://docs.langchain.com/oss/python/integrations/chat/anthropic>

## Verification boundaries

Implementation evidence must include:

- format-aware parsing/linting of the OpenAPI contract;
- JSON Schema validation of the representative persona manifest;
- unit tests for every domain invariant and stable error;
- state-machine tests for initial, revision, failed-revision retry, duplicate
  commands, timeout auto-resume, pause/continue/end, and service restart;
- isolated SQLite tests for transactions, leases, checkpoints, and attempt
  exhaustion;
- Deep Agents integration tests proving the persona-bound supervisor invokes
  only the four named synchronous subagents, returns structured results, and
  cannot access host files, shell, MCP, or cross-thread memory;
- checkpoint tests proving a stopped run resumes the same `thread_id`, approved
  business stages are not regenerated, and a new revision-attempt run uses a
  new thread;
- fake-relay tests proving `ChatAnthropic` request mapping, tool-use capability,
  SDK and middleware retries disabled, structured output validation, bounded
  agent execution, and safe failures;
- a configured relay smoke test that separately reports external-service
  evidence;
- at least two real operator-supplied persona packages before claiming
  persona-selection and persona-effect UAT.

Engineering tests with non-production fixtures cannot prove real persona
quality.

## Decisions and open items

### Decisions

- Modular monolith, FastAPI/Pydantic HTTP layer, direct SQLite repository, one
  embedded worker, and an embedded Deep Agents/LangGraph creative runtime.
- One persona-bound `workflow_supervisor` plus four synchronous specialist
  subagents: `story_architect`, `episode_planner`, `script_writer`, and
  `quality_reviewer`.
- A preconfigured LangChain `ChatAnthropic` model uses the operator-supplied
  `base_url`, `api_key`, and `model_id`, with automatic retries disabled.
- `AsyncSqliteSaver` provides durable per-run thread checkpoints in the same
  local SQLite database. `StateBackend` provides checkpointed thread scratch.
- V1 memory is thread-scoped only. Cross-creation writable `StoreBackend` memory
  is deferred until its governance contract is defined.
- SQLite domain records own business truth; LangGraph checkpoints own
  in-progress agent execution state only.
- Immutable content-addressed persona snapshots.
- Public initial and revision run states remain separate.
- Initial and revision run resources share the same progress schema and
  workbench progress component.
- Checkpoint recovery resumes the existing Deep Agents thread while approved
  business checkpoints remain authoritative.
- Wall-clock timeout recovery is distinct from recursion, structured-output,
  quality-gate, and relay/provider failure handling.
- `ContentPackage` and `DeliveryReport` are separate contract objects.

### Open items

- No architecture decision remains open.
- The governing delivery source is
  <https://github.com/mindcarver/pengine/issues/1>.
- Four bundled nine-file persona packages are explicitly non-production
  prototypes; creator-confirmed production persona quality remains unverified.
- The configured relay has exercised real tool-use and structured-output paths,
  but a complete workbench initial-plus-revision acceptance run for Issue #10
  remains required.
