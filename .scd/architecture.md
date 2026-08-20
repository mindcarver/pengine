---
managed_by: scd-architecture
status: ready
sources:
  - https://github.com/mindcarver/pengine/issues/1
  - https://github.com/mindcarver/pengine/issues/10
  - https://github.com/mindcarver/pengine/issues/37
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
Agents/LangGraph creative runtime through two fixed role routes that share one
relay URL and API key: Anthropic Messages with `claude-opus-5` for generation
and creative repair, plus a configured review model using its matching
Anthropic, OpenAI, or DeepSeek client protocol.

V1 does not include authentication, multi-user isolation, public deployment,
operator-selectable model roles or providers, production persona authoring,
automatic persona learning, cross-creation writable memory, asynchronous or
remote subagents, host filesystem or shell access, task
listing/export/deletion, streaming transport, uncommitted-candidate preview, or
deployment automation. Committed per-episode drafts are available as read-only
progress evidence but are never formal delivery.

## Domain model

### Persona package

A `PersonaPackage` is an operator-owned, versioned set of eight UTF-8 Markdown
content files plus one `manifest.json` (persona schema v3: `paradigm`, `project`,
`l0`, `soul`, `l3`, `l4`, `l5`, `l6`). Historical v1 packages carried nine files
with separate `l1`/`l2`; they remain restorable only as immutable snapshots for
existing runs and are no longer selectable for new work. Source files are
read-only to the application.

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
- `quality_rejected`
- `succeeded`
- `failed`

Each `WorkflowRun` owns one immutable Deep Agents `thread_id`. An initial run
and every revision-attempt run use different thread identifiers. The thread
holds the supervisor's plan, synchronous subagent interactions, message state,
and virtual scratch files. It never substitutes for the domain run record.

The revision resource separately exposes:

- `unavailable` until the initial run succeeds;
- `available` before feedback is frozen;
- `queued`, `running`, `auto_resuming`, `paused`, `ended`, `failed`,
  `quality_rejected`, or `succeeded` after feedback is frozen.

Internal stages are:

1. `loading_persona`
2. `selecting_l0_variant`
3. `generating_story_outline`
4. `generating_character_relationships`
5. `generating_episode_outline`
6. `generating_episode_scripts`
7. `accepting_l0`
8. `accepting_l4`
9. `assembling_delivery`

The workbench groups those internal stages into six stable user stages:

1. determine the creative direction;
2. generate the story outline;
3. generate character biographies and relationships;
4. generate the episode outline;
5. generate episode scripts;
6. review the finished work.

The final user stage exposes L0 creative-core alignment and L4 craft/value
review as separate sub-statuses without collapsing their internal gates.

While generating episode scripts, a run resource may additionally expose the
durably recorded total, completed, and current episode numbers plus committed
per-episode drafts. The workbench presents those records as read-only
navigation, including after an ended or failed run; it does not infer or expose
text for an uncommitted episode.

The persona-bound `workflow_supervisor` advances these stages by delegating to
four primary synchronous subagents:

- `story_architect` handles L0 selection, story outline, character biographies,
  and relationship logic through stage-specific structured tasks;
- `episode_planner` handles the episode outline;
- `script_writer` handles episode scripts;
- `quality_reviewer` is retained only so already-persisted legacy L0/L4 gate
  runs can be read or resumed; current gates are evidence labels carried by
  bound reviews.

The `workflow_supervisor`, `story_architect`, `episode_planner`,
`script_writer`, direct story/outline patch generators, `episode_repair`, and
`story_repair` always use the generation route. `quality_reviewer`,
`canon_reviewer`, `episode_reviewer`, `series_reviewer`, and the
`repair_constraint_extractor`/`repair_constraint_validator` helpers always use
the review route. Roles cannot be swapped and do not fall back to one another.

Episode-outline planning is chunked. The run first generates a compact
whole-season `OutlineSeasonMap` that divides episodes into consecutive natural
groups of 1-4 episodes along action, reveal, temporal, relationship, suspense,
or stage boundaries. Each group then generates and persists canonical
per-episode Markdown prose plus a structured continuity sidecar (facts,
timeline, clues, and obligations); deterministic validation and an independent
review-route check precede each group's immutable content-hash checkpoint, with
at most two repair rounds per rejected group and resume from the first
uncommitted group. A sidecar schema failure retries only the sidecar and never
rewrites the persisted Markdown. Once every group is committed, the service
deterministically assembles the complete `StoryContract` and runs a whole-season
final review before locking. The contract is the sole machine-readable source
for cast membership, relationships, typed facts and units, temporal order,
character knowledge, clue lifecycle, and per-episode obligations. A failing
assembled contract may be repaired by the generation route's bounded outline
patch at most twice before pausing.

The schema accepts `episode_count >= 1`, but that is not production evidence for
arbitrarily long series. Outline planning, locking, and cross-batch validation
are now chunked, but full 60-100-episode production reliability still requires
real isolated runs to verify context growth, call budgets, whole-series
consistency, and total duration; Pengine must not claim it until then.

Scripts are written in design-bound natural groups. For each episode,
`script_writer` returns runtime-bordered plain script text plus a compact
sidecar (episode number, content hash, typed `EpisodeStateDelta`, and evidence
targets). Deterministic validation compares them with the locked contract and
preceding folded `SeriesState`. Under an active SeriesBible the per-episode
model review is skipped and semantic consistency is deferred to the milestone
structural reviews; `episode_reviewer` remains only for the legacy path without
an active SeriesBible. The skill-scoped `episode_repair` subagent may repair
only the current unlocked episode, at most twice. A successful commit atomically
persists the full script, delta, folded state, semantic evidence, repair count,
and content/state hashes; script text that succeeded while its sidecar failed
retries only the sidecar. Specialist skills are loaded only into their matching
review or repair subagents, never into the supervisor globally.

There are two distinct checkpoint meanings:

- LangGraph checkpoints preserve the in-progress Deep Agents thread after agent
  steps so the same run can resume after a process restart.
- Business checkpoints are approved stage outputs recorded by the creation
  service. Only these checkpoints may advance the domain stage or contribute to
  a delivery.

The supervisor and subagents may propose structured results and evidence. The
creation service validates them, enforces the three-attempt budget, and remains
the sole authority that approves a stage or marks a run succeeded.

### Design package (SeriesBible)

A unified run builds one atomic story-design package once the L0 variant and the
approved story-outline, biography, relationship, and episode-outline projections
are available. The package is one immutable `SeriesBible` candidate that bundles
the four Markdown projections with the machine-readable `StoryContract`; every
projection and hash belongs to exactly one candidate, and no cross-version
mixture is observable through the API, UI, checkpoints, or restart state.

When the episode-outline result declares a `story_contract`, the creation
service assembles a candidate and runs deterministic universal validation
(schema, references, uniqueness, ordering, explicit arithmetic, and projection
consistency) plus the genre-activated rules the candidate declares. `mystery`
activates reveal and clue obligations; a general-genre idea is never rejected
for missing mystery-only mechanics. The bound configured-review-route global design review is
the review-route `canon_reviewer` result for that exact candidate id and content
hash; another candidate's review cannot approve it.

A candidate becomes `active` only through a transactional/CAS promotion that
requires passing deterministic validation and a passing bound review. At most
one complete automatic rebuild per run lineage may follow a confirmed design
defect; a second requires explicit authorization. Late or superseded candidates
are retained as immutable `stale`/`superseded` evidence with their review and
usage and can never move the active pointer or trigger downstream work. A design
hash/epoch change supersedes the whole prior script batch and signals the writer
Issue to start a fresh batch at episode 1.

The workbench exposes the four familiar projections from the one active candidate
with an explicit unfinished label; the design package is never formal delivery.

### Versioned episode candidates (script batch)

A unified run writes the complete series as one design-bound script batch of
immutable episode candidate versions plus one active pointer per episode.

- A **script batch** binds one exact SeriesBible candidate (id, content hash,
  design epoch) and carries an active pointer per episode. At most one batch is
  active per run; a design hash/epoch change supersedes the whole prior batch,
  clears the active projection, and starts a fresh batch at episode 1.
- An **episode candidate** is immutable and binds the design identity, batch and
  epoch, episode number, candidate version, predecessor hash, the generation
  call id, bounded advisory WriterNotes, the state delta, and the folded
  SeriesState. Rewriting episode N preserves active 1..N-1, supersedes every
  active candidate N..end, and replays SeriesState strictly from the retained
  prefix; no fact, knowledge, clue, obligation, or delta from the superseded
  suffix enters the replayed context.
- A candidate becomes active only through a transactional/CAS commit that
  re-validates the exact design binding, the active predecessor pointer, the
  next version, and deterministic contract/state replay. A failing or late
  candidate is retained as non-active (or `stale`) evidence with its usage and
  can never move an active pointer.
- Every episode request is assembled by a lossless context compiler. It
  verifies the complete committed prefix by content hash, then exposes the
  active SeriesBible projections, the locked contract, the exact current-group
  Canon projection, the verbatim recent committed window, deterministically
  referenced older scripts, the folded SeriesState, and bounded WriterNotes. No
  summary substitutes prior scripts, and WriterNotes are never canonical.

The active candidate projection stays readable through the workbench after a
refresh or restart; only a complete active batch assembles into the episode
scripts, which are never formal delivery until the final gate.

### Bounded structural review and repair authorization

The SeriesBible declares structural review milestones (`review_milestones`); the
final episode is always a structural milestone. The configured review role
reviews only these milestones and the final completion.

Screenplay validation is format-agnostic. Do not add per-artifact, per-language,
or legacy-format allowlists to make a rejected script pass. Deterministic checks
may enforce only explicit structured bindings, such as a fact's own bound
evidence excerpt. Headings, labels, dialogue notation, story-world arithmetic,
dates, code, and other subject matter require contextual review and may block
only when they directly contradict hard Canon or contain a proven private-runtime
leak. After changing this policy, validate it with a fresh isolated run; retain
old drafts and reviews as audit evidence, but do not mutate, reclassify, or
resume them as compatibility work.

- A **bound structural review** observes the complete active prefix at one
  milestone and binds the exact design candidate, script batch/epoch,
  active-prefix hash, and model-call id. It returns a deterministic category
  (pass / design defect / script defect) and, for a script defect, the earliest
  affected episode. Reviews are immutable evidence; a review bound to a
  different design or batch is retained as `stale` and can never approve,
  rebuild, rewrite, or deliver the active lineage.
- A **design defect** triggers the one automatic complete design regeneration
  per run lineage: the approved outline and downstream stages are reset, the
  prior script batch is superseded, and writing restarts at episode 1. When the
  automatic budget is exhausted, the run pauses for authorization.
- A **script defect** consumes the single automatic suffix-rewrite budget shared
  by all milestone and final reviews for the script batch: active 1..N-1 are
  preserved, active N..end are superseded, SeriesState is replayed, and the
  run requeues to rewrite the suffix. When the budget is exhausted, the run
  pauses for authorization.
- A **repair authorization** is bound to the active lineage, shows the evidence,
  the affected range, and a reference token count for the active design
  projections and retained prefix at the pause (neither a lower bound nor a total
  cycle forecast), and grants exactly one generation-plus-review cycle. If a
  hard-constraint conflict remains, the run pauses with the latest review
  evidence. Generic Continue is for transient runtime/Relay/timeout states only
  and can never bypass a semantic rejection or spend a content-repair budget.
- Only a passing **bound final whole-series review** atomically freezes the
  active design and complete active script batch as formal delivery; without
  it `succeed_run` refuses to persist a delivery.

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
| Deep Agents supervisor | Run-local planning, ordered specialist delegation, bounded correction loops, structured candidate assembly | Domain state transitions, retry entitlement, final gate authority |
| Synchronous subagents | Stage-scoped creative work, skill-scoped canon/continuity review, and repair candidates | Job scheduling, direct database writes, persona mutation, silent lock mutation |
| Persona loader | Manifest/schema validation, immutable snapshots, read-only virtual context projection, bounded L5/L6 retrieval | Production persona content |
| LangGraph checkpointer | Thread messages, plans, subagent results, and `StateBackend` scratch checkpoints | Business checkpoints, public run state, revision entitlement |
| Relay clients | Role-bound `ChatAnthropic` generation plus `ChatDeepSeek`, `ChatOpenAI`, or `ChatAnthropic` review sharing one URL/key, explicit timeouts, model-identity audit, safe error mapping | Workflow retry policy, API keys at rest, cross-role fallback |
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
6. The episode planner returns both the human outline and structured story
   contract. Deterministic validation and an independent model reviewer must
   pass before the creation service locks their hash-bearing checkpoint.
7. Each episode is generated against that immutable contract and the previous
   folded series state. Deterministic and independent model checks must pass
   before its script and state evidence commit atomically.
8. `quality_reviewer` produces the structured evidence packs for the L0 and L4
   gates. The creation service evaluates the required gate contract and owns
   the pass/fail state transition.
9. Before L4, the service reconstructs the complete scripts checkpoint from
   every locked episode and verifies the contract, script, and state hashes.
10. The complete content package and delivery report commit atomically with the
   run's `succeeded` state.
11. Each guarded attempt records the current internal stage in SQLite. Resource
   queries map it and approved checkpoints to the six user stages, completed
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
5. Revision progress and interruption control use the same persisted contract and
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
- The first worker wall-clock timeout or approved temporary relay interruption
  in a user stage parks the active clock, requeues the same run and `thread_id`,
  and automatically resumes from approved business checkpoints. Relay recovery
  waits at least 10 seconds (or a longer relay `Retry-After`) before the
  unapproved in-flight stage is invoked again.
- A second shared timeout or retryable relay interruption in the same user
  stage or first unfinished episode parks the job in `paused`. Idempotent
  operator commands may requeue that same run or mark it `ended`.
  Configuration errors detectable before a request, certificate verification,
  authentication, parameter/protocol incompatibility, graph recursion
  exhaustion, invalid structured output, quality-gate rejection, missing
  checkpoints, and unknown failures remain terminal. A syntactically valid
  relay address that then has a DNS or connection failure is indistinguishable
  from a transient transport failure, so it follows the bounded recovery path
  and becomes terminal when the three-call limit is exhausted.
- Contract and per-episode content review use a separate two-repair budget.
  Exhaustion records `content_rejected`, the exact review evidence, and the
  current episode when applicable, then pauses with continue/end actions.
  Transport failures never consume this content budget, and continuing never
  unlocks or rewrites previously committed contract or episode state.
- A paused or ended job is excluded from lease-expiry reconciliation. Refresh
  reconstructs its stage, completed checkpoints, frozen elapsed time, and
  available actions from SQLite.

## Persona-package loading contract

`manifest.json` is governed by `contracts/persona-package.schema.json`. The
manifest is metadata and is not one of the content files.

For schema v3 (and v2), `package_sha256` is the SHA-256 of the canonical
ordered concatenation of the eight lowercase per-file SHA-256 values in this
order: `paradigm`, `project`, `l0`, `soul`, `l3`, `l4`, `l5`, `l6`. Historical
v1 packages hashed nine values with `l1` and `l2` in place of `soul`. The
manifest itself is excluded, avoiding a circular hash.

API `snapshot_sha256` is the domain-separated SHA-256 of `package_sha256` plus
the canonical complete manifest. It therefore addresses the immutable package
identity as well as its Markdown content: two personas or versions may reuse
identical Markdown without sharing or replacing a snapshot.

| Logical file | Fixed name | Minimum required structure | Runtime use | V1 write policy |
|---|---|---|---|---|
| Paradigm | `paradigm.md` | L0-L6 definitions; arbitration; L0 structure; translation; feedback, blank-space, gates, discipline, ownership | Rule provenance and conflict resolution | Read-only |
| Project instruction | `project.md` | Identity; L0 handling; layer arbitration; workflow; gates/feedback summary; ownership | Deterministically inlined in full into the five content generation/repair subagents and both direct patch calls; not carried by the supervisor or reviewers | Read-only |
| L0 | `l0.md` | Variants; red lines; temperature; item ownership/status markers | Included through project context; gate source | Read-only |
| Soul | `soul.md` | Stable creative identity and expression defaults, compiled offline from historical L1/L2 sources | Full text read by every model stage; never summarized, sliced, or silently truncated | Read-only |
| L3 | `l3.md` | Creative methods and cognitive path | Full text enters the working context as creative-decision method; cannot override L0 or reopen approved directions | Read-only |
| L4 | `l4.md` | Hard rules (creator-confirmed), confirmed creative advice, and Pengine-owned parameter sections | Stage-projected: L0 selection and gates read L4-A; generation stages read stage rules plus general rules; the final gate reads the full L4 | Read-only |
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

- `/persona/` is read-only and contains the full Project instruction, required
  L0 material, the full Soul text, the full L3 method text (v3), and
  stage-scoped L4 material;
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
remain separate from the immutable eight-file persona authority.

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
- `relay_base_url`, `relay_api_key`, `generation_model_id`, and
  `review_model_id` are deployment configuration. The two role-specific output
  caps are optional.
- The worker atomically constructs both process-level clients from the shared
  URL/key: `ChatAnthropic` with exactly `claude-opus-5` for generation and
  creative repair, and the Anthropic, OpenAI, or DeepSeek client selected by
  `review_model_id` for review. Missing either route fails closed before workflow
  execution; neither client is used as fallback for the other.
- The relay must support the selected clients' tool-use protocols at the
  configured root. Supervisor and subagent schemas use LangChain `ToolStrategy`
  rather than provider-native structured-output extensions.
- `PENGINE_RELAY_ADAPTER`, `PENGINE_RELAY_MODEL_ID`, and
  `PENGINE_RELAY_MAX_OUTPUT_TOKENS` are obsolete, ignored settings; they cannot
  configure or override either role.
- The API key is read from the environment, never returned by the API, stored in
  SQLite, written to logs, or committed.
- Application logs contain identifiers, states, durations, attempt counts, safe
  error identifiers, and model identifiers only. They exclude creative content,
  complete prompts, raw provider responses, and secrets.
- Persona paths are resolved below the configured persona root; absolute paths
  and traversal outside that root are rejected.
- The default Deep Agents general-purpose subagent is disabled. Eleven
  synchronous subagents are registered: the four primary subagents, the legacy
  `quality_reviewer`, `canon_reviewer`, `episode_reviewer`,
  `repair_constraint_extractor`, `repair_constraint_validator`,
  `series_reviewer`, `episode_repair`, and `story_repair`. Four of the
  review/repair agents load dedicated skills; direct story/outline patch
  generators are bounded structured generation calls, not additional subagents.
- `FilesystemBackend`, `LocalShellBackend`, sandbox `execute`, asynchronous or
  remote subagents, arbitrary MCP tools, agent-authored skills, and
  cross-creation writable memory are disabled in V1.
- Agent code has no tool that writes the persona source root, snapshots, or
  business tables. Structured stage output crosses back through the creation
  service for validation and persistence.
- Relay and parsing failures map to stable safe errors; only approved temporary
  relay interruptions enter recovery. Vendor bodies, database messages, and
  stack traces do not cross the API boundary.

## Reliability and operations

- Every configured LangChain client has `max_retries=0`, and
  no Deep Agents retry middleware is installed. The worker is the sole owner of
  the three-attempt budget.
- A finite LangGraph recursion limit and a worker-enforced wall-clock deadline
  bound each run attempt. Graph recursion exhaustion fails with its own stable
  error. A wall-clock timeout and an approved relay interruption share the
  one-auto-resume/then-pause policy and never start an unrecorded model call.
- The generation cap defaults to the Opus 5 maximum of 128,000 output tokens.
  An unset review cap is not supplemented by Pengine. DeepSeek thinking is
  disabled, and parallel tool calls are disabled on every client; provider-specific prompt caching and
  beta-only Anthropic features remain disabled until both configured relay
  routes pass smoke testing.
- Every response must report the model ID configured for its role. A missing or
  mismatched identity is a terminal protocol incompatibility, not a fallback
  signal.
- Every real model request is preflighted at the common outbound-call boundary
  against the route's verified context window
  (`generation_context_limit_tokens` / `review_context_limit_tokens`). The
  estimate covers the actual serialized system prompt, messages, tools/schema,
  complete canonical context, and the requested output reserve. If the request
  cannot fit, or the route has no trustworthy verified limit, no request is
  dispatched: the call is recorded as `preflight_blocked` and the run pauses
  (`context_budget`) with prior approved work unchanged.
- Every attempted call (generation, review, repair, and blocked attempts) is
  recorded with a unique physical `call_id`, business `operation_id`, role,
  adapter/provider/model, stage,
  episode, candidate and batch lineage, estimated input/output totals, verified
  limit, provider-reported input/output/cache usage (or explicit
  `unavailable`), duration, finish reason, and outcome. Records are written
  immediately to the SQLite `model_calls` table, emitted as structured logs,
  exposed on the creation resource (`RunProgress.model_calls`), and rendered in
  the workbench. Provider actual usage is never inferred or backfilled from
  estimates; failed, timed-out, superseded, stale, and preflight-blocked calls
  keep their own classifications in per-round and run totals.
- Outbound call budgets are distinct from graph recursion and business-stage
  attempts. Defaults are 48 generation and 32 review calls per ordinary stage,
  plus whole-script-stage totals of 192 generation and 128 review calls. The
  script stage also retains per-episode role limits. Reservation happens before
  provider dispatch; exhaustion is recorded as `agent_execution_limit`.
- An approved StoryContract checkpoint, active episode candidate, or bound
  structural review must reference the successful physical call for its exact
  operation. Synthetic run/episode-derived call ids are not valid provenance,
  and audit writes are drained before a run is published as succeeded.
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
  Manual continue/end is available only after the second shared interruption
  pauses a run.

## Compatibility, migration, and rollback

This is a greenfield V1. The HTTP contract, manifest schema, and SQLite schema
each carry an explicit version.

The implementation must lock exact tested versions of `deepagents`,
`langchain-anthropic`, `langchain-deepseek`, `langchain-openai`, `langgraph`, and
`langgraph-checkpoint-sqlite`. Upgrading any of them is a deliberate
compatibility change requiring the supervisor, subagent structured-output,
permissions, checkpoint-resume, and relay contract tests to pass again. V1 does
not track prerelease dependency versions.

Compatible additive HTTP changes may stay within V1. The progress fields and
run-control commands are additive. Breaking HTTP or persona format changes
require a new contract version. Persona source packages remain outside the
database so application rollback does not rewrite operator content.

The SQLite schema is a forward-only migration chain, currently at version 29.
Across the chain it preserves prior contract-bound content and repair records,
migrates episode attempts into explicit rewrite cycles, adds model-call
`operation_id`, binds approved outline checkpoints to physical `review_call_id`
provenance, and introduces SeriesBible candidates, script batches,
structural-review receipts, grouped outline season maps with markdown sidecars,
and quality-gate repair tracking. Migrations
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
- `ChatDeepSeek` supplies the role-bound DeepSeek OpenAI-compatible client:
  <https://docs.langchain.com/oss/python/integrations/chat/deepseek>
- `ChatOpenAI` supplies the configured OpenAI-compatible review client for
  `gpt-5.5` and `gpt-5.6-terra`.

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
  only the eleven registered synchronous subagents, routes each subagent and
  direct repair call to the fixed model role, returns structured results, and
  cannot access host files, shell, MCP, or cross-thread memory;
- checkpoint tests proving a stopped run resumes the same `thread_id`, approved
  business stages are not regenerated, and a new revision-attempt run uses a
  new thread;
- fake-relay tests proving `ChatAnthropic` generation and the configured
  DeepSeek/OpenAI/Anthropic review request mapping, exact role routing, no fallback, tool-use capability, SDK and
  middleware retries disabled, structured output validation, bounded agent
  execution, identity auditing, and safe failures;
- a configured dual-route relay smoke test that separately reports generation
  and review external-service evidence;
- at least two real operator-supplied persona packages before claiming
  persona-selection and persona-effect UAT.

Engineering tests with non-production fixtures cannot prove real persona
quality.

## Decisions and open items

### Decisions

- Modular monolith, FastAPI/Pydantic HTTP layer, direct SQLite repository, one
  embedded worker, and an embedded Deep Agents/LangGraph creative runtime.
- One persona-bound `workflow_supervisor`; eleven synchronous subagents: the
  four primary subagents, the legacy `quality_reviewer`, `canon_reviewer`,
  `episode_reviewer`, `series_reviewer`, the two repair-constraint helpers,
  `episode_repair`, and `story_repair`. Four review/repair agents load
  dedicated skills.
- Two preconfigured LangChain clients share the operator-supplied relay URL/key:
  `ChatAnthropic` with `claude-opus-5` for generation and creative repair, and a
  review client whose Anthropic, OpenAI, or DeepSeek protocol is selected from
  `review_model_id`. Both disable automatic retries; both routes are required
  and neither falls back to the other.
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
- Wall-clock timeout and approved temporary relay interruption share durable
  recovery; recursion, structured-output, quality-gate, request-preflight
  configuration, certificate verification, authentication, protocol, and
  unknown failures remain terminal. A syntactically valid relay address with a
  DNS or connection failure follows the bounded temporary-transport path.
- `ContentPackage` and `DeliveryReport` are separate contract objects.

### Open items

- No architecture decision remains open.
- The governing delivery source is
  <https://github.com/mindcarver/pengine/issues/1>.
- Of the four bundled persona v3 packages, only `shouzhuo` carries
  creator-confirmed L0/Soul/L3/L4 material; the other three remain explicitly
  non-production prototypes.
- The configured generation and review relay routes have exercised real tool-use
  and structured-output paths, but a complete workbench initial-plus-revision
  acceptance run for Issue #10 remains required.
