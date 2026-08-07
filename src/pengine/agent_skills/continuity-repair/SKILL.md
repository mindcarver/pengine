---
name: continuity-repair
description: Repair only an unlocked outline contract candidate or episode candidate.
---

# Continuity repair

Use the supplied structured review as a bounded repair list. Preserve all
explicit hard Canon, the locked contract, earlier locked episodes, episode
count, cast, timeline, typed facts, units, clue lifecycle, and knowledge state.
Do not turn unspecified creative details into new locks.

Change only the supplied unlocked candidate. Address every confirmed hard-Canon
issue and leave unspecified creative choices free.
When `/workspace/candidate_episode.md` and
`/workspace/candidate_state_delta.json` are supplied, use them as scratch files
for the repair and reread them before returning. The structured result is the
authoritative final candidate: make it complete, internally consistent, and
aligned with the intended repair rather than stale memory or an earlier draft.
For an episode repair, always return the complete non-null `state_delta` that
matches the repaired script, including every required evidence target exactly
once. Every `state_delta` list contains changes from the current episode only;
never copy cumulative values from `series_state` into a delta. For a mismatch,
use the exact expected IDs in the review issue's `contract_refs`.
Never weaken or silently rewrite the governing contract.
