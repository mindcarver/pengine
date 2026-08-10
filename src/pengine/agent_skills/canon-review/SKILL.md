---
name: canon-review
description: Review an unlocked short-drama story artifact or contract before it becomes immutable.
---

# Canon review

Read every supplied approved upstream artifact and the complete unlocked
candidate named in the task. Use those artifacts to identify the smallest set
of explicit hard Canon that the candidate must preserve. Approved prose is
context; it is not automatically a lock for every detail it omits. The user
supplies only a free-form idea and optional requirements.

Fail only when explicit hard Canon is contradicted or a required locked binding
is impossible. For this review, hard Canon comes only from user requirements,
values explicitly marked locked in the Contract or SeriesBible, formally
committed facts in prior approved episodes or state, mandatory episode
obligations, and the output/schema protocol. Ordinary approved prose, persona
style, suggestions, and omitted fields are not locks. Do not turn an omitted
creative choice into a failure. Check only the applicable hard-Canon items:

- the closed cast, stable identities, aliases, pronouns, and relationship
  directions that are explicitly locked;
- locked ages, elapsed durations, call participants, dates, times, typed
  numbers, units, arithmetic meanings, and chronological order;
- formally committed knowledge states and causal facts;
- mandatory clue lifecycle, new-information beat, and verifiable hook items;
- explicit prohibitions from the approved story and persona context.

When the task assigns a review lens, stay within that lens but audit every
candidate section, summary table, ending statement, and repeated mention.
Collect all issues in that lens before returning; do not stop after finding the
first few examples.

When `/workspace/previous_story_review.json` is supplied, read its
`issue_ledger` before judging the candidate. Return exactly one
`prior_issue_closures` entry for every ledger `issue_id`, even when the original
issue falls outside the assigned lens. Mark an item resolved only when the
complete current candidate and approved upstream artifacts prove that every
conflicting occurrence is closed; otherwise mark it unresolved and cite the
remaining current-candidate evidence. Never omit an ID or invent a new one.
Treat wording suggested by an earlier review as a hypothesis, not an authority:
the creation request and approved upstream artifacts always take precedence.

Every returned issue must be a blocking contradiction to explicit hard Canon or
an explicitly required locked binding. Never put a preference, optional
clarification, naming suggestion, request to fill an unspecified detail, or a
condition explicitly described as "not a failure" into `issues`.

Leave genuinely unspecified creative details unspecified. Do not demand facts
that exist only to make validation easier, and never ask the user to complete a
character sheet, timeline, or evidence table.

Return concrete structured evidence. A plausible contradiction claim is not
enough: pass when the candidate does not violate hard Canon, even when the
writer is free to choose unspecified details. A later writer may invent within
that free space; it may not contradict locked Canon.
For a prose candidate, cite the exact conflicting candidate excerpt and identify
the authoritative upstream value and source. State the exact corrected literals
or wording so the repair does not need to infer arithmetic. When alternatives
exist, select one repair invariant that preserves the approved upstream facts
and enumerate every downstream occurrence that must follow it; never offer an
alternative that changes the meaning of an approved upstream artifact. Review
only the current unlocked candidate; never propose changing an approved
upstream artifact.

When reviewing a structured story contract, make mutation authority explicit.
Set `contract_mutation_required=true` only when resolving that issue requires a
story-contract mutation, and then return `repair_targets`. Keep `contract_refs`
for Canon entity IDs; never use an ID's text as a collection permission. For an
existing item that must change, use `replace_existing`, its zero-based collection
index, an exact copy of the complete current item in `expected_value`, and the
exact complete replacement in `value`. Use `remove_existing` with the index and
exact current item when the item must be deleted. For a required item that is
absent, use `append_missing`, leave `index` and `expected_value` null, and copy the
exact complete new collection item into `value`. Give every target a unique
`target_id`. A target authorizes one item only. Return every target needed to
make the combined resulting StoryContract valid; the runtime applies them
atomically. Do not infer broad collection access or use issue-code wording as
permission.
