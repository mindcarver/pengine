---
name: canon-review
description: Review an unlocked short-drama story artifact or contract before it becomes immutable.
---

# Canon review

Read every supplied approved upstream artifact and the complete unlocked
candidate named in the task. The candidate may be a prose story outline,
character biography set, relationship document, or the system-generated
minimum continuity ledger. The user supplies only a free-form idea and optional
requirements.

Fail closed when any commitment is absent, ambiguous, or contradictory. Check:

- one closed cast with stable identity and relationship direction;
- every established alias, pronoun, age, elapsed duration, call participant,
  identity and relationship fact, and canonical clue meaning;
- dates, times, typed numbers, units, and arithmetic meanings;
- chronological event order;
- each character's knowledge after every episode;
- every clue's visible or audible introduction, one explanation, and callback;
- one effective new-information fact and one verifiable hook per episode;
- prohibitions from the approved story and persona context.

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

Every returned issue must be a blocking contradiction or missing binding
commitment. Never put a preference, optional clarification, naming suggestion,
or a condition explicitly described as "not a failure" into `issues`.

Leave genuinely unspecified creative details unspecified. Do not demand facts
that exist only to make validation easier, and never ask the user to complete a
character sheet, timeline, or evidence table.

Return concrete structured evidence. A plausible outline is not enough: pass
only when another writer can generate every episode without inventing canon.
For a prose candidate, cite the exact conflicting candidate excerpt and identify
the authoritative upstream value and source. State the exact corrected literals
or wording so the repair does not need to infer arithmetic. When alternatives
exist, select one repair invariant that preserves the approved upstream facts
and enumerate every downstream occurrence that must follow it; never offer an
alternative that changes the meaning of an approved upstream artifact. Review
only the current unlocked candidate; never propose changing an approved
upstream artifact.
