---
name: story-repair
description: Rewrite an unlocked character+relationships candidate to resolve every confirmed canon-review issue.
---

# Story repair

Read the current unlocked candidate in
`/workspace/current_story_candidate.md` and every confirmed review issue in
`/workspace/story_review.json`, plus every approved upstream artifact. Return
the complete corrected `character_biographies` and `relationship_logic` in one
structured result.

Resolve issues jointly across both sections: when one issue's fix changes a
fact that another section references, update every affected occurrence in both
sections consistently. Prefer one coherent rewrite over many local edits. The
two sections are one mutually consistent package, so re-audit identities, ages,
aliases, motives, secrets, relationship direction, timelines, and causal logic
across both before returning.

Keep every approved upstream artifact unchanged. Never invent canon that the
upstream artifacts leave unspecified. Copy authoritative corrected literals
directly from the review issues; never recompute ages, durations, dates, or
differences from the conflicting candidate.

When `/workspace/previous_story_review.json` is supplied, it contains the
confirmed issues that motivated the current candidate. Resolve every issue
listed there as well as every issue in `/workspace/story_review.json`.
