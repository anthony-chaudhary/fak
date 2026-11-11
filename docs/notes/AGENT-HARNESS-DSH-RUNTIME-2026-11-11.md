# Harness runtime: issue-driven task spawning (2026-11-11)

Canonical leaf for harness-scoped runtime detail referenced from `AGENTS.md`
(router-shape gate, Ref #1798): scoped harness rules live here, not in the
router itself.

## Before spawning one fresh task per issue

Including direct Codex `create_thread` waves: normalize every dependency
through the issue contract and run the repository's issue/dispatch admission
surface first. Newly authored work uses exactly three relations:

- `Start blocked by: #N` may prevent pickup only when no executable disjoint
  slice exists;
- `Coordinates with: #N` is non-blocking alignment;
- `Promotion requires: #N` gates closure or performance claims, not
  implementation.

A broad `After:` list or a hand-written prompt saying all dependency fences
are binding is not an admission contract. If an issue is triage-only, repair
its scope and typed relations before spawning it; direct task packets must
repeat the same typed pickup rule.
