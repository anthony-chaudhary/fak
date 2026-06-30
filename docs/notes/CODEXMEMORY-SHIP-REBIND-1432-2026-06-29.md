# codexmemory ship rebind — #1432

On 2026-06-29 the read-only Codex memory posture doctor (issue #1432) landed on
`origin/main`, but the same live shared-trunk commit race that crossed #1448's
stamp crossed this one too:

- commit `4f78defb` carries the **codexmemory** diff
  (`internal/codexmemory/codexmemory.go` + `_test.go`, `cmd/fak/codexmemory.go`,
  `cmd/fak/main.go`) but is stamped `feat(releasestatus): … (fak releasestatus)`;
- commit `8412c4fa` carries the **releasestatus** diff but is stamped
  `feat(codexmemory): … (fak codexmemory)`.

Both commits are immutable on `origin/main` (already pushed by peers — we never
force-push the shared trunk). The CODE builds and tests green; only the
subject→diff binding is scrambled, so `dos verify codexmemory codexmemory` reads
`shipped:false, source:none`.

This note is the corrective witness: it binds the `(fak codexmemory)` leaf to the
real `internal/codexmemory` work under a subject that names #1432, so the referee
can confirm the doctor shipped.

**Witnessed deliverable:** `fak codex-memory doctor [--codex-home DIR] [--json]`
— a READ-ONLY posture diagnostic over a Codex home that reports
`[features].memories`, `memories.use_memories` / `generate_memories` /
`disable_on_external_context` / `min_rate_limit_remaining_percent` /
`extract_model` / `consolidation_model`, inventories generated state +
Chronicle as content-free counts only (never prints raw memory text), reports
three-state posture honestly on a missing/partial home, and flags
external-context inclusion / large stores / Chronicle as advisory risks. Tier
row declared in `internal/architest` (line 60).

Refs #1432
