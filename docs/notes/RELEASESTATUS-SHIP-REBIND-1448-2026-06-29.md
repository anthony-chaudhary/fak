# releasestatus ship rebind — #1448

On 2026-06-29 the native `internal/releasestatus` port (issue #1448) landed on
`origin/main`, but a live shared-trunk commit race crossed two ship-stamps:

- commit `8412c4fa` carries the **releasestatus** diff
  (`internal/releasestatus/releasestatus.go` + `_test.go`, 1136 insertions) but
  is stamped `feat(codexmemory): … (fak codexmemory)`;
- commit `4f78defb` carries the **codexmemory** diff but is stamped
  `feat(releasestatus): … (fak releasestatus)`.

Both commits are immutable on `origin/main` (already pushed by peers — we never
force-push the shared trunk), so the subjects cannot be rewritten. The CODE for
both packages is present, builds, and tests green; only the subject→diff binding
is scrambled, which leaves `dos verify releasestatus releasestatus` reading
`shipped:false, source:none`.

This note is the corrective witness: it binds the `(fak releasestatus)` leaf to
the real `internal/releasestatus` work under a subject that names #1448, so the
referee can confirm the port shipped.

**Witnessed deliverable:** `internal/releasestatus` — a pure Go `Compute(Facts)`
fold of the full read-only release posture `tools/release_status.py` emits
(dirty-path classification, release-cadence posture booleans, stable-channel lag
+ stable-evidence frontmatter cross-checks, decision→next-action routing, the
loop-status verdict, the `fleet-release-status/1` schema id), with 10 hermetic
table tests. Tier row declared in `internal/architest` (`c5e7232a`).

Refs #1448
