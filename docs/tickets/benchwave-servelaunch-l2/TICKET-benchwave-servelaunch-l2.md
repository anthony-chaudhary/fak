<!-- fak-benchwave-key: benchwave-servelaunch-l2 -->
# Bench-wave servelaunch L2: escalating teardown + fail-closed port claim in internal/servelaunch (oss-port-bench-servelaunch-20261109 leaf 2)

```routing
lane: internal/servelaunch
tracker: pub
```

Process cause: none

## Provenance (parent oss-port-bench-servelaunch-20261109)
- Source: `anthony-chaudhary/Benchmark@b60f2f05e23ec38ae4f50ca19717c2f615ed419a`, MIT-declared
  with no root LICENSE → **clean-room conceptual adaptation, ADAPT verdict; no source copied**.
- Parent plan: fak-private `docs/tickets/borrow-port-parts/oss-port-bench-servelaunch-20261109.md`.
- Leaf 2 of 2. Depends on leaf 1 (readiness.go, already landed) — same package.

## Work items (this leaf only)
1. Create `internal/servelaunch/teardown.go` and `internal/servelaunch/port.go`: the escalating
   idempotent stop ladder (SIGTERM → bounded kill-0-style poll → SIGKILL → port-kill safety
   net, PID-by-port discovery), a fail-closed port claimer that accepts only
   positively-confirmed-free ports, and a rollback-in-reverse helper for partial launches.
2. Closed-contract acceptance:
   - StopLadder escalates through injectable signaler steps in order and stops escalating on
     confirmed exit; escalation ladder and timeouts are data, not literals;
   - idempotent: stopping an already-stopped target returns a typed sentinel
     (`ErrAlreadyStopped` or equivalent), not an error prose match;
   - PortClaim: accepts a port only when the prober positively confirms it free; a prober
     error/inconclusive result is treated as busy (typed `ErrPortProbeInconclusive`); scan
     window is bounded;
   - Rollback helper stops already-started units in reverse start order (order asserted in
     test).
3. Recon BEFORE writing (read-only): the landed `internal/servelaunch/readiness.go` (match its
   injectable-clock/typed-sentinel style) and `internal/servewarmup/warmup.go`.

## Boundaries
- Create and edit ONLY `internal/servelaunch/teardown.go`, `internal/servelaunch/port.go`, and
  their exact test siblings in this public `fak` repo. Never touch readiness.go beyond imports
  if needed, go.mod, go.sum, sibling packages, or fak-private.
- No real process kill in tests (injectable signaler/port-prober); no network.
- No new scripts (.ps1/.sh/.bat/.cmd). Anti-brittle tests: closed contracts (sentinels /
  `errors.Is`), never error prose.

## Done condition
- `go test ./internal/servelaunch -run 'TestStopLadder|TestPortClaim|TestRollback' -count=1`
  green in a fresh run (forward slashes); `go test ./internal/servelaunch/... -count=1` green;
  `go vet ./internal/servelaunch/...` clean; `gofmt` clean.
- Committed by explicit path with DCO sign-off, subject = so-what, line 2 = why, line 3 =
  witness, subject ends with `(fak oss-port-bench-servelaunch-20261109)`.
