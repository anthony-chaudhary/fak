<!-- fak-benchwave-key: benchwave-servelaunch-l1 -->
# Bench-wave servelaunch L1: readiness probe composite in new internal/servelaunch (oss-port-bench-servelaunch-20261109 leaf 1)

```routing
lane: internal/servelaunch
tracker: pub
```

Process cause: none

## Provenance (parent oss-port-bench-servelaunch-20261109)
- Source: `anthony-chaudhary/Benchmark@b60f2f05e23ec38ae4f50ca19717c2f615ed419a`, license
  MIT-declared with no root LICENSE → **clean-room conceptual adaptation, ADAPT verdict; no
  source copied**. Behavior contracts below come from the pinned private study note (§A).
- Parent plan: fak-private `docs/tickets/borrow-port-parts/oss-port-bench-servelaunch-20261109.md`.
- This leaf is leaf 1 of 2; leaf 2 (teardown + ports) is dependency-chained and dispatches
  after this leaf lands.

## Work items (this leaf only)
1. Create `internal/servelaunch/readiness.go`: a readiness-probe composite that polls a health
   URL at a fixed interval with a deadline, emits progress heartbeats, races a caller-supplied
   crash side-channel (process-exit signal + last-log-lines fetcher) so a fast crash returns
   `Crashed` in seconds instead of riding the timeout, and picks a warm/cold deadline budget
   via a caller-supplied cache-state probe (cold when the probe reports cold, warm when warm).
2. Closed-contract acceptance:
   - typed verdict Ready / Timeout / Crashed distinguishable via sentinel errors (`errors.Is`),
     never error-prose matching;
   - crash side-channel fires `Crashed` before the deadline when the exit-callback fires, and
     the crash evidence carries the fetched last-log lines;
   - deadline budget is a pure function over the cache-state probe result; both branches tested;
   - heartbeat callback invoked at the configured interval on long waits; injectable
     clock/ticker — no real wall-clock sleeps in tests.
3. Recon BEFORE writing (read-only): `internal/armbench/manifest.go` (bench-harness consumer
   shape) and `internal/servewarmup/warmup.go` (existing serving-lifecycle neighbor whose
   naming to match).

## Boundaries
- Create and edit ONLY `internal/servelaunch/readiness.go` and its exact test sibling
  `internal/servelaunch/readiness_test.go` in this public `fak` repo. Never touch go.mod,
  go.sum, sibling packages, or fak-private.
- No new scripts (.ps1/.sh/.bat/.cmd). Automation stays native Go.
- No real process spawning and no external network in tests (httptest or fake poller only).
- Anti-brittle tests: assert closed contracts (sentinels / `errors.Is`), never error prose.

## Done condition
- `go test ./internal/servelaunch -run 'TestReadiness' -count=1` green in a fresh run (forward
  slashes); `go vet ./internal/servelaunch/...` clean; `gofmt` clean.
- Committed by explicit path with DCO sign-off, subject = so-what, line 2 = why, line 3 =
  witness, and the parent trailer `(fak oss-port-bench-servelaunch-20261109)` in the subject
  so borrow-queue plan detects the parent from commit history.
- The public pre-commit hook runs the secret leak audit and boundary gates automatically; do
  not bypass them.
