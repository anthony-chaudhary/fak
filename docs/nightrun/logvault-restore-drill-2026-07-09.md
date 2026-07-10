# logvault restore drill — first recorded run (2026-07-09)

Issue [#2453](../../) (feat(logvault): restore verb + restore drill), part of epic
#2447. A backup nobody has restored from is a hypothesis. This is the first
recorded run of the restore-path cadence drill, turning `fak logvault`'s
recoverability promise into a witnessed capability.

## What shipped

- **Restore verb** — `fak logvault restore <source-id> [-to DIR] [-at SEQ] [-force]`:
  copies one source's manifest-replayed state OUT of the vault into a fresh target
  (never in-place over a live store without `-force`), re-hashing every restored
  byte against the manifest chain and re-running the chained-journal verifiers
  (guard decision journals, usage logs) over restored journals. Library:
  `internal/logvault/restore.go` (`Vault.Restore`); CLI: `cmd/fak/logvault.go`.
- **Drill verb + script** — `fak logvault drill [<source-id>] [-ledger PATH]`
  restores one source into a temp dir, verifies, and appends a durable `DrillRow`
  to the vault's `drill-log.jsonl`. The cadence hook is
  `scripts/logvault-restore-drill.sh` (`make logvault-drill`): hermetic and
  self-contained (seeds its own source + vault, sandboxes HOME / config dirs), so
  it runs on any host / in CI without touching live state.

## Drill run — `make logvault-drill` (DRILL PASS)

```
logvault-restore-drill: capture seeded source -> temp vault (…/vault)
logvault capture  vault=…/vault
  dispatch-runs          files=1      unchanged=0      full=1     append=0    rewrite=0    errors=0   copy=36B
  dos-state              (absent on this box)
  …                      (absent on this box)
  user-fak-state         (absent on this box)
  harness-store          (absent on this box)
TOTAL files=1 copy=36B errors=0 (WITNESSED: sizes stat'd, hashes computed, by this run)
logvault-restore-drill: drill (restore one source into a temp dir + verify)
logvault drill  vault=…/vault  source=dispatch-runs (at seq=1)
  files=1 bytes=36B from-history=0 mismatches=0 journals=0 journals-failed=0
  DRILL PASS — restore round-tripped clean, row appended to drill-log.jsonl · anchor seq=1 hash=6da99e794489…
logvault-restore-drill: OK — the restore path round-trips clean
```

`mismatches=0` is the drill's re-hash-against-the-chain verdict: every restored
byte re-hashed to its manifest-attested sha256. The seeded drill source is a plain
log, so `journals=0` here (the drill exercises the re-hash path on every run).

## Guard-journal re-verify witness — `go test ./internal/logvault`

The acceptance's "passing verify on a restored guard journal" clause is pinned by
`TestRestoreRoundTripsCleanWithVerifiedJournal`: it captures a REAL hash-chained
guard decision journal, restores it, and re-runs the journal verifier against the
RESTORED copy (the equivalent of `fak audit verify`), asserting 0 mismatches and
2 sound rows. `TestRestoreAtReconstructsOlderPrefixState` pins `-at` older-state
reconstruction; `TestRestoreReportsUnprovableFileFailClosed` pins the fail-closed
posture (a bitrotten mirror is NAMED, never copied as "done").

```
=== RUN   TestRestoreRoundTripsCleanWithVerifiedJournal
--- PASS: TestRestoreRoundTripsCleanWithVerifiedJournal (0.04s)
=== RUN   TestRestoreAtReconstructsOlderPrefixState
--- PASS: TestRestoreAtReconstructsOlderPrefixState (0.03s)
=== RUN   TestDrillRestoresVerifiesAndJournalsPassRow
--- PASS: TestDrillRestoresVerifiesAndJournalsPassRow (0.05s)
PASS
ok  	github.com/anthony-chaudhary/fak/internal/logvault	0.285s
```

## Generation-close evidence (gen/now)

- **Promotion evidence**: the restore verb + drill are shipped, trunk-safe,
  reversible, and gated by focused tests (`go test ./internal/logvault` green) and
  a passing hermetic drill run recorded above. The drill is wired into
  `make logvault-drill` so it runs on a cadence and the path cannot rot silently.
- **Demotion / retirement evidence**: none — this promotes the vault from
  capture-only to restore-proven; nothing is retired. The prior "restore is an
  untested hypothesis" state is the thing demoted.
- **Invalidating assumption**: the drill's re-hash proves byte-integrity of the
  restore, but the hermetic seeded source is a plain log, so the drill run itself
  does not exercise the chained-journal re-verify branch — that branch is proven
  by the unit test above, not by the cadence drill. If a future regression only
  breaks chained-journal re-verify (not the re-hash path), the hermetic drill
  would stay green while the test catches it; seeding the drill with a real
  chained journal (needs a journal-writer surface the CLI does not yet expose)
  would close that gap. Recorded as the next checkable step, not a shipped claim.
