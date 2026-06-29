# Example output

A captured run of `./examples/steward-demo/run.sh --no-color`. The demo is deterministic —
the same `sample-steward.json` always yields these verdicts. A `✓` means the verdict matched
the witness model; the run exits `0`.

Reproduce: `./examples/steward-demo/run.sh --no-color`.

```
fak kernel — single-invariant steward demo  config=sample-steward.json
  a steward RAISES only when (pattern matches) AND (an INDEPENDENT witness corroborates); a ✓ means the verdict matched the witness model

STEWARDS  — one invariant each; raise needs an independently-authored witness
  no-secret-in-context  invariant: no secret-shaped bytes are ever admitted to the model's context window
      ✓ secret admitted, independently scanned -> RAISE            RAISE                    witness="secret-shaped bytes admitted to context: sk-abcde…"  by secretscan
      ✓ same pattern, NO independent scan authored a witness -> SUPPRESSED ABSTAIN                  pattern matched but NO independent witness — self-accusation suppressed
      ✓ clean context, nothing to scan -> ABSTAIN                  ABSTAIN                  invariant pattern did not trip

  lease-disjointness  invariant: no two live file-tree leases share a path prefix
      ✓ two leases collide on a tree prefix, arbiter confirms -> RAISE RAISE                    witness="a and b collide on internal/abi/internal/abi/types"  by arbiter
      ✓ fully disjoint leases -> ABSTAIN                           ABSTAIN                  invariant pattern did not trip

  kpi-regression  invariant: the current p50 never regresses past the committed baseline by more than tol
      ✓ current p50 doubled vs committed baseline, scoreboard confirms -> RAISE RAISE                    witness="p50 regressed: baseline=1 current=2"  by scoreboard
      ✓ current == baseline -> ABSTAIN                             ABSTAIN                  invariant pattern did not trip

  vdso-soundness  invariant: a cached-path result is bit-identical to a fresh pure recompute
      ✓ cache hit disagrees with a fresh recompute, replayer confirms -> RAISE RAISE                    witness="vdso cache hit != fresh pure call (recompute disagreed)"  by replayer
      ✓ cache hit equals the fresh recompute -> ABSTAIN            ABSTAIN                  invariant pattern did not trip

META-STEWARD  — after a 3-sweep soak, prune any steward that NEVER fired
      keep  no-secret-in-context   fired 3× during the soak
      keep  lease-disjointness     fired 3× during the soak
      keep  kpi-regression         fired 3× during the soak
      keep  vdso-soundness         fired 3× during the soak
      prune clock-monotonic        fired 0× — guards a clock-skew invariant the platform made impossible; the probe can no longer trip, so the steward is dead weight

  meta-steward pruned: ['clock-monotonic']   kept: ['no-secret-in-context', 'lease-disjointness', 'kpi-regression', 'vdso-soundness']

summary: steward demo passed  ·  4 raises (each with an independent witness)  ·  1 un-witnessed match(es) suppressed  ·  1 never-firing steward pruned
  the load-bearing result: a steward raises ONLY when an independently-authored witness corroborates the invariant match — the model can never self-accuse, and a check that can never fire is pruned.
```

The two rows to read together are the first two `no-secret-in-context` rows: the invariant
pattern is **identical** (the same `sk-…` token), and the only difference is whether an
*independent* scan authored the witness. With a witness the steward **RAISES**; without one
it is **SUPPRESSED** — the model cannot self-accuse. That asymmetry is the whole point.

The authoritative witness that the shipped feature behaves this way is the Go test suite,
`internal/steward/steward_test.go` (units 87–92):

```
go test ./internal/steward/ -v -run \
  'TestSecretInContext|TestLeaseDisjointness|TestKPIRegression|TestVDSOSoundness|TestPrunePopulation'
```
