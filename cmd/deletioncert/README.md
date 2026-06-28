# deletioncert - provable-deletion certificate

`deletioncert` provides two end-to-end demonstrations:

1. **Provable-deletion certificate** (`-selfcheck`): a no-model, deterministic proof that a selected KV span can be evicted so the surviving context is byte-identical to a run that never saw it, then bound into a tamper-evident deletion certificate.

2. **Per-tenant KV cache-isolation benchmark** (`-isolation-bench`): a structural floor over a fak-authored adversarial corpus that proves the L3 tier isolates tenants (G4 scope gate) and verifies page digests (G1 digest gate), with a leaky baseline that demonstrates the metric discriminates.

Both modes run offline over in-memory backends with no model weights, GPU, API key, network service, or durable output file required.

## Prerequisites

Requires Go only from the repository root. Both modes complete in seconds and return exit code 0 only if all checks pass.

## Provable-deletion demo

```bash
go run ./cmd/deletioncert -selfcheck
go run ./cmd/deletioncert -selfcheck -out deletioncert.json
```

The optional `-out` path writes the minted certificate JSON.

### What you see

```text
== fak provable-deletion demo ==
never-saw  continuation = [11 11 11 11 11 11]
kept-secret continuation = [11 11 11 11 11 109]
evicted    continuation = [11 11 11 11 11 11]
PROVEN: evicted == never-saw (max|Delta|=0); kept-secret differs.
VERIFY (intact)        -> valid=true sig=true anchor=true bound=true equiv=true
VERIFY (cert forged)   -> valid=false
VERIFY (scope forged)  -> valid=false
VERIFY (journal rewrit)-> valid=false
OK - provable-deletion certificate minted, verified, and tamper-rejected.
```

`go test ./cmd/deletioncert` now runs the selfcheck path as a regression guard in addition to the pure helper tests.

### Scope

This demo does not claim third-party timestamping, independent issuance, or a production deletion service. The certificate is self-signed v1; it proves integrity of the recorded facts and fail-closed verification over the synthetic-model witness. For the full honesty fences, see [`../../CLAIMS.md`](../../CLAIMS.md) and [`../../docs/proofs/deletioncert.md`](../../docs/proofs/deletioncert.md).

## Isolation benchmark

```bash
go run ./cmd/deletioncert -isolation-bench
go run ./cmd/deletioncert -isolation-bench -out isolation-result.json
go run ./cmd/deletioncert -isolation-bench -seed 42
```

### What you see

```text
== per-tenant KV cache-isolation benchmark ==
  scope: l3-working-set
  seed: 42
  corpus: 14 cases
  passed: 14
  failed: 0
  baseline leaked: 4/14 (discrimination proof)
  valid: true

OK — per-tenant KV cache-isolation benchmark passed.
```

### What it measures

The benchmark validates two isolation guarantees over a fixed adversarial corpus:

1. **Cross-tenant isolation (G4)**: Private pages (ScopeAgent/ScopeTenant) are NEVER served to a different tenant, while same-tenant sharing is preserved.

2. **Digest verification (G1)**: Pages returned by the L3 tier MUST hash to their claimed digest. A collision or mis-tag is REFUSED for EVERY reader.

The corpus covers:
- G4 bite: cross-tenant private refused
- G4 serve: fleet-scoped pages served across tenants
- G4 same-tenant: within-tenant sharing preserved
- G1 bite: digest mismatch refused (even on permissive paths)
- G1 fail-closed: empty digest refused
- Concurrent-load dimension: interleaved multi-tenant operations
- Control-path only: large pages with O(1) gate decisions

The **leaky baseline** (a backend without the G1/G4 gate) MUST demonstrate leaks, proving the metric discriminates.

### Scope

The benchmark proves **L3 tier isolation** (`l3-working-set`), NOT deletion from weights, backups, or replicas. It is a structural floor over a fak-authored corpus (AgentDojo-style), NOT a public-leaderboard rank. No incumbent benchmark exists — this establishes the metric, it does not compete on one.

For the threat model, corpus details, and reproduction artifacts, see [`../../docs/proofs/isolation-bench.md`](../../docs/proofs/isolation-bench.md).
