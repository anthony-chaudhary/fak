# per-tenant KV cache-isolation benchmark

## Overview

This benchmark establishes a **provable-deletion / per-tenant KV cache-isolation** metric for the fak L3 tier. It is a **structural floor over a fak-authored corpus** (AgentDojo-style), NOT a public-leaderboard rank. No incumbent benchmark exists — the mem0 2026 survey explicitly names both deletion dynamics and concurrent multi-tenant isolation as open gaps with no widely adopted public benchmark.

**Scope: `l3-working-set`** — this proves the L3 tier isolation, NOT deletion from weights, backups, or replicas. The honest boundary is stated below.

## What this measures

The benchmark validates **two isolation guarantees**:

1. **Cross-tenant isolation (G4)**: A private page (ScopeAgent/ScopeTenant) is NEVER served to a different tenant, while same-tenant sharing is preserved (the efficiency win is untouched).

2. **Digest verification (G1)**: The bytes the L3 get returns MUST hash to the page's claimed Digest. A collision or mis-tag is REFUSED for EVERY reader, so a corrupt page is served to no one.

The metric is **two-fold**:
- `deletion_isolation_score`: all cases pass → 100%, any failure → 0%
- `baseline_discrimination`: leaky baseline MUST fail (proves the metric discriminates)

## Threat model

### Adversary capability

The benchmark assumes an adversary who can:

1. **Read from the shared L3 pool** under a chosen tenant identity
2. **Observe a cross-tenant read attempt** (hit/miss timing oracle)
3. **Attempt to read a page owned by a different tenant**

The adversary CANNOT:

1. **Forge the page's content digest** (G1 catches this)
2. **Bypass the scope gate** (G4 catches this)
3. **Modify the page's control-path metadata** (backend is trusted)

### What the benchmark proves

- **No cross-tenant leak**: tenant B cannot read tenant A's private pages, even if they share the same content digest (prefix collision or mis-tag)
- **Same-tenant sharing preserved**: within a tenant, the cache hit path still works (efficiency win not regressed)
- **Fleet sharing preserved**: public/fleet-scoped pages are still served across tenants (opt-in sharing)
- **Digest integrity**: any page returned by the L3 tier is verified against its claimed digest before admission

### What the benchmark does NOT prove

- **Deletion from weights, backups, or replicas**: scope is `l3-working-set`, NOT `deleted-everywhere`. The certificate proves the L3 tier no longer serves the page, NOT that it's gone from all tiers.
- **Independence from the issuer**: the benchmark is self-run (no external auditor). The result is reproducible but not independently verified.
- **Side-channel closure**: timing channels beyond hit/miss are out of scope.
- **Middle-of-sequence eviction**: this is a read-path benchmark; write-time eviction (G2/G3) is separate work.

## The harness

The benchmark runs in `cmd/deletioncert` with the `-isolation-bench` flag:

```bash
go run ./cmd/deletioncert -isolation-bench
go run ./cmd/deletioncert -isolation-bench -out result.json
go run ./cmd/deletioncert -isolation-bench -seed 42
```

**Hardware requirements**: NONE. Runs on any machine with Go. Model-free, GPU-free, key-free, network-free.

**Runtime**: < 1 second for the 14-case corpus.

### The corpus

The corpus is a **fixed, seeded, fak-authored adversarial read-back corpus** of 14 deterministic cases:

| Case | Tests | Oracle |
|------|-------|--------|
| `cross-tenant-agent-private-refused` | G4 bite | REFUSED with `L3_CROSS_TENANT_SCOPE_DENIED` |
| `cross-tenant-tenant-private-refused` | G4 bite | REFUSED with `L3_CROSS_TENANT_SCOPE_DENIED` |
| `cross-tenant-fleet-served` | G4 serve | ADMITTED (fleet-scoped page) |
| `same-tenant-agent-private-served` | G4 same-tenant | ADMITTED (within-tenant sharing) |
| `same-tenant-tenant-scoped-served` | G4 same-tenant | ADMITTED (within-tenant sharing) |
| `digest-mismatch-fleet-refused` | G1 bite | REFUSED with `L3_PAGE_DIGEST_MISMATCH` |
| `digest-mismatch-same-tenant-refused` | G1 bite | REFUSED with `L3_PAGE_DIGEST_MISMATCH` |
| `empty-digest-fails-closed` | G1 fail-closed | REFUSED with `L3_PAGE_DIGEST_MISMATCH` |
| `concurrent-*` (5 cases) | Concurrent load | ADMITTED/REFUSED per (scope, owner, reader) |
| `large-fleet-page` | Control-path only | ADMITTED (single O(1) gate decision) |

Each case is **deterministic**: given the seed (default 42), the same bytes, scope, owner, and reader always produce the same verdict.

### The oracle

The oracle is a **deterministic pass/fail check** per case:

- `expectAdmitted`: whether the gate should admit the page
- `expectReason`: the typed refusal token (empty on admit)

The harness fails closed on any mismatch.

### The leaky baseline

The benchmark runs the SAME corpus against a **NULL / leaky baseline** — a `MockL3Backend` that stores and retrieves pages WITHOUT the G1/G4 gate. The baseline MUST demonstrate leaks (it MUST admit cross-tenant private reads), proving the metric discriminates:

```
baseline leaked: N/14 (discrimination proof)
```

If the baseline does NOT leak, the metric is not discriminating and the benchmark fails.

## The result

The benchmark emits a JSON result artifact:

```json
{
  "schema": "fak.isolation-bench/v1",
  "seed": 42,
  "scope": "l3-working-set",
  "generated_at": "2026-06-27",
  "git_commit": "unknown",
  "corpus_size": 14,
  "passed_cases": 14,
  "baseline_fails": true,
  "baseline_failed_cases": 4,
  "valid": true,
  "boundary": "fak's structural floor over a fak-authored read-back corpus, NOT a public-leaderboard rank; the max|Δ|=0 / L3 all-miss / scope-refusal witnesses are the honest figures, never a vs-naive-cache multiple"
}
```

- `valid`: true only if all cases pass AND the baseline leaked
- `baseline_failed_cases`: number of cross-tenant private leaks the baseline demonstrated
- `failed_cases`: per-case failure details (name, want, got, reason) — omitted when the run is clean
- `boundary`: the verbatim honest limit, stamped on EVERY emitted artifact so it travels with the score and is never buried in adjacent prose

### Honesty fences asserted over the artifact (CI-gradeable)

The harness re-checks three fences over the EMITTED result before it exits — they are not self-reports:

1. **Boundary present**: the verbatim `boundary` string above must be stamped, else the run fails closed.
2. **Scope honest**: `scope` is `l3-working-set` and no over-claim token (`deleted-everywhere`, `weights`, `backups`, `replicas`, `embeddings`) appears anywhere in the artifact.
3. **Control-path only**: no page payload byte from any corpus case may surface in the artifact — only digests, scopes, tenant tags, and counts. A corpus or field edit that smuggles in a secret reds CI.

These fences are pinned by Go tests in `cmd/deletioncert/deletioncert_test.go` (run under `go test ./...`, a `make ci` step) AND by the headless witnesses `deletioncert-isolation-bench[-out|-seed]` in `tools/demo_headless_smoke.py` (run under `make ci`'s `demo-headless-smoke`), so the benchmark is wired into CI on both the unit and command-surface paths.

## Reproduction

To reproduce the benchmark exactly:

1. Check out the fak repo at the `git_commit` in the result
2. Run `go run ./cmd/deletioncert -isolation-bench -seed <seed>`
3. Verify `valid` is true and `baseline_failed_cases` matches

The result JSON is committed under `docs/proofs/isolation-bench-result.json` for third-party verification.

## Related work

- **Issue #1065**: This benchmark addresses the "no incumbent exists" gap for deletion/isolation benchmarks.
- **mem0 State of AI Agent Memory 2026**: Names deletion/eviction dynamics and concurrent multi-tenant isolation as open gaps with no public benchmark.
- **L3 disaggregated cache epic (#75)**: The gate (child D) this benchmark validates.
- **DeletionCertificate (cmd/deletioncert)**: The sibling receipt for KV working-set eviction.

## Honest fences

- **Scope is `l3-working-set`**: This proves L3 tier isolation, NOT deletion from weights, backups, or replicas.
- **Self-run, not independently verified**: The benchmark is reproducible but not third-party-certified.
- **Structural floor, not leaderboard rank**: There is no incumbent benchmark; this establishes the metric, it does not compete on one.
- **WITNESSED capability**: All primitives (MockL3Backend, AdmitL3SharedPage) are fak-authored and fak-controlled.
- **Zero external deps**: Model-free, GPU-free, key-free, network-free. Runs entirely in-process.

## Contact

Questions or concerns? Open an issue on the fak repo.