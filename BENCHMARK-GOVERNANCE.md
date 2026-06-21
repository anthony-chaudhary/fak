# BENCHMARK GOVERNANCE — DOS-Centric Benchmark Claims System

> **One place for all benchmark claims.** This document defines the deterministic, DOS-centric
> process for creating, verifying, and publishing benchmark results. Every claim traces back
> to committed artifacts with `dos_commit_audit` verification.

**Last updated:** 2026-06-19
**Status:** Living process — update when benchmark workflow evolves

---

## The Three Files

| File | Purpose | Owner |
|---|---|---|
| **BENCHMARK-AUTHORITY.md** | Single source of truth for all committed numbers | Results |
| **BENCHMARK-GOVERNANCE.md** | This file — the process and DOS discipline | Process |
| **BENCHMARK-TEMPLATE.md** | Standardized format for new model results | Format |

---

## The DOS-Centric Benchmark Process

### Phase 1: Run the Benchmark

```bash
# Example: RadixAttention benchmark
go run ./cmd/radixbench \
  -dir internal/model/.cache/smollm2-135m \
  -quant \
  -out experiments/radixattention/radixbench-smollm2-135m-q8.json
```

**Output:** Committed JSON artifact with raw measurements.

### Phase 2: Verify with DOS

```bash
# After committing, audit the claim
dos commit_audit <commit-hash>
```

**Expected verdict:** `OK` (diff-witnessed) or `CLAIM_UNWITNESSED` (needs work)

**What this checks:**
- The commit message claims X
- The diff actually touches files related to X
- The change is not empty/self-contradictory

**Note:** `dos_commit_audit` does NOT verify correctness — only that the claim isn't empty.
Run tests for correctness (`go test ./...` or workload-specific).

### Phase 3: Document in Authority

Update `BENCHMARK-AUTHORITY.md` with:

```markdown
| Claim | Number | Model | Baseline | Commit | Artifact |
|---|---|---|---|---|---|
| **Result name** | **X.X×** | Model Y | Baseline Z | `<hash>` | `path/to/artifact.json` |
```

### Phase 4: Ship with Stamp

Commit with Conventional-Commits format + DOS stamp:

```bash
git commit -m "feat(benchmark): [result name]

- Add BENCHMARK-AUTHORITY.md entry: X.X× speedup on Model Y
- Artifact: path/to/artifact.json (committed)
- Verify: dos_commit_audit <hash> → OK

(fak benchmark)"
```

The stamp binds the claim to `dos verify fak benchmark` for future checks.

---

## The Deterministic Checklist

Before a benchmark claim is "shipped," verify:

- [ ] **JSON artifact exists** in repo at committed path
- [ ] **Commit message** describes the result accurately
- [ ] **`dos_commit_audit <hash>`** returns `OK`
- [ ] **BENCHMARK-AUTHORITY.md** updated with entry
- [ ] **Cross-references** added from related docs
- [ ] **Baseline is clear** (what are we comparing against?)
- [ ] **Reproduction command** documented
- [ ] **Tombstoned outdated claims** if replacing old numbers

**No benchmark is shipped until all boxes are checked.**

---

## The Truth Syscalls

DOS provides three syscall-level primitives for benchmark verification:

| Syscall | Use Case | What It Checks |
|---|---|---|
| `dos_commit_audit` | Verify commit matches claim | Diff does the KIND of thing message says |
| `dos_verify` | Verify phase shipped | Git history + stamp grammar shows phase complete |
| `dos_arbitrate` | Check concurrent work safety | File-tree disjointness before running benchmarks |

**All benchmark claims flow through these syscalls.** No trust without witness.

---

## Regime Boundaries — What Each Number Means

| Regime | Measures | Baseline | Example |
|---|---|---|---|---|
| **Raw throughput** | tok/s | None (single-stream) | fak: 7.8 tok/s vs llama.cpp: 127 tok/s |
| **Reuse efficiency** | Cache hit rate | Zero cache | 86.7% hit rate (inside SGLang 50-99% band) |
| **Live speedup** | Wall-clock ratio | Full re-prefill | 4.87× (SmolLM2-135M, agents workload) |
| **Session value-add** | Total work saved | Naive stateless | 11.2–14.5× (vs no KV persistence) |

**Never mix regimes.** A 4.87× live speedup is NOT comparable to an 11.2× session value-add —
they measure different baselines.

---

## The Anti-Inflation Rules

To prevent benchmark inflation:

1. **One primary number per result.** The headline must be a single, defensible number.
2. **Baseline must be stated.** "4.87× speedup" means nothing without "vs full re-prefill."
3. **No cherry-picking.** Report the regime that matters, not the best-looking number.
4. **Tombstone old claims.** When numbers change, old claims are marked ❌, not silently deleted.
5. **Reproduction required.** Every claim must have a command that reproduces it.

---

## Cross-Index: Related Skills

> **📋 Skills Index:** See **[.claude/skills/README.md](../.claude/skills/README.md)** for the complete
> skills catalog with DOS integration details.

| Skill | Relevance to Benchmarks |
|---|---|
| `/release` | Version bump + changelog for result releases |
| `/curate-cluster` | Index + commit gardening for benchmark artifacts |
| `/plan-audit` | Check if benchmark plan tasks shipped |
| `/trajectory-audit` | Cross-session token/cost efficiency analysis |

---

## Example: Full SmolLM2-135M RadixAttention Ship

### What Happened

1. **Run:** `go run ./cmd/radixbench ...` → `radixbench-smollm2-135m-q8.json`
2. **Commit:** `a200c3d` with JSON + doc changes
3. **Audit:** `dos_commit_audit a200c3d` → OK (diff-witnessed)
4. **Document:** Updated BENCHMARK-AUTHORITY.md with 4.87× entry
5. **Cross-ref:** Added links from RADIXATTENTION-RESULTS.md
6. **Ship:** Pushed to `origin/master`

### Traceability Chain

```
Claim: 4.87× live speedup on SmolLM2-135M Q8
  │
  ├─ Commit: a200c3d
  │   └─ dos_commit_audit → OK (diff-witnessed)
  │
  ├─ Artifact: fak/experiments/radixattention/radixbench-smollm2-135m-q8.json
  │   ├─ live_baseline_ms: 240,994
  │   ├─ live_radix_ms: 49,452
  │   └─ live_prefill_speedup: 4.87
  │
  ├─ Authority: fak/BENCHMARK-AUTHORITY.md
  │   └─ Entry in Quick Reference table
  │
  └─ Cross-ref: fak/RADIXATTENTION-RESULTS.md
      └─ Links to authority with headline number
```

Every link in the chain is checkable by another agent.

---

## Next Model: Qwen3.6-27B Checklist

When benchmarking Qwen3.6-27B:

- [ ] Run benchmark: `go run ./cmd/bench-qwen36 ...`
- [ ] Commit JSON artifact
- [ ] Run `dos_commit_audit <hash>` → document result
- [ ] Copy `BENCHMARK-TEMPLATE.md` to `QWEN36-RESULTS.md`
- [ ] Fill all template fields with measured data
- [ ] Add entry to `BENCHMARK-AUTHORITY.md` Quick Reference
- [ ] Add cross-reference from relevant docs
- [ ] Commit with Conventional-Commits + stamp
- [ ] Run `/release` if shipping a versioned result
- [ ] Push to `origin/master`

---

## The DOS Discipline Summary

**No claim without witness.** Every benchmark number must have:

1. **Committed artifact** (JSON in repo)
2. **Git commit** with `dos_commit_audit` verification
3. **Authority entry** in BENCHMARK-AUTHORITY.md
4. **Clear baseline** (what we're comparing against)
5. **Reproduction command** (how to re-run)

**If any of these is missing, the claim is not shipped.**

---

## Troubleshooting

### `dos_commit_audit` returns CLAIM_UNWITNESSED

**Cause:** Commit message claims X but diff doesn't touch X-related files.

**Fix:** Either (a) reword commit to match diff, or (b) add the missing files.

### Authority entry exists but JSON is missing

**Cause:** JSON wasn't committed or path is wrong.

**Fix:** Find the JSON, commit it, update authority entry with correct path.

### Can't reproduce claimed number

**Cause:** Workload params or baseline mismatch.

**Fix:** Check reproduction command matches claimed workload exactly.

---

## Appendix: File Locations

```
fak/
├── BENCHMARK-AUTHORITY.md      # Single source of truth for all numbers
├── BENCHMARK-GOVERNANCE.md      # This file — the process
├── BENCHMARK-TEMPLATE.md        # Format for new model results
├── RADIXATTENTION-RESULTS.md    # RadixAttention detailed results
└── SESSION-VALUE-STACK-RESULTS.md # Session value-add detailed results

docs/benchmark/
├── README.md                    # Cross-machine infrastructure overview
├── QUICKSTART.md                # Get started with multi-node benchmarks
└── CROSS-MACHINE-INFRASTRUCTURE.md # Full design spec and schema

.claude/skills/
└── README.md                    # Skills index with DOS integration

.claude/memory/
└── cross-machine-benchmark-infrastructure.md # Cross-node tracking

fak/experiments/
└── radixattention/
    └── radixbench-smollm2-135m-q8.json  # Example committed artifact
```

---

**Bottom line:** This system makes benchmark claims **verifiable, traceable, and reproducible**.
Every number traces back to a committed artifact verified by DOS syscalls.
