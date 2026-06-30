# BENCHMARK GOVERNANCE — DOS-Centric Benchmark Claims System

> **One place for all benchmark claims.** This document defines the deterministic, DOS-centric
> process for creating, verifying, and publishing benchmark results. Every claim traces back
> to committed artifacts with `dos_commit_audit` verification.

**Last updated:** 2026-06-30
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
- [ ] **Measurement Status block** is present in the benchmark doc
- [ ] **Reproduction command** documented
- [ ] **Tombstoned outdated claims** if replacing old numbers

**No benchmark is shipped until all boxes are checked.**

---

## Measurement Status Policy (#72)

Every benchmark document that publishes or repeats a benchmark number must include
this block near the top, before any result table:

```markdown
## Measurement Status

- Dataset: [source, size, version/hash]
- Model: [name/version/date, or none for model-free deterministic geometry]
- Runs: [n iterations and dates, or n=0 live runs for theory/model-only]
- Artifacts: [committed JSON/log/trace paths]
- Status: THEORETICAL | MEASURED | VERIFIED
```

Definitions:

- **THEORETICAL** means a formula, deterministic geometry model, simulation, or
  projection. It can document a floor or plan, but it is not a benchmark result
  headline and must be labeled in every table or paragraph that cites it.
- **MEASURED** means real data plus real execution produced the recorded number.
  Token counts, timings, and sample size come from logs or artifacts, not assumed
  geometry.
- **VERIFIED** means MEASURED plus an independent reproduction or external
  grader/witness.

The repo also uses finer provenance words such as `MODELED`, `SIMULATED`,
`OBSERVED`, and `WITNESSED`. Those may appear beside the status, but they do not
replace the status block. A theoretical/model-only number cannot be promoted in
copy as "cheaper", "faster", "win", "SOTA", or "measured" unless the same sentence
names it as theoretical/model-only and states the missing measurement.

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
| **Session value-add** | Total work saved | Naive stateless | 5.3–7.4× (SmolLM2-135M, re-measured — see [BENCHMARK-AUTHORITY.md F1](BENCHMARK-AUTHORITY.md#f1--tombstone-note-2026-06-19-governance-rule-4)) |

**Never mix regimes.** A 4.87× live speedup is NOT comparable to a 5.3× session value-add —
they measure different baselines.

---

## Within-Run Ratios — Single-Box Discipline

**Definition:** A *within-run ratio* is a speedup or comparison measured during a single benchmark execution on the same machine (same hardware, same configuration, same workload). This eliminates cross-run variability (machine state, OS scheduling, thermal throttling) and makes the ratio robust even when absolute wall-clocks are hardware-dependent.

**What this means:**
- Deterministic metrics (token counts, cache hit rates, cell counts) are **hardware-independent** — they reproduce exactly on any machine because they count discrete events.
- Live wall-clock ratios are **single-box** — they are authoritative *as within-run ratios* because the baseline and optimized arms run back-to-back on the same box under the same conditions.
- Within-run ratios may vary across hardware (e.g., 4.87× on Mac M3, 2.60× on x86_64 for the same 135M RadixAttention benchmark), but the *trend* is hardware-independent (climbs toward the 7.50× token-speedup ceiling as model size grows).

**Governance rule:** When citing wall-clock ratios, always specify whether they are:
1. **Within-run** (same box, same run — authoritative for the trend)
2. **Cross-run** (different runs, possibly different boxes — must disclose hardware and conditions)

---

## The Anti-Inflation Rules

To prevent benchmark inflation:

1. **One primary number per result.** The headline must be a single, defensible number.
2. **Baseline must be stated.** "4.87× speedup" means nothing without "vs full re-prefill."
3. **No cherry-picking.** Report the regime that matters, not the best-looking number.
4. **Tombstone old claims.** When numbers change, old claims are marked ❌, not silently deleted.
5. **Reproduction required.** Every claim must have a command that reproduces it.

---

## The Industry (Competitive) Scorecard — fak vs SOTA

The authority records *fak's own* numbers; the **industry scorecard** is the
referee that turns them into an honest competitive position. It is the
outward-facing member of the scorecard family (alongside the inward
repo-hygiene / code-quality / doc-appeal sticks), and it answers the one question
the authority's tables don't: *where does fak stand against the field — parity
with vLLM, true gain over the next-best alternative, and where does it lose?*

It is **industry-first**: the source of truth is a researched taxonomy of the
dimensions the field competes on, not the handful fak happened to measure — so the
scorecard can never quietly omit the regimes where fak has no number. It is
**modular**: a directory of small JSON files, one concern per file.

| Piece | What it is |
|---|---|
| `tools/industry_scorecard.data/_taxonomy.json` | The industry-first dimension catalog: every dimension the LLM-serving / agent-infra field competes on (throughput, KV/prefix reuse, quant, decoding, distributed, models, agent-fleet, cost, ops, security), each with the current SOTA bar and a **dated source**. |
| `tools/industry_scorecard.data/_competitors.json` | The SOTA-system registry (vLLM, SGLang, TensorRT-LLM, llama.cpp, …) with `last_reviewed` dates. |
| `tools/industry_scorecard.data/rows-*.json` | fak's honest position on each dimension, grouped by category — mostly named `no-claim` gaps, with the measured leads / parities / losses traced to a commit/artifact. |
| `tools/industry_scorecard.py` | The deterministic grader. Folds the honesty KPIs into **parity-debt** (of the rows that exist) and the taxonomy positioning into **coverage** (of the field), blended into the composite score. Also `--gaps` (coverage backlog) and `--stale` (industry-drift backlog). |
| `docs/industry-scorecard/` | The generated modular doc folder (`--markdown-dir`): an index + one page per group + the dimension catalog + the update process. Indexed from `INDEX.md`. Never hand-edited. |
| `/industry-score` skill | The **update process** on two cadences: as the industry moves (new dimension → coverage drops; stale SOTA bar → `--stale`) and as fak moves (a benchmark turns a `no-claim` into a measured row). |

**The doctrine it enforces mechanically** (the same anti-inflation rules above, as
checks): every win is vs a tuned/SOTA baseline, **never naive** (`baseline_sota`);
every verdict matches its ratio, so a `lead` can't be claimed while trailing
(`verdict_consistency`); a correctness oracle can only be parity and a theoretical
ceiling can never be led; every contracted regime is covered **including its losses**
(`axis_coverage` requires a `trails` row); every fak number traces to a
commit/artifact/authority doc (`fak_traced`) and every competitor number is
sourced (`competitor_sourced`); a non-comparable comparison is disclosed
(`apples_disclosed`). Two numbers are driven: **parity-debt to zero** (every claimed
comparison honest) and **coverage to 100%** (every industry dimension at least
positioned) — and both stay there as new benchmarks land and the field moves.

**How it relates to this process:** a new benchmark flows through Phases 1–4
above into `BENCHMARK-AUTHORITY.md`; the `/industry-score` pass then **records**
it as a competitive row (a win, a parity, or an honest loss) — it never measures
or invents a number. The authority is the *numbers*, HERO is the *pitch*, the
industry scorecard is the *referee*.

## Cross-Index: Related Skills

> **📋 Skills Index:** the `.claude/skills/README.md` catalog (agent-side tooling — not
> published) holds the complete skills list with DOS integration details.

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
