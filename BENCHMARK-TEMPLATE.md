# Benchmark Results Template — [Model Name]

> **📊 AUTHORITY:** This document's benchmark results are indexed in **[BENCHMARK-AUTHORITY.md](BENCHMARK-AUTHORITY.md)**,
> the single source of truth for all committed performance claims.

**Date:** YYYY-MM-DD
**Commit:** `<hash>`
**DOS Verify:** `dos_commit_audit <hash>` → **OK** (diff-witnessed)

## Measurement Status

- Dataset: [source, size, version/hash]
- Model: [name, version, date; use "none" only for model-free deterministic geometry]
- Runs: [n iterations, dates; use "n=0 live runs" only for theory/model-only]
- Artifacts: [links to committed JSON, logs, traces]
- Status: THEORETICAL | MEASURED | VERIFIED

## Summary

| Claim | Number | Baseline | Context |
|---|---|---|---|
| **Primary result** | **X.X×** | [baseline description] | [workload description] |

## What This Measures

[Brief description of what the benchmark tests and why it matters]

## Workload

- **Shape:** [agents, turns, tokens, etc.]
- **Model:** [model name, precision, size]
- **Hardware:** [CPU/GPU, relevant specs]

## Results

### [Result Category 1]

| Metric | Baseline | Optimized | Speedup |
|---|---|---|---|---|
| Wall-clock | XXX ms | YYY ms | **Z.Z×** |
| [Other metric] | AAA | BBB | C.C× |

### [Result Category 2]

[Additional result tables as needed]

## Verification

- Committed artifact: `path/to/artifact.json`
- `dos_commit_audit <commit>` → **OK**
- Reproduction command:

```bash
[command to reproduce]
```

## Cross-References

- Related: [other benchmark documents]
- Baseline comparison: [which SOTA paper/method this compares against]

## Discussion

[Key insights, regime differences, caveats]

---

## Template Usage Instructions

When benchmarking a new model:

1. **Copy this template** to a new file named appropriately (e.g., `MODELNAME-RESULTS.md`)
2. **Fill all placeholders** with actual measured data
3. **Verify with DOS**: Run `dos_commit_audit <hash>` and record result
4. **Update BENCHMARK-AUTHORITY.md**: Add an entry to the authority table
5. **Commit**: Ship with a Conventional-Commits message and proper stamp

**Required fields for every benchmark:**
- Date
- Commit hash
- DOS verification status
- Measurement Status block (Dataset, Model, Runs, Artifacts, Status)
- Primary result number with baseline
- Reproduction command
- Committed artifact path

**No benchmark is considered "shipped" until all above are present.**
