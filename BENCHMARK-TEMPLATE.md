# Benchmark Results Template — [Model Name]

> **📊 AUTHORITY:** This document's benchmark results are indexed in **[BENCHMARK-AUTHORITY.md](BENCHMARK-AUTHORITY.md)**,
> the single source of truth for all committed performance claims.

**Date:** YYYY-MM-DD
**Commit:** `<hash>`
**Artifact schema:** `benchmark_artifact.schema = "fak-benchmark-artifact/1"`
**Run ID:** `<benchmark_artifact.run_id>`
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
- **Model snapshot:** [HF commit / GGUF sha256 / quantization hash]
- **Hardware:** [CPU/GPU, relevant specs]
- **Harness:** [harness name + harness version]
- **Dependencies/build:** [Go version, CUDA/Metal version if used, build tags/flags]
- **Config hash:** [`benchmark_artifact.config.hash`]

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
- `benchmark_artifact.witness.dos_verify_result`: `OK`
- Witness test: `path/to/test` (`benchmark_artifact.witness.test_path`)
- Invalidation state: `benchmark_artifact.invalidated.is_invalid = false`
- Reproduction command:

```bash
[command to reproduce]
```

## Scientific Record

Every committed JSON artifact for this result must carry a top-level
`benchmark_artifact` object. The shared Go harness (`benchcli.MarshalReport` /
`benchcli.WriteReport`) stamps this automatically; external harnesses must emit
the same fields.

Required fields:

- **Version tracking:** `fak_commit` (full commit), `fak_version`, `harness_version`, `model.{name,precision,source_commit,source_url,hash}`, `dependency_versions`, `build.tags`
- **Invalidation:** `invalidated.{is_invalid,reason,replacement_run_id}`; mark superseded results manually and run automatic invalidation when commits touch `internal/model/`, `internal/compute/`, `internal/radixkv/`, benchmark harness code, model identity, or benchmark config
- **Lineage:** `lineage.source_artifact`, `witness.{test_path,dos_verify_result,dos_commit_audit,reproduction_command}`, and the baseline recorded in `results.baseline` or `lineage.baseline`
- **Config/results:** `config.hash`, `config.parameters`, `results.{metrics,units,baseline}`

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
