# ARM64 q-kernel score matrix

This page is the operator contract for the score-only Mac benchmark artifacts under
`experiments/benchmark/runs/by-machine/node-macos-a/*/score.json`.

Run the audit:

```bash
go run ./cmd/benchscore -root experiments/benchmark/runs/by-machine/node-macos-a
```

For machine-readable output:

```bash
go run ./cmd/benchscore -root experiments/benchmark/runs/by-machine/node-macos-a -json
```

## What the scorer proves

- Every parsed artifact has `schema: fak.arm64-qkernel-score.v1`.
- Recorded speedups are recomputed from the raw tok/s or ms fields in the same `score.json`.
- `accepted_*` interpretation statuses require `verification.status: pass`.
- Decode follow-up rows are compared against the canonical Q8 decode baseline carried in the score file.
- Exploratory rows, such as the 14B Q4_K load/decode probe, stay in the matrix without receiving an apples-to-apples speedup.

## Current Mac status

The June 26, 2026 Mac score set validates as:

- Accepted: Metal prefill P256 on Qwen2.5-1.5B, and prompt-heavy end-to-end P512+D32 / P1024+D32 on the same HF snapshot.
- Negative: Q8 batch decode, f32 batch decode, and Q8 GGUF lean decode do not clear the tracked 2x/3x decode target.
- Exploratory: 14B Q4_K GGUF split-shard decode is recorded as a functional resident-path probe, not a baseline claim.

## Adding more models

Create one run directory per model/source/probe, then add a `score.json` with:

- `machine`, `captured_at`, `benchmark_clone.rev`, and any `benchmark_clone.local_patches`.
- `model.name`, `model.source_kind`, and `model.precision`.
- Raw command flags and environment, especially `-hf`, `-gguf`, `-quant`, `-metal`, `-prefill-sizes`, `-decode-steps`, `-reps`, `FAK_WORKERS`, and `FAK_BUDGET`.
- Raw timings under `results`, `arms`, `probes`, or `end_to_end`, plus the baseline values used for any speedup.
- `verification.status: pass` for any accepted row.

Useful next model rows:

- SmolLM2-135M HF: small-model sanity row for the same Metal prefill and batch decode shape.
- Qwen2.5-0.5B HF/GGUF: lower-memory dense Qwen row to test whether decode flatness is model-size independent.
- Qwen2.5-7B HF/GGUF: larger dense Qwen row to extend the model ladder beyond 1.5B.
- One non-Qwen dense HF snapshot already supported by `internal/model`: checks that the `-hf` path and score schema are not Qwen-only.

Do not mark a row accepted just because raw speed clears a target. Accepted rows need both the speedup math and a passing verification block; otherwise leave the interpretation as exploratory or negative.
