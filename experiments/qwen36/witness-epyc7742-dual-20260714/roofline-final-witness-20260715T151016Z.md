# EPYC-7742 decode roofline — FINAL acceptance witness for #4626 (scrubbed)

Repo-persisted, **scrubbed aggregate** transcription of the model-backed roofline
witness that satisfied and **closed #4626** ("C3: EPYC memory-bandwidth roofline
witness — STREAM + decode-bw-utilization %"). It exists so C1/C2/C5 runs
(#4624 / #4625 / #4623) can reference the realized memory wall and
`decode_bw_util_%` **from the repo**, per the issue's approach item 3 ("persist a
dated roofline manifest per box that C1/C2/C5 runs reference"). The earlier
sibling files in this dir (`bandwidth.jsonl`, `results.jsonl`, `README.md`)
captured only the coarse 2026-07-14 go-f64 triad (89.5 GB/s interleave) whose
README flagged the real roofline as future work; this file records the
dependency-free C STREAM Triad run that actually closed the leaf.

**This is a transcription of already-witnessed, already-public scrubbed evidence
— not a fresh hardware run, and not a 10 tok/s claim.** Raw host/control detail
stays in the private evidence archive; only scrubbed aggregate numbers and their
digests are reproduced here. The 10 tok/s program target remains open under epic
#4623 (deliberately not closed by this leaf).

## Provenance (independently checkable)

- Producer commit: `4f764a8889a6` (`cmd/q4kdiag -require-roofline` harness) —
  verified ancestor of `origin/main`.
- Witness revision: `df6ae047e442ccc123df5c48b8a7d0435a59252d` — verified ancestor
  of `origin/main`.
- Box: dual AMD EPYC-7742 (256 threads, 8 NUMA nodes, ~1 TB RAM, no GPU), CPU server.
- Model: resident `Qwen3.6-27B-Q4_K_M.gguf` (exact pinned artifact).
- Dated producer manifest: schema `benchmark/run-manifest.v1`, run timestamp
  `20260715T151016Z`.
- Scrubbed roofline artifact attached to the 24h resource campaign #4367.

## Memory-bandwidth roofline — dependency-free C STREAM Triad (same-run)

Per-node observed GB/s (n0..n7):
`26.96, 26.90, 26.85, 26.91, 26.75, 26.94, 26.96, 26.78`

- **Aggregate interleaved peak: 98.87 GB/s.**
- Guard honored: a zero-peak STREAM measurement fails closed (exit 2) before model
  loading rather than reporting a bogus 100% util.

## Model-backed decode `decode_bw_util_%` (one-step bounded recurrent witness)

| cell | tok/s | bytes/token | achieved GB/s | `decode_bw_util_%` | first token |
|---|---|---|---|---|---|
| f32 recurrent  | 0.8990 | 19,227,217,920 | 17.2852 | **17.48** | 248068 |
| int8 recurrent | 0.2620 | 19,227,217,920 |  5.0373 | **5.09**  | 248068 |

- `decode_bw_util_% = (bytes/token × tok/s) ÷ STREAM_peak` against the same-run
  98.87 GB/s wall.
- First-token identity `248068` agreed across both cells (correctness held).
- One-step bounded roofline witness; explicitly **not** a 10 tok/s claim.

## Evidence digests (recompute to check against the #4626 closing comment)

Independent readback at `2026-07-15T15:15:21Z`:

- manifest  SHA-256 `a2a4253e044dcf046c7f0d2a3aa2dd69217e439a9d714808b52e660e7c0f986a`
- results   SHA-256 `7ff56954d2800fec766ddd2ccb1b4adc09b39712b076f2042d90f887d07043dd`
- run log   SHA-256 `218926139116323283f030e8c50e43670b0c216ee0cb26e8e530d3429f0e6119`
- plan      SHA-256 `4c36135d3fcc0e1aa72f39a601677f2a20ed67fc4152eaca14f450c49cfbacf0`

Independent re-verification (2026-07-15T15:51Z, corroboration only, no new run)
recomputed these 4/4 **PASS, 0 mismatches**; receipt `fak.reverify.receipt.v1`
SHA-256 `b3a675482298f97df6edb3bf4f1413b3f5b2b33ee13bf0a3b18317ee8f52ddd6`.

## Status

- #4626 — CLOSED (2026-07-15T15:18:47Z), acceptance witnessed + independently
  re-verified; this file persists the closing witness in-tree for reference.
- Keep-open siblings confirmed OPEN: #4379, #4738, #4854.
- Program target (sustained 10 tok/s) remains OPEN under epic #4623.
