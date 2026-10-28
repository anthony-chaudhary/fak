# GSQ-RCO follow-ons — public campaign index (lane `gsq-rco-followons`)

Source convention: the upstream release
`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` (HF, pinned revision
`ed59f92082b1e93c0e96d60a8b11aab089b52f09`; GSQ arXiv 2604.18556, RCO arXiv
2605.00649) ships per-tensor allocation sidecars and recommends SSD-mmap
residency for its fixed n-gram shard. This campaign tracks the public-repo
follow-ons: adopt the sidecar convention as a fail-closed audit probe, and
turn the residency recommendation into a measured, class-based design input.

Set total estimate: 4 points (2 tickets x 2 points, ESTIMATED).

| ID | Title | Status | Unblock event / gate | Estimate |
| --- | --- | --- | --- | --- |
| GR-02 | [Read and validate `.rco-allocation.txt` sidecars against the parsed tensor inventory](GR-02-allocation-sidecar-reader.md) | open ([#14093](https://github.com/anthony-chaudhary/fak/issues/14093)) | none (host-testable now) | 2 |
| GR-04 | [Tensor-class residency mapping with measured NVMe row-read latency floor](GR-04-tensor-class-residency-mapping.md) | open ([#14094](https://github.com/anthony-chaudhary/fak/issues/14094)) | none (design + host measurement now); planner tiering is a named follow-on | 2 |

Notes:

- All upstream perf/quality numbers cited in this campaign are [source-only]
  (upstream artifacts/cards; no local reproduction yet).
- Consuming lane for both tickets: qwen38-flash-next bringup (not started
  yet). Runtime-side mixed-precision scheduling lives in epic #13097; out of
  scope here.
