# Mac keepalive qualification — 2026-10-03

The finite Metal keepalive reaches the production Q4 session route and passes physical output-parity, lifecycle, and reachability tests on this Mac15,7 / Apple M3 Pro / 36 GiB host. It remains **off by default**. The complete sanitized native reports, source and binary hashes, independent review, and test logs are in [receipt.json](receipt.json).

The final candidate uses 2,048 atomic polling iterations per GPU command buffer and at most two buffers in flight. Both ON runs submitted work without a reported failure: 22,595 / 22,936 buffers, with maximum completed GPU durations 2.490 / 3.017 ms. Both OFF runs submitted zero buffers. The earlier 200,000-iteration candidate triggered `MTLCommandBufferErrorDomain` code 1, `Impacting Interactivity`; those runs are retained as rejected evidence and excluded from the final comparison.

The matched candidate-binary OFF/ON/ON/OFF comparison used Qwen3.5-0.8B-Q4_K_M, prefill P=4, decode prompt P=4, T=32, five repetitions per process, default 12 CPU workers and six Q8 decode workers, and the native exclusive GPU lease. The execution regime was legacy Metal Q4_K/Q8 hybrid, including CPU Q8 work.

| Arm | Decode tok/s | Prefill tok/s | Keepalive failed |
| --- | ---: | ---: | --- |
| OFF 1 | 8.523 | 9.318 | false |
| ON 2 | 19.814 | 7.482 | false |
| ON 3 | 19.531 | 10.780 | false |
| OFF 4 | 10.197 | 10.434 | false |

The median observed decode rate was 9.360 tok/s OFF and 19.672 tok/s ON, a 2.102× ratio in this cohort. Shared-host load averages were 26.85–28.51 on 12 logical cores, with peer tests and background work active. This is a point-in-time comparison, not a general throughput guarantee or an isolated causal estimate. Prefill did not show a consistent gain.

The physical idle-gap microbenchmark checked bit-identical GEMV output on every iteration. Median host GEMV latency fell from 357.229 to 201.196 µs after a requested 200 µs gap, and from 570.054 to 327.867 µs after a requested 5 ms gap. Actual recorded sleeps were approximately 250–262 µs and 5.821–5.966 ms; the sleep is excluded from the measured operation latency. These three-repetition microbenchmarks used two Go workers and do not substitute for the normal-core model comparison.

Source identity at measurement is pinned by base revision `a3caf52953c5b771cd2653bff30c0e9d7f5e27be`, exact changed-file hashes, and candidate binary SHA-256 `ac2e1cae50fffe21cd5bfd4acfa10737b282cded7debf4945eeb4bb06eee80dc`. The binary was built from the recorded working tree; its printed HEAD alone does not identify the candidate. No 27B-model, energy, thermal-cost, or default-enable qualification is claimed.

The shipped source is commit `6d1a868f2eff1124a396579da5e3447275c66723`. Every measured changed-file SHA-256 matches that commit; the receipt records the canonical aggregate source hash `e831feb1295a8cff46c6da6021f2b0d5c31cd4f8f749801988441c9060798493`.
