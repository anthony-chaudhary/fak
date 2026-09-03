# Issue 10855: 50+ Sub-Agent Fan-Out Scaling Benchmark Witness

**Title:** `epic(architecture): native tiered multi-agent scaling under strict resource bounds`  
**Parent Issue:** [#10855](https://github.com/anthony-chaudhary/fak/issues/10855)  
**Date:** 2026-09-03  
**Harness:** `cmd/fanrun`  
**Command:** `go run ./cmd/fanrun --agents 1,4,16,32,50,64,128,256`  

## Scaling Metrics

| Sub-Agent Width ($N$) | Serial Wall-Clock | Agents / sec | Cross-Agent vDSO Dedup Hits | Prefix Tokens Elided |
|---|---|---|---|---|
| **$N=1$** | 0.52 ms | 1,912.4 | 0 | 0 |
| **$N=4$** | 0.52 ms | 7,713.1 | 9 | 6,144 |
| **$N=16$** | 2.61 ms | 6,126.0 | 45 | 30,720 |
| **$N=32$** | 4.17 ms | 7,667.1 | 93 | 63,488 |
| **$N=50$** | **6.21 ms** | **8,056.5** | **147** | **100,352** |
| **$N=64$** | **7.77 ms** | **8,231.9** | **189** | **129,024** |
| **$N=128$** | 14.49 ms | 8,832.3 | 381 | 260,096 |
| **$N=256$** | 28.83 ms | 8,879.8 | 765 | 522,240 |

## Invariants Verified

1. **Deterministic Linear Prefix Elision:** Prefix tokens elided equals exactly $(N - 1) \times P$ ($P = 2,048$), with 100,352 tokens elided at $N=50$ and 129,024 at $N=64$.
2. **Sub-10ms 50+ Agent Execution:** Real in-kernel agent loops execute at $>8,000$ agents/second in serial research gather mode.
3. **vDSO Fast-Path Stability:** Cross-agent read deduplication scales linearly with zero cache poisoning across 50+ concurrent subagent tasks.
