# bench_long_context.py — SGLang HiCache Long Context Benchmark

## Location
`sglang/benchmark/hicache/bench_long_context.py`

## Purpose
Evaluates SGLang's HiCache (hierarchical KV cache: L1 GPU → L2 host RAM → L3 distributed storage) for the **shared prefix** scenario: multiple queries reference the same long document, so the KV cache for that prefix is computed once and reused.

LMSYS blog (Sep 2025) reported up to 6x throughput improvement and 80% TTFT reduction with HiCache.

## Key Parameters

| Parameter | Default | Source | Notes |
|-----------|---------|--------|-------|
| `--num-clients` | 256 | `bench_multiturn.py:31-34` | Number of queries to use from dataset; capped by `len(dataset["queries"])` at `bench_long_context.py:34` |
| `--max-parallel` | Hardcoded 24 | `bench_long_context.py:94` | Max concurrent in-flight requests |
| `--num-rounds` | Hardcoded 1 | `bench_long_context.py:93` | Single round (not multi-turn) |
| `--request-rate` | Swept | `bench_long_context.py:97` | Hardcoded sweep: `[24, 16, 12, 8, 4, 2, 1]` |
| `--dataset-path` | required | CLI | Path to preprocessed JSON with `{"contexts": {...}, "queries": [...]}` schema |
| `--model-path` | meta-llama/Llama-3.1-8B-Instruct | `bench_multiturn.py` | Must match running server |

## How It Works

1. Loads dataset JSON (`bench_long_context.py:33`) with `"contexts"` (long documents) and `"queries"` (each referencing a context ID + question + reference_answer)
2. Creates `min(num_clients, len(queries))` requests by concatenating `contexts[context_id] + question` (lines 38-52)
3. Main loop (line 97) runs 7 iterations at different request rates `[24, 16, 12, 8, 4, 2, 1]`
4. Each iteration: flush cache → sleep 1s → send all requests with Poisson-distributed pacing → collect metrics

### Total requests = `num_clients × 7` (one per rate iteration)

## What `num_clients` Actually Does
- **NOT concurrent connections** (that's `max_parallel`)
- Sets the total number of queries/requests per rate iteration
- Each "client" sends exactly 1 request (`num_rounds=1`)
- With `num_clients=1`: only query index 0 is used; rest of dataset loaded but untouched

## Request Rate Sweep
The `[24, 16, 12, 8, 4, 2, 1]` sweep finds the **saturation point** — the rate at which TTFT spikes. At low rates, requests benefit from cached prefixes; at high rates, eviction pressure degrades performance. With `num_clients=1`, request_rate is irrelevant (no inter-request delay needed).

## Key Metrics (lines 57-64, 78-83)
1. **TTFT** — primary metric; cache hits reduce prefill time
2. **Cached tokens / cache hit rate** — how many tokens served from cache vs recomputed
3. **ITL** (inter-token latency)
4. **Latency** (total request latency)
5. **Prompt length** and **generated length**

## Time Units: Seconds vs Milliseconds

**Critical distinction between benchmark scripts:**

- **`bench_long_context.py` / `bench_multiturn.py`** collect timings via `time.perf_counter()` and store them as **raw seconds** (no conversion). All metrics (`ttft`, `itl`, `latency`) in their output are in **seconds**.

- **`bench_serving.py`** (lines ~346-365) explicitly multiplies every metric by `* 1000` and names fields with `_ms` suffixes:
  ```python
  mean_ttft_ms=np.mean(ttfts or 0) * 1000,
  mean_tpot_ms=np.mean(tpots or 0) * 1000,
  mean_e2e_latency_ms=np.mean(e2e_latencies) * 1000,
  ```

**When comparing results between scripts, multiply bench_long_context values by 1000 to convert to ms.**

## The Wiki Dataset (`loogle_wiki_qa.json`)
- Preprocessed version of [LooGLE](https://huggingface.co/datasets/bigai-nlco/LooGLE) dataset (Wikipedia-sourced docs)
- Average ~24K tokens per context, some >100K
- Reformatted into `{"contexts": {...}, "queries": [...]}` schema
- Multiple queries reference the same context — this creates the shared prefix pattern
- Preprocessing script is NOT in the repo

## Typical Invocation

**Server:**
```bash
python3 -m sglang.launch_server \
    --model-path /DeepSeek-R1/ --tp 8 \
    --enable-hierarchical-cache \
    --hicache-ratio 2 --hicache-io-backend kernel \
    --page-size 64 --context-length 65536
```

**Benchmark:**
```bash
python3 bench_long_context.py \
    --model-path /DeepSeek-R1/ \
    --dataset-path loogle_wiki_qa.json
```

## Why Someone Would Set `num_clients=1`

Legitimate reasons:
1. **Baseline latency** — isolate raw prefill+decode time with zero contention (floor for TTFT)
2. **Smoke test** — verify pipeline works, no OOM, CUDA graphs capture correctly
3. **Profiling** — attach profiler (Pyroscope, nsys) without noise from concurrent requests
4. **L1→L2→L3 tier retrieval testing** — with 1 client, TTFT differences between iterations measure cache tier retrieval latency vs recomputation

### Caveat for tier testing
The flush at `bench_long_context.py:99` calls `/flush_cache`. If this purges **all** tiers (L1+L2+L3), every iteration is a cold recompute and you're NOT testing tier retrieval — just the same cold prefill 7 times. Need to verify whether `/flush_cache` clears only L1 or all levels. May need to modify/remove the flush to properly test hierarchical retrieval.

## Why `num_clients=1` Misses the Point for Prefix Reuse
With 1 query there's only 1 context referenced — zero shared prefix reuse. Need 2+ queries referencing the same context for caching to matter. Default of 256 is designed for this.

## Related Benchmarks in `benchmark/hicache/`

| Script | Scenario | Cache Pattern |
|--------|----------|---------------|
| `bench_long_context.py` | Shared prefix (long docs + questions) | Same document prefix reused across queries |
| `bench_multiturn.py` | Multi-turn conversations | History grows across rounds; round-robin maximizes eviction pressure |
| `bench_mix.py` | Mixed workload simulation | Configurable round distributions, inter-round intervals, realistic session mix |

## Sources
- [LMSYS HiCache Blog Post](https://lmsys.org/blog/2025-09-10-sglang-hicache/)
- [SGLang HiCache Benchmark (GitHub)](https://github.com/sgl-project/sglang/tree/main/benchmark/hicache)
- [HiCache Design Docs](https://docs.sglang.io/advanced_features/hicache_design.html)
- [Mooncake x SGLang HiCache](https://kvcache-ai.github.io/Mooncake/design/hicache-design.html)
- [Strata Paper](https://arxiv.org/html/2508.18572v1)

## See Also
- [HiCache Prefetch & Write Policies](../../docs/hicache-prefetch-write-policies.md) — trade-offs between `best_effort`/`timeout`/`wait_complete` prefetch and `write_through`/`write_through_selective`/`write_back` write policies, with benchmarking recommendations
- [CAMA vs Mooncake: Read/Write Trade-offs](../../docs/cama-vs-mooncake-read-write-tradeoffs.md) — why this benchmark (read-dominated shared prefix) is where CAMA's architecture shines vs Mooncake
- [End-to-End Bottleneck Guide](../../docs/end-to-end-bottleneck-guide.md) — full-stack profiling from SGLang HiCache controller down to the NIC
