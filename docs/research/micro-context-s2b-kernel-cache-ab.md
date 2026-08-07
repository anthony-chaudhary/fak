# Micro-context S2b: controlled in-kernel prefix-cache A/B

**Maturity:** observed, fixture-scoped. **Captured:** 2026-08-07. **Issues:** #5817; partial witness for #5804.

## Question and controls

Does one immutable agent base improve useful aggregate service when fak controls and observes the model cache? The endpoint was fak's OpenAI-compatible in-kernel engine with `FAK_INKERNEL_RADIX=on`, Qwen2.5-0.5B Instruct Q8_0, on the sanctioned GCP L4 node. This particular endpoint executed the CPU/reference Q8 AVX-512 path; the L4 hosted the controlled node but this result is **not** a CUDA throughput result.

Both arms used one physical worker, eight sequential requests, `max_tokens=1`, and identical model/config. Each arm started a fresh fak process (deterministic cold cache). The shared arm preserved one byte-identical 701-token prompt base. The unique arm inserted `Unique-{task:08x}` at the first semantic bytes after the fixed chat template, rather than changing a suffix that could legitimately retain common-prefix reuse. Within each arm the process stayed alive, providing the warm-preserve control. Base SHA-256 is recorded in the artifact.

## Observed result

| Arm | Wall | Contexts/s | Prompt tokens | Kernel/usage cached | TTFT p50 |
|---|---:|---:|---:|---:|---:|
| Tuned sequential unique | 67.661 s | 0.1182 | 5,760 | 85 | 8.100 s |
| Shared immutable base | 31.313 s | 0.2555 | 5,608 | 4,859 | 3.329 s |

The shared arm was **2.160813x** the useful context throughput of the tuned unique sequential arm at this scope. Its first cold request reported one cached token; the seven warm requests reported 4,858 cached tokens out of 4,907 prompt tokens (99.0014%). In both arms, response `usage.prompt_tokens_details.cached_tokens` exactly equaled the endpoint-native `fak_gateway_kv_prefix_reused_tokens_total` delta. Prompt and turn totals also reconcile exactly.

This is a cache-benefit result, not an orchestration-concurrency result: both arms had one worker. `/v1/batches` was unsupported (HTTP 404), so no provider-native batch result is blended into the comparison. The prior S2 `not-yet` cache verdict is superseded only for this controlled fixture.

## Claim grade and boundaries

`fak claim-check` returned `net-true` for the statement “Shared immutable agent bases delivered 2.16x context throughput versus first-byte-unique tuned sequential prompts,” against the fresh-process, one-worker unique arm. The net cost of retaining the shared radix entry is included in the measured wall time. Scope is eight one-token Qwen2.5-0.5B Q8 requests on fak's CPU-reference in-kernel path.

Excluded: CUDA throughput, software concurrency, provider billing cache, longer decode, memory/KV capacity scaling, queue-tax scaling, eviction behavior, and cross-model generalization. Those exclusions matter: this witness establishes realized local prefix reuse and its fixture-level service benefit; it does not complete #5804's concurrent arm, RAM/KV, queue, eviction/miss-reason, or full prefill/decode accounting.

## Reproduce the witness check

```powershell
go test ./cmd/microcontextdemo
go run ./cmd/microcontextdemo -verify-kernel-prefix-ab experiments/microcontext/s2b-gcp-inkernel-prefix-ab-pass-2026-08-07.json
```

The machine ledger contains request rows, cold/warm splits, process controls, provenance, exact endpoint-counter reconciliation, and explicit excluded claims. It contains no endpoint address or credential.
