# Micro-context S5a: 1,000 real-model contexts on controlled CUDA

**Status:** witnessed on 2026-08-06. This is the small-first ramp in #5820; it does not close the 10,000-context S5 soak in #5792.

The run used one bounded client worker against the native controlled-CUDA `fak serve` path on the sanctioned L4 node. The public ledger removes the endpoint address. The fixed corpus and one worker intentionally measure the stable serial envelope after multiworker attempts overloaded or timed out.

## Captured result

| Measure | Observed |
|---|---:|
| Submitted / completed / failed | 1,000 / 1,000 / 0 |
| Physical workers / peak client in flight | 1 / 1 |
| Wall time | 1,045.554 s |
| TTFT p50 / p95 | 1.058 / 1.144 s |
| Prompt / completion tokens | 703,890 / 8,000 |
| Usage responses | 1,000 |
| Usage-derived prompt / decode rate | 673.22 / 7.65 tok/s |
| Nonempty, usage-bearing results / wall-second | 0.956 |
| Client peak RSS | 16.5 MiB |
| Server sampled peak RSS | 5.22 GiB |
| Server sampled peak Go heap | 4.84 GiB |
| Resource samples | 69 over 1,020 s |

Verifier:

```powershell
go run ./cmd/microcontextdemo -verify experiments/microcontext/s5-gcp-1000-cuda-workers1-pass-2026-08-06.json
```

The verifier admits only the declared 100/1,000/10,000 scale ladder, reconciles submitted = completed = turns = usage responses, requires zero failures and bounded physical workers, and requires resource/claim-boundary fields at 1,000 and 10,000 contexts.

## Roofline and queue boundary

The endpoint exposed host process and Go-runtime memory through `/proc` and `/debug/vars`, but did **not** expose resident KV slots, device-memory occupancy, prefix-cache hits, or queue age. Therefore the ledger reports the host/server envelope and explicitly records missing KV-capacity and queue-age evidence; it makes no GPU-KV capacity or cache-benefit claim. Endpoint max in-flight was two because the one inference request and independent metrics probe overlapped.

The completion verifier is deliberately narrow: every stream returned nonempty text and provider usage. That is a useful-result transport contract, not semantic task-quality evidence. #5794 owns the independently authored quality ledger.

## Operational recovery

The stale CUDA process listed models but its completions hung, and its executable had already been unlinked. The exact executable was recovered from `/proc/<pid>/exe` before stopping the stale process. Because `/tmp` and `/var` were mounted `noexec`, the recovered binary was launched only after making the sanctioned execution mount executable. This is a node-recovery fact, not a model or backend performance claim; public evidence contains no private address.

## Claim boundary

This rung establishes that 1,000 logical contexts can retire exactly over one stable physical model slot with bounded client memory. Prompt/decode rates are observed provider usage divided by wall time, not server-internal kernel rates. It does not establish a same-model backend speedup, semantic useful-task throughput, prefix-cache benefit, resident GPU-KV capacity, or multi-user fairness.
