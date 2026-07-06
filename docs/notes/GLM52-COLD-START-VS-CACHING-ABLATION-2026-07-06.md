# GLM-5.2 first-request cold tax vs. KV-prefix caching — an ablation, 2026-07-06

**Question.** GLM-5.2 on the datacenter box shows cold turns of **~511.3s** and **~501.2s**,
then warm repeats of **~0.62s** and **~1.80s**, with `cache_bit=true` witnessed on *both*
the cold and the warm turns. If KV reuse is witnessed on the cold turn too, why is it ~500s?

**Answer.** First-request latency and KV-prefix cache value are **two orthogonal axes**. The
~500s is one-time backend **warmup**, not a KV-prefix cache miss — and `cache_bit=true` on the
cold turn is an *aggregate-scope* artifact, not evidence the cold turn was cache-accelerated.

## The two axes

| Axis | What it is | When paid | What it costs | The signal |
|---|---|---|---|---|
| **A — process/server cold start** | weight load into VRAM + CUDA graph capture + DeepGEMM/torch JIT kernel compile + KV pool alloc | **once per server process** (and today, per boot) | ~500s | *not* currently emitted |
| **B — KV-prefix cache (cold vs warm prefix)** | first sight of a system+tools+repo prefix has nothing cached → full prefill; repeat sight → RadixAttention serves it | per distinct prefix | prefill of the prefix; sub-second once warm | `kv_prefix.reused_tokens` (WITNESSED) |

The ~500s is axis A. `cache_bit` measures axis B.

## The ablation — hold the prefix constant, vary only cold/warm

Use the `"say pong"` probe (11 input tokens) so KV-prefix caching can save ≤11 tokens of
prefill — negligible. Any cold→warm delta on this fixed tiny prefix is therefore **warmup**,
not caching. Git-tracked witnesses, `experiments/agent-live/gcp-glm-night2-20260706T133856Z/`
(GCP H100 Mega, corroborating the GPU server mechanism):

| Run | Prefix | Cold/warm | Wall |
|---|---|---|---|
| `direct-chat.error.json` | 11 tok `say pong` | **cold (1st)** | canceled at **180024ms** (180s HttpClient cap) |
| `sglang-direct-chat.json` | 11 tok `say pong` | warm | **1032ms** |
| `dogfood-claude-glm52-sglang-win.json` | Claude turn | warm | **1172ms** |

Identical 11-token prefix ⇒ **≥99% of the cold→warm delta is warmup, not KV-prefix cache
value.** The warm 1.0s pong independently shows steady-state decode is fast, so the ~500s is
amortizable one-time cost, not a recurring per-turn price.

The backend warmup work is visible in the bring-up log
(`GLM52-GCP-DOGFOOD-BRINGUP-2026-07-05.md`, steps 14–16):

```
Load weight begin. avail mem=77.33 GB
Capture target verify CUDA graph begin...
Entering DeepGEMM JIT Pre-Compile session
Try DeepGEMM JIT Compiling for <GEMM_NT_F8F8BF16>
```

## Why `cache_bit=true` on the cold turn is a trap

`cachewitness.CacheBit()` is **aggregate-run scope**, not per-turn attribution
(`internal/cachewitness/cachewitness.go`):

```go
const CacheBitScopeAggregateRun = "aggregate-run-kv-prefix-reuse"
// ... "not a per-turn attribution witness."
func (r Record) CacheBit() bool { return r.KVPrefix.ReusedTokens > 0 }
```

So `cache_bit=true` means the cache engaged *somewhere in the window* (the later warm turns) —
**not** that the ~500s cold turn was cache-accelerated. Presenting the cold turn's latency next
to an unqualified `cache_bit=true` is exactly the WITNESSED/OBSERVED conflation the
`cachewitness` package exists to prevent.

## Follow-up tickets (improve the first request)

- **#3051** — `serve`: gate readiness on a completed warmup inference (not port-bind), so the
  operator's first *real* turn is warm; launcher sends the warmup turn; raise the cold timeout.
- **#3052** — `serve/infra`: persist DeepGEMM/torch.compile/CUDA-graph caches across restarts so
  the cold tax is paid once per (model, quant, arch), not per boot.
- **#3053** — `cachewitness`: emit the first-request warmup tax as its own signal, de-conflated
  from aggregate `cache_bit`, so a cold turn is never read as cache-accelerated.

## Caveat on provenance

The **511.3s / 501.2s** cold figures are the operator's GPU server (sm_80) observation; the ablation
witnesses above are the git-tracked GCP H100 Mega (sm_90) night2 run. They are used as
*mechanism proof* (fixed-prefix cold≫warm ⇒ warmup), consistent across both boxes. The GCP cold
pong was cut off at the 180s client cap, so its true cold time is only bounded ≥180s, not
measured — itself a bug folded into #3051.
