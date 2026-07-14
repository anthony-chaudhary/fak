# Issue #1012 — fak in-kernel KV-prefix cache-bite witness (live GLM-5.2)

- Generated: `2026-07-13`
- Verdict: **`WITNESSED`** — `CACHE_BIT=YES`
- Host: `lab-dgx2` (8× A100-40GB, all idle; 1 TB host RAM)
- Sidecar: `experiments/agent-live/issue-1012-kvprefix-cachebite-glm52-dgx-20260713.json`

## What this proves

fak's **own** in-kernel KV-prefix cache (RadixAttention) biting on a **live GLM-5.2 Q4 in-kernel gateway** — the top open, previously-PENDING #1012 datum. A bounded ~140-token solved-ticket prefix is primed on turn 1, then replayed on turn 2 under one fixed `X-Trace-Id`; turn 2's prefill matches the cached prefix and serves **140 of its 160 prompt tokens from cache**, prefilling only the 20 new suffix tokens.

This is distinct from provider prompt caching: `fak_gateway_inference_cached_prompt_tokens_total` (the provider `cache_read`) stays **0** throughout, so the 140 reused tokens are entirely fak-owned.

## Gateway (real in-kernel GLM-5.2, not the mock)

| field | value |
|---|---|
| `/healthz` planner | `inkernel` (real `InKernelPlanner`, not the scripted mock) |
| served model | GLM-5.2 Q4_K_M (`/mnt/glm/glm52-q4/GLM-5.2-Q4_K_M-00001-of-00008.gguf`) |
| serve flags | `--gguf … --backend cuda --cpu-offload-experts --addr :8090` |
| resident | 437,273.85 MiB (Q4_K experts on host RAM, dense+KV on GPU 0) |
| decode traffic | 423.47 GiB/tok (the cpu-offload wall) |
| fak | v0.40.0 @ `471c3d9`, `go build -tags cuda` (libfakcuda.a sm_80 prebuilt) |

> Note: the served **model-id string** defaults to `"mock"` (cosmetic — no `--model` was set). The **compute is genuinely GLM-5.2 Q4**: `planner:"inkernel"`, `q4k=true`, and a real prefill at **0.2 tok/s** (720 s for 154 tokens) — the deterministic mock planner returns instantly, so a 12-minute prefill is proof of real decode.

## Per-turn evidence (gateway `inkernel_chat` log)

| turn | role | prompt | cacheable | **reused** | prefill | prefill time |
|---:|---|---:|---:|---:|---:|---:|
| 1 | cold prime | 154 | 0 | 0 | 154 tok | 720.62 s |
| 2 | replay+suffix | 160 | 140 | **140** | 20 tok | 105.84 s |

`X-Trace-Id=cachebite-1783963811` (fixed on both turns); `max_tokens=1`, `temperature=0`.

## Metric delta (`/metrics`, before → after)

| metric | before | after | delta |
|---|---:|---:|---:|
| `fak_gateway_kv_prefix_reused_tokens_total` | 0 | **140** | **+140** |
| `fak_gateway_kv_prefix_prompt_tokens_total` | 0 | 314 | +314 |
| `fak_gateway_inference_cached_prompt_tokens_total` (provider) | 0 | 0 | 0 |
| `fak_cache_saved_by_mechanism{owner="fak",mechanism="kv_prefix_reuse"}` | — | 140 | — |

Realized reuse: **87.5 %** of turn 2's prompt (140/160); 44.6 % aggregate across both turns (140/314).

## Recipe (the two prior-failure modes, neutralized)

1. **Turn 2 never completed** (GLM decode wall) → `max_tokens=1`: reuse is a **prefill-time** property, witnessed without paying decode.
2. **Turn 2 in a different trace scope** → **fixed `X-Trace-Id`** on both turns (the gateway keys the KV-prefix session off that header).
3. **Mock-planner masquerade** → gate on `/healthz` `planner=="inkernel"` before trusting anything.

Run harness: `scratchpad/glm_cachebite.sh` (self-contained build → serve → gate → 2 turns → metric delta → sentinel), launched detached via `fak-private/tools/dgxsh.py bg` + an on-box waiter. Transport unblock (fresh session when the persistent bridge sessions wedge): `dgxsh.py newsess`.
