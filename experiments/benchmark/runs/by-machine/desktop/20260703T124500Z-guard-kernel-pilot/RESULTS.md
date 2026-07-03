# Guard kernel-control pilot — CPU-scale, this box (2026-07-03)

A **runnable-today** pilot of the #2509 ablation, standing in for the box-gated
GPU server matrix. It proves the guard-value axes end-to-end on a kernel we fully
control — fak's own in-kernel engine — at CPU scale, so every number below is
WITNESSED on this host, not `not yet`. The GPU server matrix swaps the model
(GLM-5.2) and the box; the *seams and their witnesses are identical* and are the
ones exercised here.

**Model / kernel:** Qwen2.5-1.5B-Instruct Q8_0, served by
`fak serve --engine inkernel` (pure fak forward pass, greedy argmax, no
provider, no GPU). **Binary:** `8b6e0d7a` (+uncommitted pilot files).
**Workload:** one identical driver (`driver.py`) — 2 warmup + 20 single-turn +
6 shared-prefix multiturn streamed completions — run once per arm:

- **direct** (guard-off): driver → `fak serve` at `127.0.0.1:8091`.
- **guard** (guard-on): `fak guard --gguf … --provider openai -- python driver.py`
  — the guard stands up its own in-process gateway with the in-kernel model as
  the upstream (`upstream: in-kernel … LOCAL — fak runs the model itself; no API
  key, no network`) and injects only `OPENAI_BASE_URL` into the child.

## The five axes — witnessed

| # | Axis | Witness on this box | Artifact |
|---|---|---|---|
| 1 | **Witnessed economics** | fak's own KV-prefix cache **BIT**: reused **615 / 1370** prefill tokens = **44.9%** across 28 turns (frozen 0 / partial 14 / cold 14). Provider `cache_read` = **0** (no provider). | `cache-witness-guard.json`, `metrics-*.txt` |
| 2 | **Clean hop pricing** | guard-on−off single-turn TTFB: **median −0.056 s, mean −0.116 s** (n=20/arm). The hop is **within measurement noise** of a ~4.5 s CPU decode — i.e. the adjudication cost is unmeasurably small next to generation, exactly the claim. On a serve we control this is priceable; against an API it is not (jitter ≫ hop). | `timings-*.jsonl` |
| 3 | **Enforcement depth** | forced `wipe_disk` tool call → **DENIED** by the kernel: `DEFAULT_DENY/TERMINAL`, `1 denied before a wasted round-trip`. The model's own reply is rewritten to *"All proposed tool calls were refused by the fak kernel."* | `canary-guard.json`, `guard-audit-canary.jsonl` |
| 4 | **Credential-surface elimination** | K-arm child env holds **zero provider credentials** (`cred_like_names` = `[GLAMA_API_KEY]` is an ambient shell var, not injected by guard; no `ANTHROPIC_*`/`OPENAI_API_KEY` present — the upstream is in-kernel, "no API key, no network"). | `childenv-guard.json` |
| 5 | **Deterministic adjudication** | two **independent** gateway processes (direct serve vs guard's in-process gateway) on the identical workload produced **byte-identical** KV counters — reused **615/1370**, 28 turns, partial 14 / cold 14 — the greedy-argmax reproducibility that makes guard behavior regression-diffable. Impossible against a sampled API. | `metrics-direct.txt` vs `metrics-guard.txt` |

## Journal integrity

`fak audit verify guard-audit-canary.jsonl` → **OK: 1 hash-chained row, chain
intact (no edit since written)**. The deny is tamper-evident, not a log line.

## Honesty fence (trust classes)

- **WITNESSED** (fak controls): KV-prefix `reused/prompt_tokens` and the reuse
  ratio; the deny verdict + hash-chained journal; the credential-env audit; the
  cross-process determinism.
- **WITNESSED-derived timing** (wall-clock on this box): the TTFB distributions
  and the guard-on−off delta.
- **OBSERVED**: nothing from a provider here (there is none) — provider
  `cache_read` = 0 is structural, not a fak claim. Decode tok/s is a reading of
  this CPU host and is never attributed to a fak action.

No number crosses the WITNESSED/OBSERVED line; the packet inherits the
[GLM52 results-doc fence](../../../../../docs/benchmarks/GLM52-FAK-KERNEL-CACHE-VALUE-RESULTS.md).

## What this pilot does and does NOT establish

- **Does:** every axis of `fak guard`'s kernel-control value is real and
  measurable *today* on a kernel we own — the deny fires pre-round-trip, the KV
  cache bites, the hop is sub-noise, the child holds no secret, and the verdict
  stream is deterministic. The design note's axes are no longer only-on-paper.
- **Does not:** produce the GLM-5.2 numbers or the K2 (vLLM/SGLang) arm — those
  need the GPU server class box (still `not yet`, #1012 / `witness-guard-kernel-ablation-mgpu`).
  This pilot is the CPU-scale existence proof that the harness and every witness
  work end-to-end, so the GPU server run is a substitution of model+box, not new wiring.

## Reproduce

```bash
go build -o fak ./cmd/fak
./fak serve --gguf <Qwen2.5-1.5B-Instruct.Q8_0.gguf> --model qwen2.5-1.5b --addr 127.0.0.1:8091 --context-budget-tokens 4096 &
FAK_PILOT_BASE=http://127.0.0.1:8091 python driver.py direct .
kill %1
./fak guard --audit guard-audit.jsonl --gguf <…Q8_0.gguf> --provider openai -- python driver.py guard .
./fak guard --audit guard-audit-canary.jsonl --gguf <…Q8_0.gguf> --provider openai -- python canary.py .
./fak swebench cache-witness --metrics-file metrics-guard.txt --out cache-witness-guard.json
./fak audit verify guard-audit-canary.jsonl
```
