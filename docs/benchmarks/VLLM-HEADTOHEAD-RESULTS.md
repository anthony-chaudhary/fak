---
title: "fak vs vLLM head-to-head: the adjudication tax + the engine bench (gated scaffold)"
description: "How fak measures itself against vLLM as a first-class peer: a gateway adjudication-tax witness (fak-fronts-vLLM vs raw-vLLM) and a vLLM engine in the GPU throughput bench. Status: pending-measurement — every vLLM cell is a placeholder until a serving-node run lands a committed artifact; the only real numbers here are the measured SGLang sibling's."
---

# VLLM-HEADTOHEAD-RESULTS — fak vs vLLM, measured the same way SGLang is

> **Status: pending-measurement (GATED scaffold).** No vLLM GPU run has landed yet.
> **Every numeric cell in the vLLM tables below is a placeholder (`TBD`)** and stays
> that way until a run on a serving node writes a committed artifact under
> `experiments/vllm/` or `experiments/benchmark/runs/`. The only real *vLLM-comparison*
> numbers in this document are the **measured SGLang sibling's** (§4), and they are fenced
> there and labelled as the SGLang stack — never copied into a vLLM table as if measured.
> (Two things here ARE real but are NOT vLLM-vs-fak *numbers*: the on-host
> **adjudication-overhead floor** in §3a — a GPU-free kernel read-path measurement — and
> the §1 `gateway-openai` **wire integration**, proven GPU-free in CI by
> `internal/gateway` `TestChatProxyFrontsVLLMAndSGLangServedToolCalls`. Both are
> pass/fail integration facts, not throughput comparisons.) This is
> the honesty contract this repo holds every benchmark to (`BENCHMARK-GOVERNANCE.md`):
> no fabricated numbers, and a comparison shows where fak *loses*.

vLLM (PagedAttention + continuous batching, Kwon et al., SOSP 2023) is the de facto open
SOTA serving engine. fak now measures itself against it **as a first-class peer**, on the
same two surfaces it already measures llama.cpp and SGLang on:

1. **The engine throughput bench** — vLLM is a selectable engine in `tools/gcp_bench.py`,
   head-to-head against llama.cpp and fak's own engine on the *same* GPU (§2). This is the
   llama.cpp-style comparison.
2. **The gateway adjudication tax** — `tools/vllm_tax_witness.py` fronts a raw vLLM `/v1`
   endpoint with `fak serve` and measures the cost of routing every turn through fak's
   adjudication plane (§3). This is the SGLang-GPU-server-style comparison
   ([`QWEN36-27B-GPU-SERVER-RESULTS.md`](QWEN36-27B-GPU-SERVER-RESULTS.md)).

**The honest verdict, stated before any number lands:** on raw single-instance throughput
fak is **not** vLLM's peer and does not try to be — vLLM's continuous batching is its whole
design. fak's value here is the **adjudication / coherence / measurement plane, not raw
tok/s**; the gateway tax is the price of that plane, and fak is **expected to trail raw
vLLM**, exactly as it trails raw SGLang (§4). On big models fak **fronts** vLLM rather than
out-serving it. The one axis fak leads is a *different* one — cross-worker / cross-session
prefix reuse that composes *on top of* vLLM's per-instance prefix caching
([`fak-vs-alternatives-comparison.md`](../fak-vs-alternatives-comparison.md)).

## 1. Surfaces (coding agent on a vLLM-served model)

| Surface | What it proves | Status |
|---|---|---|
| `agent` | fak's coding agent drives the vLLM/SGLang-served model; every tool call adjudicated | PENDING-MEASUREMENT (live node) |
| `gateway-openai` (**wire**) | `fak serve` fronts a vLLM (`--enable-auto-tool-choice`) / SGLang upstream over the OpenAI-compatible wire and runs every proposed tool call through fak's adjudication plane before forwarding | **TESTED — GPU-free, CI** (`TestChatProxyFrontsVLLMAndSGLangServedToolCalls`) |
| `gateway-openai` (**wall-clock**) | the live latency / decode-tok/s tax of that same hop against a real served model | PENDING-MEASUREMENT (live node) — §3 |
| `mcp-http` | the same served model reached over fak's MCP surface | PENDING-MEASUREMENT (live node) |

The **wire** integration is the part that needs no GPU: it proves `fak serve` decodes the
exact tool-call wire each engine emits (vLLM's `--enable-auto-tool-choice` `tool_calls`,
SGLang's `tool_calls` with content alongside) and adjudicates each one (allow kept / deny
dropped / transform redacted) before forwarding — the protocol-level drop-in, proven in CI
today so the lane is a *tested integration* rather than prose. Only the **wall-clock** tax
(§3) and the live-model surfaces stay host-gated.

```bash
go test ./internal/gateway -run TestChatProxyFrontsVLLMAndSGLangServedToolCalls -v   # GPU-free wire witness
```

Artifact: the wire integration is proven in CI by the named Go test (above); the live-node
smoke (`experiments/vllm/surface-smoke.json`) stays pending.

## 2. Engine throughput head-to-head (fak engine vs vLLM vs llama.cpp, same GPU)

`tools/gcp_bench.py` runs vLLM as a first-class engine alongside llama.cpp and fak's own
engine on one GPU and one model, folding every engine's normalized row into one
`result.json` (`fak.gcp-vm-bench.v2`). **Apples-to-apples fence:** vLLM loads the **HF
(bf16)** form of the model and reports an **aggregate continuous-batching** throughput
(vLLM's actual strength); the llama.cpp and fak rows use the **GGUF Q8** weights and a
single-stream slice. The `precision` and `note` fields in each engine row disclose this —
the numbers are comparable in *hardware* and *model identity*, not in precision or batching
regime, and the table must be read with that disclosure.

| Engine | Backend / precision | prefill tok/s | decode tok/s |
|---|---|---|---|
| llama.cpp | llama.cpp CUDA / Q8_0 | TBD | TBD |
| **vLLM** | vLLM PagedAttention / bf16 (aggregate) | n/a (server) | TBD |
| fak-cuda | fak CUDA backend / f32 | TBD | TBD |

Artifacts (pending): a run dir under `experiments/benchmark/runs/by-machine/<machine>/`
with `result.json` carrying a `vllm` engine row.

## 3. Gateway adjudication tax (fak-fronts-vLLM vs raw-vLLM)

`tools/vllm_tax_witness.py` runs the same deterministic chat payload N times against (a) raw
vLLM and (b) `fak serve` in front of it, interleaved to cancel server variance, and reports
the **tax** as a delta:

- `latency_tax = gateway median latency / raw median latency` (≥1 ⇒ fak slower)
- `decode_tps_tax = raw median decode_tps / gateway median decode_tps` (≥1 ⇒ fak slower)

| Metric | raw vLLM | fak-fronts-vLLM | tax |
|---|---|---|---|
| median latency (s) | TBD | TBD | TBD |
| median decode tok/s | TBD | TBD | TBD |

fak is **expected to trail** (tax ≥ 1) — the witness records the tax so a *regression* in it
can be caught against a recorded baseline (KEEP/REVERT, like `tools/bench_witness.py`); it
**never frames the tax as a fak win**. Artifact (pending):
`experiments/vllm/adjudication-tax-witness.json`.

### 3a. The on-host adjudication-overhead FLOOR (GPU-free, MEASURED — [#451](https://github.com/anthony-chaudhary/fak/issues/451))

The §3 table above is the *end-to-end network* tax — it needs a live vLLM serving node and
stays `TBD` until one runs. But a provider weighing the drop-in does not have to wait for a
GPU to learn the **lower bound** of what fronting their engine with `fak serve` costs: the
per-call overhead fak's adjudication plane adds **before any byte crosses the wire**. That
floor is measurable on any laptop, today, with no model and no GPU — and it is published here
so the "complementary drop-in" claim carries a number instead of prose.

Every tool call fak routes folds over the kernel's driver/policy registries (the
adjudicators, fast-paths, result-admitters, emitters the decide path consults). That fold is
the repo's **zero-alloc read path**: a single atomic-pointer load, no mutex, no per-call
allocation — and, critically, **flat as the policy/driver count grows**, so the drop-in cost
does not creep as a provider registers more policy.

| Path (no model, no GPU) | ns/call | allocs/call | scaling |
|---|---:|---:|---|
| Zero-alloc registry read fold (the decide path walks it every call) | **~0.55** | **0** | **FLAT, N=1→1000 drivers** |
| Full per-call adjudication decide (read-only ALLOW, end-to-end) | **~362–373** | 5 | per-call, sub-µs |

- **What "FLAT" buys a provider.** Adding the 1000th policy/driver costs the read path
  nothing — `~0.55 ns/op, 0 allocs/op` at N=1 and at N=1000 alike. The capability floor a
  provider deploys in front of vLLM/SGLang does not get more expensive per call as it grows.
- **Proven, not asserted.** `internal/abi/registry_scaling_test.go::TestRegistryReadsZeroAlloc`
  FAILS the build if any future change reintroduces a per-call allocation or lock-copy on a
  read accessor (0 allocs/op with 256 drivers of every kind). The number is `BenchmarkRegistryReadScaling`.
- **Honest fence — this is a FLOOR, not the tax.** The end-to-end gateway tax a provider
  actually pays is dominated by the network proxy hop and the engine, **not** by this fold —
  see the SGLang measurement in §4 (0.75× at peak, converging to a ~3% tax at saturation).
  The full per-call *decide* (`362 ns` allow, the "Pure-kernel decide latency" row in
  [`BENCHMARK-AUTHORITY.md`](https://github.com/anthony-chaudhary/fak/blob/main/BENCHMARK-AUTHORITY.md)) is the realistic per-call cost
  fak adds before the engine runs; the `~0.55 ns` zero-alloc fold is the registry-read
  *component* of it that the issue names. None of these three numbers is the vLLM end-to-end
  tax — that stays `TBD` (§3) until a serving-node run lands an artifact.

Reproduce (any host, ~3 s):

```bash
go test ./internal/abi -run TestRegistryReadsZeroAlloc -v          # the zero-alloc proof
go test ./internal/abi -bench BenchmarkRegistryReadScaling -benchmem   # the ns/call, flat across N
go test ./internal/adjudicator -bench 'BenchmarkDecide$' -benchmem      # the full per-call decide
```

Machine-stamped for the table above: AMD Ryzen 9 9950X, linux/amd64, 2026-06-26 (the
zero-alloc/flat-scaling **shape** is hardware-independent and reproduces the test exactly;
the absolute ns is single-box, per `BENCHMARK-GOVERNANCE.md` regime rules).

## 4. The measured sibling: the SGLang stack (real numbers, NOT vLLM's)

The vLLM tables above are placeholders. The **SGLang** head-to-head *did* run, on the same
gateway-fronting design, and its numbers are the closest measured evidence for what the vLLM
tax curve is likely to look like — but they are **SGLang's, not vLLM's**, and must not be
read as a vLLM result.

On an 8-GPU datacenter server (SGLang 0.5.10, TP=8, bf16; Qwen3.6-27B), fak-gateway vs raw
SGLang, completion tok/s at 64 requests/concurrency:

| concurrency | fak-gateway | raw SGLang | fak/raw |
|---:|---:|---:|---:|
| 8 | 249.6 | 415.4 | 0.60× |
| 64 (peak) | 1085.6 | 1451.6 | **0.75×** |
| 128 (saturation) | 1074.4 | 1103.2 | 0.97× |

**fak-gateway trails raw SGLang 0.75× at peak and converges to a ~3% tax at saturation**
(conc 128: 0.97×), where the per-request proxy cost is amortized. fak's value here is the
adjudication / coherence / measurement plane, **not raw tok/s** — single-stream/raw
throughput is *not* fak's axis. Source:
[`QWEN36-27B-GPU-SERVER-RESULTS.md`](QWEN36-27B-GPU-SERVER-RESULTS.md) §2 and
`experiments/qwen36/gpu-server-r4-20260622/compare.json`. The vLLM tax (§3) is *expected* to behave
similarly and is **to be measured, not assumed**.

## 5. Honest fences

- **Native-engine gap unchanged.** fak's own CUDA engine still can't load a quantized /
  multi-GPU 27B (no quantized device GEMM, no multi-GPU NCCL; f32 27B exceeds a single
  GPU's memory). The GPU-server path is and remains llama.cpp / vLLM / SGLang-serves +
  fak-fronts. The §2 bench therefore runs a small model that fits one GPU.
- **vLLM's regime is concurrency.** The §2 single-stream-ish slice is the directly-runnable
  head-to-head, **not** vLLM's best regime; its continuous-batching aggregate is recorded as
  such. Do not read a §2 row as "fak beats vLLM" or "vLLM beats fak" on throughput — they
  are different precisions and batching regimes on the same hardware.
- **vLLM tool-call parsing differs from SGLang.** A real vLLM agent run must set vLLM's own
  tool-call parser / `--enable-auto-tool-choice` for the served model, or the agent sees no
  `tool_calls`. This is a config item to verify on the run, not a settled value.
- **No fabricated numbers.** Until a serving-node run lands a committed artifact, every vLLM
  cell stays `TBD`. The SGLang numbers in §4 are the SGLang stack's, fenced and labelled.

## 6. Reproduce (run on a serving node)

```bash
# Run on a serving node with a CUDA GPU (on this Windows laptop, the GPU is unavailable / the native exe is WDAC-blocked).
# status: pending-measurement — the commands below are the gated reproduce path.

# 1. Engine throughput head-to-head (vLLM as a first-class engine vs llama.cpp + fak):
python tools/gcp_bench.py --tier g2-l4 --engine llama,vllm,fak-cuda

# 2. Stand a raw vLLM endpoint up (the ENGINE=vllm branch of the serve script):
ENGINE=vllm bash tools/glm52_sglang_vllm_serve.sh    # or a Qwen vLLM serve variant

# 3. Measure the fak gateway adjudication tax over that raw vLLM endpoint:
python tools/vllm_tax_witness.py \
    --base-url http://127.0.0.1:8000/v1 --model <served-model> --count 8 --record
```

Every shipped number from such a run must land a committed artifact (the gcp_bench run dir,
`experiments/vllm/adjudication-tax-witness.json`) — exactly as the SGLang run did under
`experiments/qwen36/gpu-server-r4-20260622/`.

## Related

- [fak vs SGLang on an 8-GPU server (measured sibling)](QWEN36-27B-GPU-SERVER-RESULTS.md)
- [fak vs llama.cpp across every performance axis (CPU + GPU)](LLAMACPP-HEADTOHEAD-RESULTS.md)
- [fak vs vLLM, SGLang & provider KV caching (positioning)](../fak-vs-alternatives-comparison.md)
- [Industry scorecard — serving dimensions](../industry-scorecard/serving.md)
