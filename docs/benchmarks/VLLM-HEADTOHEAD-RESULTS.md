---
title: "fak vs vLLM head-to-head: the adjudication tax + the engine bench (gated scaffold)"
description: "How fak measures itself against vLLM as a first-class peer: a gateway adjudication-tax witness (fak-fronts-vLLM vs raw-vLLM) and a vLLM engine in the GPU throughput bench. Status: pending-measurement — every vLLM cell is a placeholder until a serving-node run lands a committed artifact; the only real numbers here are the measured SGLang sibling's."
---

# VLLM-HEADTOHEAD-RESULTS — fak vs vLLM, measured the same way SGLang is

> **Status: pending-measurement (GATED scaffold).** No vLLM GPU run has landed yet.
> **Every numeric cell in the vLLM tables below is a placeholder (`TBD`)** and stays
> that way until a run on a serving node writes a committed artifact under
> `experiments/vllm/` or `experiments/benchmark/runs/`. The *only* real numbers in this
> document are the **measured SGLang sibling's** (§4), and they are fenced there and
> labelled as the SGLang stack — never copied into a vLLM table as if measured. This is
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
| `agent` | fak's coding agent drives the vLLM-served model; every tool call adjudicated | PENDING-MEASUREMENT |
| `gateway-openai` | `fak serve` fronts vLLM over the OpenAI-compatible wire | PENDING-MEASUREMENT |
| `mcp-http` | the same served model reached over fak's MCP surface | PENDING-MEASUREMENT |

Artifact (pending): `experiments/vllm/surface-smoke.json`.

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
`experiments/qwen36/dgx-r4-20260622/compare.json`. The vLLM tax (§3) is *expected* to behave
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
# Run on a serving node with a CUDA GPU (this laptop has none / is WDAC-blocked).
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
`experiments/qwen36/dgx-r4-20260622/`.

## Related

- [fak vs SGLang on an 8-GPU server (measured sibling)](QWEN36-27B-GPU-SERVER-RESULTS.md)
- [fak vs llama.cpp across every performance axis (CPU + GPU)](LLAMACPP-HEADTOHEAD-RESULTS.md)
- [fak vs vLLM, SGLang & provider KV caching (positioning)](../fak-vs-alternatives-comparison.md)
- [Industry scorecard — serving dimensions](../industry-scorecard/serving.md)
