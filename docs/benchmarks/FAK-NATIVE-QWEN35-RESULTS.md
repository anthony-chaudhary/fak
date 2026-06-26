---
title: "fak in-kernel Qwen3.5/3.6 Gated-DeltaNet on M3"
description: "fak's own in-kernel forward pass runs the Qwen3.5/3.6 hybrid Gated-DeltaNet architecture end-to-end, witnessed from 0.8B safetensors to the 27B q4_k_m GGUF on Apple M3."
---

# fak's own engine runs Qwen3.6-27B (`qwen35`) — witnessed on M3

> The headline for the goal lane: fak's **own in-kernel forward pass** — not
> llama.cpp — now runs the **Qwen3.5/Qwen3.6 hybrid Gated-DeltaNet architecture**
> end-to-end in chat on Apple M3 Pro. The 0.8B safetensors run is the coherent
> f32 architecture witness; the **27B q4_k_m GGUF now also loads and generates
> through fak's pure in-kernel path** via the GGUF->Q8 cached GDN runtime.

## Witness (reproducible)

```
$ /tmp/fakchat -hf ~/.cache/fak-models/qwen3.5-0.8b \
    -p "What is the capital of France? Answer in one short sentence." -n 40
model=qwen3_5  load=1203ms  prompt_tokens=32  backend=fak in-kernel Gated-DeltaNet (f32, cacheless)
<think>

</think>

Paris is the capital of France.
```

Correct, coherent, and it even emits the Qwen `<think>` block. The full path is
fak's own: `internal/tokenizer` (Encode/Decode, oracle-validated) → ChatML template
→ `model.Forward` running the **Gated-DeltaNet linear-attention scan + gated
full-attention** (`internal/model/qwen35.go`, ported from transformers
`Qwen3_5GatedDeltaNet`) → sampling → detokenize → stream.

## 2026-06-19 cached-decode refresh

The original M3 chat witness above was the cacheless path. Current fak now routes the
Qwen3.5/Qwen3.6 f32 safetensors path through `Session.Prefill` / `Session.Step`:
full-attention KV and the Gated-DeltaNet recurrent conv/state both live in the session.
That makes `cmd/fakchat` and `cmd/qwen35check` use cached decode instead of rerunning
whole-sequence `Forward` for each generated token. Current unit witnesses are
`TestQwen35HybridSessionMatchesForwardAndPersistsState` and
`TestQwen35HybridQuantTokenLoopPersistsState`.

## 2026-06-19 Qwen3.6-27B pure-fak GGUF witness

This is the real 27B artifact on this M3 Pro, with no `llama-server`, no external
OpenAI-compatible proxy, and no llama.cpp in the execution path:

```sh
/usr/bin/time -l go -C fak run ./cmd/fakchat \
  -gguf /Users/USER/.cache/fak-models/gguf/Qwen3.6-27B.q4_k_m.gguf \
  -tok /Users/USER/.cache/fak-models/tokenizers/qwen3.6 \
  -p "Say OK." \
  -n 1
```

Observed output:

```text
model=qwen35  load=75505ms  prompt_tokens=22  backend=fak in-kernel Gated-DeltaNet (GGUF->Q8, cached)
<think>
---
prefill: 22 tok in 40.62s (0.5 tok/s)  |  cached qwen3_5 decode: 1 tok in 16.25s (0.1 tok/s)
      135.67 real       339.68 user        97.23 sys
         25785204736  maximum resident set size
```

What this proves:

- The real 15 GB `Qwen3.6-27B.q4_k_m.gguf` parses as `qwen35`, maps all 851 tensors,
  loads through `ggufload.LoadModelQuant`, and builds a runnable `*model.Model`.
- `cmd/fakchat` runs tokenizer -> ChatML ids -> `Session.Prefill` -> sample ->
  detokenize -> `Session.Step` entirely inside fak.
- The 27B model fits this 36 GB M3 Pro in the current GGUF->Q8 runtime, peaking at
  about 25.8 GB RSS.
- The first generated token matches llama.cpp b9707 greedy raw-ChatML output for the
  same 22-token prompt: token id `248068`, `<think>`.
- The #93 oracle has now been extended beyond one token. llama.cpp b9707 returns
  `[248068, 198, 90700]` (`<think>\nThinking`) for the same prompt; fak's current
  GGUF->Q8 path returns `[248068, 198, 8160]`, so the first two tokens match and the
  third token is the current real-artifact divergence. Artifacts:
  `experiments/qwen36/llamacpp-qwen36-multitoken-oracle-20260619.json` and
  `experiments/qwen36/native-gguf-q8-multitoken-parity-20260619.json`.

What this does **not** claim yet:

- It is not a llama.cpp speed parity result. The current pure-fak path is an
  unoptimized CPU/Q8 reference for this architecture and is about 0.1-0.5 tok/s here.
- It is not a full-logit or multi-token parity claim yet. The same 22-token ChatML
  prompt tokenizes byte-exactly against llama.cpp and now matches the first two
  llama.cpp greedy tokens, but the three-token #93 oracle fails at step 2 (`8160`
  instead of llama.cpp's `90700`).

## 2026-06-26 `FAK_QPROFILE` per-op phase profile + bottleneck split (#438)

The 22-token smoke above reports only one prefill/decode number — it cannot say *where*
the time goes (GDN projections vs conv/scan vs full-attention vs MLP vs head). `cmd/fakchat`
now answers that: set **`FAK_QPROFILE=1`** and the cached `qwen3_5` (Gated-DeltaNet) prefill
and decode each emit a per-op wall-time split to stderr. The profiler is opt-in and the
default path attaches none (zero instrumentation cost). The instrumented op classes cover the
whole hybrid forward, every item the #438 acceptance list names:

- **embedding** (`embed`), **input/post/final norms** (`input_norm`, `post_attn_norm`,
  `final_norm`), **activation quantization** (`q8_panel_quantize` / `q8_norm_quant`).
- **GDN linear-attention**: in-projection (qkv/z/a/b) `qwen35_linear_in_proj`, depthwise
  **conv1d** `qwen35_linear_conv`, q/k-norm `qwen35_linear_qk_norm`, **recurrent delta scan**
  `qwen35_linear_recurrent`, **gated RMSNorm** `qwen35_linear_gated_norm`, **out-projection**
  `qwen35_linear_out_proj` (and the `_step_` twins on the decode path).
- **Gated full-attention**: `qwen35_full_qkv_proj`, `qwen35_full_split_gate`,
  `qwen35_full_qk_norm_rope`, `qwen35_full_attn`, `qwen35_full_gate`, `qwen35_full_o_proj`.
- **MLP** `mlp_gate_up_proj` / `mlp_activation` / `mlp_down_proj` (decode: `mlp_decode`) and
  the **logits/head** `lm_head_q8`.

The mechanism is pinned by `TestQwen35HybridQPhaseProfilerRecordsPrefillAndDecode`
(`internal/model/qwen35_test.go`), which asserts both phases record their ops on a synthetic
hybrid — so the profiler can run GPU-free in CI and cannot silently stop recording an op.

### The measured profile (Qwen3.6-27B q4_k_m, Apple M3 Pro, `-tags fakmetal`)

Captured clean on `node-macos-a` (M3 Pro / 36 GB, llama-server stopped for the whole pool),
full method, fences, and raw artifact in
[`docs/notes/MAC-QWEN36-27B-Q4K-METAL-PERF-DIAGNOSIS-2026-06-26.md`](../notes/MAC-QWEN36-27B-Q4K-METAL-PERF-DIAGNOSIS-2026-06-26.md)
(artifact: `experiments/benchmark/runs/by-machine/node-macos-a/20260626T055239Z-q4k-metal-decode-27b/score.json`).
Clean headline: **decode 1.2 tok/s** (64 tok / 54.12 s), **prefill 0.6 tok/s** (29 tok /
48.27 s) vs the llama.cpp-Metal bar 7.29 / 51.55 and the 3× goal 2.7. The phase split below is
from that run (a slightly longer prompt than the n=1 smoke, which gives the decode loop real
steps to attribute); the per-op attribution, not the single-run tok/s, is the durable signal.

**Decode** (total 54122 ms / 64 tokens = 845 ms/token):

| phase | ms | % | calls | ms/call |
|---|---:|---:|---:|---:|
| `mlp_decode` | 29200 | **54.0%** | 4096 | 7.13 |
| `qwen35_linear_step_in_proj` | 8697 | 16.1% | 3072 | 2.83 |
| `qwen35_linear_step_out_proj` | 3453 | 6.4% | 3072 | 1.12 |
| `qwen35_linear_step_recurrent` | 3199 | 5.9% | 3072 | 1.04 |
| `full_attn_qkv_proj` | 2900 | 5.4% | 1024 | 2.83 |
| `lm_head_q8` | 1934 | 3.6% | 64 | 30.2 |
| `full_attn_o_proj` | 1066 | 2.0% | 1024 | 1.04 |
| rest (attn, conv, norms, gate) | ~3700 | 6.8% | — | — |

**Prefill** (total 48273 ms / 29 tokens): `mlp_gate_up_proj` **54.7%** (412 ms/call),
`mlp_down_proj` 18.2%, `qwen35_linear_in_proj` 11.2%.

The matmuls (MLP + projections) are ~85% of both phases. The cause is **orchestration, not
arithmetic**: each decode token runs ~336 *separate* Metal command-buffer GEMVs, each ~360 µs
launch/sync-bound on top of ~98 µs of bandwidth-limited work, so `mlp_decode` sits at 7.1
ms/call where a ~40 MB Q4_K read at ~150 GB/s should be ~0.27 ms. The kernels are correct
(GEMV cosine 1.000000 vs CPU; greedy decode token-parity) — they are launch-starved.

### Top-two bottlenecks → follow-up tickets with before/after gates (#438 → #59)

| # | bottleneck (profile evidence) | lever | follow-up | before → after gate |
|---|---|---|---|---|
| 1 | **Per-call command-buffer overhead** — `mlp_decode` 54% + every projection; ~336 command buffers/token, ~360 µs each | **one MTLCommandBuffer per token** (resident decode forward; the shipped resident *prefill* twin pays the launch once) | **#67** | decode **1.2 → ≥ 2.7 tok/s** (3× goal); `BenchmarkMetalQ4KGemvBatch` (64 GEMVs in one command buffer) is 5.2× faster/GEMV → projected ~5.9 tok/s |
| 2 | **Low in-kernel `q4k.m` utilization** — GEMV 32.2 GB/s (≈21% of ~150) , GEMM 4.66 GB/s / 364 GFLOP/s (≈5% FLOP) | **kernel-efficiency pass**: coalesced dequant, threadgroup/grid sizing, `simdgroup_matrix`; resident weights (no per-call upload) | **#68** (decode GEMV) · **#69** (resident weights) | GEMV **21% → ≥ 60%** device BW (≥ ~90 GB/s); decode toward the llama.cpp-Metal **7.29** bar |

Both levers are tracked under the Metal epic **#59**; #67 is the primary (decode) lever and
#68/#69 the kernel/residency lever. This profiling ticket (#438) does **not** claim the speed
fix — it splits it, evidenced.

## How it works today

- **Arch dispatch**: `cfg.IsQwen35Hybrid()` (from `layer_types`) selects the hybrid
  forward. 24 layers for 0.8B; the 27B is the same op set at 64 layers (48 GDN + 16
  full-attn).
- **Linear-attention (GDN)**: causal depthwise conv1d + SiLU, q/k L2-norm + query
  scale, recurrent gated delta rule (decay `exp(-exp(A)·softplus(a+dt_bias))`,
  sigmoid β, rank-1 state update), per-head gated RMSNorm, out-proj.
- **Full-attention**: split per-head `[query|gate]`, sigmoid output gate, partial
  RoPE (rotary_dim < head_dim), qk-norm.
- **Decode is cached for the f32 safetensors path**: `Session.Prefill` builds the
  recurrent GDN state and full-attention KV once; `Session.Step` advances both. Cache
  eviction for GDN state is deliberately fail-loud until the quarantine semantics are
  implemented for recurrent state.

## The 27B-size status on a 36 GB box (precise)

The f32 path is still too large for 36 GB, but the GGUF->Q8 path now runs:

| Path | Footprint | Fits 36 GB? | Missing |
|---|---|---|---|
| f32 (`LoadSafetensorsDir`, the validated GDN path) | 27B×4 ≈ **108 GB** | ✗ | — |
| GGUF->Q8 cached runtime | observed **25.8 GB RSS** | ✓ | speed/broader logit-parity work |
| native q4 GGUF runtime | ≈ 16 GB | ✓ in principle | direct q4 kernels / no Q8 expansion |

The size gate is closed for an end-to-end command-line smoke. The remaining gap is
performance/correctness evidence at the real-artifact level: load-time reduction (#95),
direct q4 residency (#96), GDN/full-attention phase profiling and acceleration (#97),
device prefill for the GDN/full-attention projections (#92), and a short llama.cpp/HF
oracle to prove logits rather than just execution (#93).

## Native parity witness commands (#442)

These are the exact commands behind the `qwen35` correctness witnesses, with their
boundaries. None of them need the 27B artifact or a GPU — the oracle is a **tiny,
CPU-instantiable** `qwen3_5` fixture (the published Qwen3.6-27B is the only real size, so
the fixture is built from the *real* HF `Qwen3_5ForCausalLM` modeling code with random
weights, exactly as the GLM/OLMo2/MiniMax oracles are). The fixture and its export are
gitignored; everything else is committed.

**1 — build the tiny `qwen3_5` fixture and export the HF oracle** (needs `transformers>=5.10`
+ torch on a plain CPU box):

```bash
python internal/model/make_qwen35_tiny.py .cache/qwen35-tiny
python internal/model/export_oracle.py --online \
  --model .cache/qwen35-tiny --out internal/model/.cache/oracle-qwen35 \
  --prompt-ids-json '[[785,6722,315,9621,374],[16,11,220,17,11,220,18,11,220,19,11],[750,912,2877,11,293,982,262,470]]'
```

**2 — run the in-kernel parity + tensor-name gates** (native `go test` is blocked on a
native-Windows host by an OS Application-Control policy, so run the suite under WSL via
`fak/test.ps1`; on Linux/macOS run `go test` directly):

```powershell
# from the repo root
$env:FAK_ORACLE_DIRS = '.cache/oracle-qwen35'
.\fak\test.ps1 -count=1 ./internal/model/ -run Qwen35
go test ./internal/ggufload -run Qwen35 -count=1
```

When the oracle is present this proves, per fixed token-id prompt: the loader derives the
hybrid `qwen3_5` knobs ((1+w) RMSNorm, qk-norm, sigmoid output gate, partial RoPE); each
layer maps to its own mixer tensor names (`linear_attn.*` on the Gated-DeltaNet layers,
`self_attn.{q,k,v,o}_proj` on the gated full-attention layer); per-layer hidden-state
cosine ≥ 0.9999 vs HF; argmax parity at every position; and cached `Session.Prefill`
reproduces the cacheless `Forward` last-position logits. Without the oracle every
`TestOptionalQwen35…` case **skips** cleanly, so CI without weights stays green. The
gates: `TestOptionalQwen35HybridOracleForwardMatchesHF` (in-kernel forward/cache parity)
and `internal/ggufload`'s `TestQwen35GGUFConfigCanonicalizesHybridTensorsAndRunsForward`
+ `TestOptionalQwen35GGUFMapsEveryTensorName` (tiny- and real-GGUF tensor-name mapping).

**3 — llama.cpp generated-token comparison** on the real 27B artifact (the parity bar,
#88/#93). This is the documented command; it needs llama.cpp + the GGUF and is GPU/Metal
host-gated, so it is a host-bound witness, not a CI gate:

```sh
# greedy raw-ChatML token ids on the shared prompt — compare against fak's own decode
llama-cli -m Qwen3.6-27B.q4_k_m.gguf -p "<the shared 22-token ChatML smoke prompt>" \
  -n 3 --temp 0 --top-k 1 --samplers greedy
# fak's own decode for the same prompt:
go run ./cmd/fakchat -gguf Qwen3.6-27B.q4_k_m.gguf -tok <tok-dir> -p "Say OK." -n 1
```

Pinned oracle/measurement artifacts:
`experiments/qwen36/llamacpp-qwen36-multitoken-oracle-20260619.json` (llama.cpp b9707
returns `[248068, 198, 90700]`) and
`experiments/qwen36/native-gguf-q8-multitoken-parity-20260619.json` (fak's current
GGUF→Q8 path returns `[248068, 198, 8160]` — first two tokens match, the third is the
current real-artifact divergence). The tiny-fixture oracle proves the *architecture* is
bit-faithful to HF; the 27B llama.cpp token comparison is the remaining real-artifact
parity work, tracked open below.

## Status summary

- ✅ Qwen3.6-27B runs end-to-end in chat **on this setup** in llama.cpp b9707 Metal —
  the speed/quality parity bar.
- ✅ Qwen3.6-27B now also runs end-to-end in chat through **fak's own in-kernel engine**
  using the local GGUF->Q8 cached GDN path.
- ✅ The real 27B GGUF path now matches llama.cpp's greedy first token on the exact
  ChatML smoke prompt (`<think>`, id `248068`).
- ✅ The Qwen3.6 **architecture** runs end-to-end in chat in **fak's own engine**
  (0.8B witness above).
- ✅ The f32 hybrid session path now has cached prefill/step decode witnesses, so
  standalone test benches no longer exercise only the cacheless path.
- ⏳ Recurrent-state eviction/quarantine, direct q4/chunked/device GDN performance work,
  and broader real-artifact llama.cpp/HF logit witnesses remain open.

For GPU server and standalone endpoint-backed test benches, use a multi-GPU datacenter GPU serving host.

_Witnessed 2026-06-18 and refreshed 2026-06-19 on Apple M3 Pro (36 GB). fak rows are fak's own forward pass._
