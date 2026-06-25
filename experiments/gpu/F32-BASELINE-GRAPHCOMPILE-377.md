# f32 "baseline broken": `GraphCompile:false` is the default, not the defect (#377)

> **Resolution of [#377](https://github.com/anthony-chaudhary/fak/issues/377)** —
> *"f32 baseline broken — CUDA graph non-functional; 26.8 tok/s vs 100-120 target,
> `backend.caps.GraphCompile: false`."* The premise is a **misdiagnosis**, the same shape
> as [#378](Q8-WSL-CUDA-RUNTIME-378.md) and [#379](Q8-CLOCK-RESOLUTION-379.md): the number
> is not a broken kernel, and `GraphCompile: false` is the **intended default**, not a
> "non-functional" path. Two facts close it, both grounded in artifacts already committed
> to this directory — **no new measurement is claimed**:
>
> 1. **CUDA graphs are OFF by design and would not help here** — even enabled they are a
>    *measured no-win* on this exact model (a slight regression), so they are categorically
>    not the missing 4× lever.
> 2. **100-120 tok/s is the wrong bar for f32** — f32 is the *slowest* precision; the
>    committed same-box witnesses show GPU f32 decoding *slower* than GPU Q8, exactly as
>    bandwidth dictates. 26.8 tok/s f32 is f32's launch-serialized speed, not a fault.

## Note on the cited artifact

The issue's `26.8 tok/s` comes from `experiments/gpu/progressive/P1-01-smollm2-f32.json`,
a file from the internal tracker's progressive-benchmark workflow that was **not** carried
into this tree (no `experiments/gpu/progressive/` directory exists here). So this resolution
does not re-derive that one number; it reasons from the **committed** SmolLM2-135M and
RTX-4070 witnesses in *this* directory, which establish the regime the 26.8 sits in. Every
number below is read from a file you can open.

## Fact 1 — `GraphCompile: false` is the deliberate default, and graphs are a measured no-win

The CUDA backend advertises the capability *exactly* when the graph path is live, and that
path is gated OFF unless `FAK_CUDA_GRAPH=1`
([`internal/compute/cuda.go`](../../internal/compute/cuda.go)):

```go
// Caps()
return Caps{Async: true, DeviceMemory: true, GraphCompile: graphEnabled, ...}
//                                            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// graphEnabled = os.Getenv("FAK_CUDA_GRAPH") == "1"   // init(), default false
```

So under any default build, `GraphCompile` reports **false** — and that is correct, not
"non-functional." The capture/replay path itself shipped and is witnessed: commit `300d745`
*"feat(compute): CUDA Graphs / capture for batch-1 decode (#483)"* added
`GraphBegin/GraphEndLaunch/GraphReset`, the `-tags cuda` parity witness
(`TestCUDAGraphDecodeParity`, argmax-exact + logit cosine ≥ 0.999 vs the eager path), and
the advertise/fallback contract test (`TestCUDAGraphCompileCapGated`, which *asserts*
`GraphCompile == false` when graphs are disabled). The path is functional; it is shelved on
purpose. The reason is recorded right at the gate (`cuda.go`):

> *PER-TOKEN capture is a measured dead end: re-instantiating a ~600-node graph every token
> costs ~what the 600 launches it replaces cost (**7.0 vs 7.5 tok/s** on SmolLM2-135M — no
> net win). The real win is instantiate-ONCE + replay-many, which needs a length-agnostic
> graph … a tracked redesign (issue #35/#3).*

Read that against the issue's expectation: enabling graphs takes SmolLM2-135M from 7.5 to
**7.0** tok/s — a ~7% *regression*, not a 4× gain. CUDA graphs amortize the per-launch host
tax (the ~30% residual between fak Q8 and llama.cpp Q8 analyzed in
[`Q8-CLOCK-RESOLUTION-379.md`](Q8-CLOCK-RESOLUTION-379.md) and `GPU-QWEN-RESULTS.md` §4),
*if* the graph is instantiated once and replayed — which the per-token capture is not. They
are not, and cannot be, the lever that turns 26.8 into 100-120.

The issue's three "Possible Causes" each fall to this:

| Issue's possible cause | Verdict from the tree |
|---|---|
| 1. CUDA graphs unsupported on RTX 4070 (sm_89) | **No** — graph capture shipped in #483 and runs on this Ada board; it is *gated off by default* (`FAK_CUDA_GRAPH`), a config default, not a hardware limit. |
| 2. WSL2 CUDA driver limitation | **No** — the WSL2 toolkit-runtime blocker is the separate #378, already resolved by the no-sudo `setup_cuda_wsl.sh`; the CUDA backend initializes and decodes on this box. |
| 3. Build configuration issue | **No** — `GraphCompile: false` is what a default build is *supposed* to report; the graph path is opt-in, and even opted-in it does not help (above). |

## Fact 2 — 100-120 tok/s is not an f32 bar; f32 is the slowest precision

Decode is memory-bandwidth bound: a batch-1 GEMV reads each weight once per token, so
throughput scales (roughly) with the inverse of the weight footprint. f32 is **4 bytes per
weight**; Q8_0 is ~1 (int8 code + a thin per-block scale band). f32 is therefore the
*slowest* path, not a fast "baseline" — and the committed same-box witnesses show exactly
that. All three are SmolLM2-135M, app `v0.24.0`, captured 2026-06-19, in this directory:

| precision · backend | file | decode tok/s |
|---|---|---:|
| **f32 · GPU** (Vulkan, RX 7600) | [`q8gpu-smollm2-135m-gpu-f32-20260619.json`](q8gpu-smollm2-135m-gpu-f32-20260619.json) | **16.5** |
| **Q8_0 · GPU** (Vulkan, RX 7600) | [`q8gpu-smollm2-135m-gpu-q8-20260619.json`](q8gpu-smollm2-135m-gpu-q8-20260619.json) | **24.6** |
| **Q8_0 · CPU** (pure-Go int8) | [`q8gpu-smollm2-135m-cpu-q8-20260619.json`](q8gpu-smollm2-135m-cpu-q8-20260619.json) | **176.9** |

Two things follow directly:

- **GPU f32 (16.5) is *slower* than GPU Q8 (24.6)** on the identical box. Asking f32 to hit
  100-120 — 4-7× *above* the lighter Q8 precision on the same hardware — is backwards: it
  demands the heaviest precision beat the lightest by a wide margin. The 26.8 tok/s the issue
  reports for f32 CUDA on the RTX 4070 sits in the same launch-serialized regime as these
  numbers (cf. the committed CUDA witness below); it is f32's speed, not a broken kernel.
- **For a 135M model the CPU wins by ~7×** (176.9 vs 24.6). At this size the per-op GPU
  kernel-launch tax (~600 launches/token, host-serialized) dominates the actual compute, so
  the small model is *below the GPU crossover* — the very crossover this directory studies in
  `crossover-qwen2.5-1.5b-{cpu,gpu}-q8-20260619.json`. 100-120 tok/s on SmolLM2-135M is
  reachable — on the **CPU**, where Q8 already clears it at 176.9 — but not by a graph fix on
  the GPU.

For scale, the committed CUDA witness on the RTX 4070 itself — a *3B* model in the lighter Q8
precision — decodes at **25.1 tok/s** at boost clock
([`qwen2.5-3b-q8-cuda-4070.json`](qwen2.5-3b-q8-cuda-4070.json),
[`README.md`](README.md#throughput-from-the-json)). A 135M f32 run at 26.8 tok/s on the same
board is consistent with that launch-bound regime, not anomalous within it.

## What would actually move the number (none of it is a graph fix)

1. **Precision** — quantize f32 → Q8_0/Q4_K. This is the lane's whole point (fak's own native
   Q8 GEMV, `fcuda_matvec_q8`); it cuts the weight bytes ~4× and is the largest single lever.
2. **Device choice / the crossover** — a 135M model belongs on the CPU path (176.9 tok/s),
   not the GPU; the GPU only pays off above the crossover this directory measures.
3. **GPU clock state** — base vs boost is a 2.25× swing on this hardware family, the subject
   of [#379](Q8-CLOCK-RESOLUTION-379.md); verify boost before trusting any low GPU number.
4. **The launch tax** (the ~30% GPU residual to llama.cpp) — addressed by an
   *instantiate-once, replay-many* length-agnostic CUDA graph (the tracked #35/#3 redesign),
   **not** the per-token capture #377 reaches for, which regresses.

## What is genuinely still open (not this issue)

The f32 "baseline broken" framing is closed: `GraphCompile: false` is the intended default,
per-token graph capture is a measured no-win, and f32's throughput is the bandwidth-bound
speed the committed witnesses show — there is no broken kernel here. The genuinely-open,
separately-tracked items are the **instantiate-once/replay-many** length-agnostic graph
(#35/#3 — the real launch-tax lever) and the device prefill batching noted in
[`README.md`](README.md). Neither is "CUDA graph compilation non-functional."

## Reproduce (the committed witnesses this rests on)

```bash
# the three same-box SmolLM2-135M numbers (f32 GPU < Q8 GPU << Q8 CPU):
cat experiments/gpu/q8gpu-smollm2-135m-gpu-f32-20260619.json   # decode.tok_per_sec ~16.5 (f32, GPU)
cat experiments/gpu/q8gpu-smollm2-135m-gpu-q8-20260619.json    # decode.tok_per_sec ~24.6 (Q8,  GPU)
cat experiments/gpu/q8gpu-smollm2-135m-cpu-q8-20260619.json    # decode.tok_per_sec ~176.9 (Q8, CPU)

# the default build reports GraphCompile:false because the graph path is FAK_CUDA_GRAPH-gated:
grep -n 'graphEnabled' internal/compute/cuda.go                # init(): os.Getenv("FAK_CUDA_GRAPH")=="1"
git show 300d745 --stat                                        # #483: graph capture shipped + gated OFF
```
