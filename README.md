<p align="center">
  <picture><source media="(prefers-color-scheme: dark)" srcset="visuals/brand/fak-logo.svg"><img src="visuals/brand/fak-logo-ink.svg" alt="fak logo" width="320"></picture>
</p>

# fak

fak is a local coding runtime.
Run a model on your own computer and keep your prompts local. Connect your
coding agent through fak’s gateway to allow reads while blocking writes.

[![fak benchmark sweep: Inkling-2-Mini on RTX 5080 and kernel checks, sorted from strongest wins to the single-stream fence](visuals/61-hero-benchmark-sweep.png)](docs/benchmarks/README.md)

On one RTX 5080 16 GB, fak’s own CUDA engine decodes Inkling-2-Mini at
122.34 tok/s in-process (`kv=f16`, auto context 29,184). Over HTTP at 16K context
(`C4_p2048_d256`), fak reaches 2,017.61 ms median time to first token (TTFT) and
130.44 tok/s aggregate end-to-end throughput, ahead of the `realugbun/llama.cpp`
fork (3,718.34 ms / 122.47 tok/s) and stock `llama.cpp` (4,683.83 ms / 115.69 tok/s),
while single-stream served decode still trails `realugbun` (102.45 vs 118.63 tok/s
at `C1_p512_d128`). Prefill uses 4-bit expert weights and activations (NVFP4 W4A4,
default on an `sm_120f` build from `main`) and reads a 512-token prompt at 5,861 tok/s
against 3,012 for the fork. The [benchmark index](docs/benchmarks/README.md) links the
[Wave-2 RTX 5080 receipt](docs/benchmarks/published/rtx5080-inkling2-v060/RESULTS.md)
and the other comparisons.

A matched HTTP test on one RTX 5080 measured aggregate prefill for uncached
512-token prompts with one output token per request. Fak revision `e7ad0c8dee5`,
run with the fixed benchmark profile, led the realugbun llama.cpp fork, the
faster of the two OSS builds tested, in both halves of the ABBA run.

![Aggregate prompt throughput for Fak and the realugbun fork at 2, 4, and 8 concurrent requests on an RTX 5080. The table below gives the measured rates.](visuals/inkling2-cuda-prefill.svg)

| Concurrent requests | Fak prompt tok/s | Fork prompt tok/s | Smaller half-run speedup |
| ---: | ---: | ---: | ---: |
| 2 | 6,037.586 | 3,337.363 | 1.802× |
| 4 | 6,038.160 | 3,537.808 | 1.675× |
| 8 | 6,033.886 | 3,648.577 | 1.633× |

The rates are pooled medians. The one-token fixture yielded no visible first
token, so this run measured prompt throughput and left visible latency unmeasured.

On Blackwell consumer GPUs (`sm_120`), fak ships native NVFP4 (E2M1 with
block-16 E4M3 scales) and FP8 global attention scratchpads (KV cache) in one
zero-Python Go+CUDA binary. Datacenter Python stacks such as SGLang/FlashInfer
and vLLM already ship FP4 KV. Local consumer runtimes like `llama.cpp` (#26989)
and Ollama still use integer `q4_0`/`iq4_nl` KV.

Hardware-witnessed on one 16 GB RTX 5080 (Inkling-2-Mini 26B hybrid MoE ·
`UD-Q3_K_XL` · `ctx=131,072` · 121k uncached tokens · rev `eb68a8cf8a` · opt-in
`FAK_INKLING2_KV=nvfp4`):

- All 24/24 expert layers stay resident in 16 GB VRAM (0 GiB CPU/PCIe spill vs 12/24 layers spilled on FP16).
- 128k prefill rises 4.21× (283 to 1,190 tok/s) and end-to-end wall time drops 4.10× (451s to 110s).
- Decode rises +21.7% (13.9 to 16.9 tok/s) with exact 128k needle retrieval and 4/4 `llama.cpp` greedy oracle pass.
- Both `nvfp4` and `fp8` KV stay opt-in while stricter needle-retrieval KL gating is in progress.

## Try fak

[Install fak for your platform](INSTALL.md#local-model-setup), then open a
terminal.

On Apple Silicon, `fak up` probes unified memory, picks a model tier, and
opens a chat prompt:

```sh
fak up   # -> opens a local chat prompt
```

On Windows x64 and Linux x64, `fak up` is not available (it is the Apple
Silicon provisioner). Run the small SmolLM2 model in-kernel instead. The first
run downloads it:

```sh
fak run smollm2 "What input breaks def first(xs): return xs[0]? Write a test that catches it."
```

Look for a test that passes an empty list and checks for `IndexError`.

On an M3 or later Mac with 24 GB of memory or more and macOS 26.4 or later,
fak v0.60.0 can also run Inkling-2-Mini on the GPU by name:

```sh
go install github.com/anthony-chaudhary/fak/cmd/fak@v0.60.0
fak run inkling2-mini
```

Or build from a clone of `main`: `git clone --depth 1
https://github.com/anthony-chaudhary/fak.git`, then `go build -o fak ./cmd/fak`
(needs Go 1.26+).

The first run downloads about 14.6 GB. On an Apple Silicon Mac with 24 GB or
more, `fak run` with no model picks Inkling-2-Mini as well. The release
installer downloads the prebuilt Metal archive on Apple Silicon. Smaller Macs and other
platforms get smaller model tiers instead. The [Mac guide](docs/fak/mac-local-models.md#inkling-2-mini-the-first-run-default)
has the requirements and the server commands.

Inkling-2-Mini uses concurrent prefill by default when the CUDA backend supports
native NVFP4 W4A4 and f16 sequence and cohort operations. The admission policy
uses a 25 ms batch gap and a 75 ms batch wait across eligible devices. f16 uses
half the KV bytes of fp32. The server sizes context and checks fit against
available device memory, so capacity and throughput vary by device and workload.
The collection window can add latency to an isolated request.
Use the path to your GGUF file.

```sh
fak serve --engine inkernel --backend cuda \
  --gguf /path/to/Inkling-2-Mini-UD-Q3_K_XL.gguf \
  --addr 127.0.0.1:8080
```

For 128k all-resident context on a 16 GB Blackwell (`sm_120`) GPU, opt into
NVFP4 (or `fp8`) global KV cache:

```sh
FAK_INKLING2_CTX=131072 FAK_INKLING2_KV=nvfp4 \
  fak serve --engine inkernel --backend cuda \
  --gguf /path/to/Inkling-2-Mini-UD-Q3_K_XL.gguf \
  --addr 127.0.0.1:8080
```

Set `FAK_INKLING2_SERVE_BATCH=0` to disable batching. To reproduce the fixed
640-token benchmark context and its settings, use the
[benchmark environment example](examples/inkling2-cuda-prefill.env.example).

For the commands and their main flags, see the [CLI reference](docs/CLI.md).

For build requirements, platform verification and other model sizes, see the
[local model setup guide](GETTING-STARTED.md#local-model-setup).

For the version history, see [CHANGELOG.md](CHANGELOG.md).

[Connect your coding agent](docs/integrations/README.md) · [Hardware benchmarks](docs/benchmarks/README.md) · [Reproduce locally](docs/repro-packet.md#inkling-2-mini-reproducer-cli-only) · [How fak compares with llama.cpp](docs/native-inference-goal.md) · [Documentation](docs/index.md)

[Apache-2.0 license](LICENSE)

<!-- readme-verified: 2026-11-27 vs VERSION 0.60.0 + BENCHMARK-AUTHORITY · appeal-verified: 2026-11-25 · process: tools/readme_freshness_audit.py + tools/doc_appeal_scorecard.py + /refresh-readme -->
