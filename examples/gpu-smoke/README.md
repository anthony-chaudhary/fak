# GPU smoke walkthrough

This is the adoption-shaped GPU entry point: pick the backend that matches the
machine, run one real-model correctness witness against the CPU reference, then
run a short decode benchmark so the output includes a throughput number.

The smoke needs a HuggingFace safetensors snapshot with `config.json` and
`model.safetensors`. The canonical small model is
`HuggingFaceTB/SmolLM2-135M-Instruct`; set `FAK_GPU_SMOKE_HF` to that snapshot
directory. The scripts also look in the local HuggingFace cache for that model.
Expected runtime: on a configured GPU host, each smoke typically completes in 2-5 minutes
and is deterministic for a fixed snapshot, backend, and token-count setting.

## Decision tree

| Hardware | Command | What a pass means |
|---|---|---|
| AMD Radeon on native Windows | `examples/gpu-smoke/run-vulkan.sh` | Vulkan backend built with `-tags vulkan`; greedy decode is argmax-exact vs CPU reference; decode tok/s is printed. This is numerical parity, not throughput parity. |
| NVIDIA on Linux or WSL | `examples/gpu-smoke/run-cuda.sh` | CUDA backend built with `-tags cuda`; `FAK_CUDA_GRAPH=1` by default; greedy decode is argmax-exact vs CPU reference; decode tok/s is printed. The model must fit the GPU. |
| Neither | Use the default pure-Go CPU path | No GPU dependency: `go run ./cmd/modelbench -hf <snapshot>` or the normal `fak` binary without GPU build tags. |

## Model setup

If the SmolLM2 snapshot is already in the HuggingFace cache, no flag is needed.
Otherwise point the scripts at it:

```bash
export FAK_GPU_SMOKE_HF="$HOME/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/<sha>"
```

`scripts/fetch-model.sh` downloads SmolLM2 and exports fak-format weights for
the normal in-kernel path; the download side usually leaves the HuggingFace
snapshot in the cache path above. `gpucheck` itself needs the original
safetensors snapshot because it compares the device backend against the CPU
reference from the same HF weights.

Optional knobs:

```bash
FAK_GPU_SMOKE_TOKENS=10       # gpucheck greedy tokens
FAK_GPU_SMOKE_STEPS=32        # modelbench decode steps
FAK_GPU_SMOKE_PREFILL=16      # modelbench prefill size
FAK_GPU_SMOKE_DECODE_REPS=1   # keep smoke short by default
FAK_GPU_SMOKE_OUT=.cache/gpu-smoke
```

## AMD / Vulkan

Run from the repo root in Git Bash or another bash that can launch PowerShell:

```bash
examples/gpu-smoke/run-vulkan.sh
```

The Vulkan backend is currently native-Windows only (`//go:build vulkan &&
windows`). The script uses `internal/compute/build_vulkan.ps1 binary` so the
same Vulkan SDK, shader, and cgo environment used by the witness tests builds
`cmd/gpucheck` and `cmd/modelbench`. It sets `FAK_VULKAN_SPIRV` to the compiled
shader directory before running the binaries.

Honest scope: the RX 7600 witness is real and numerically correct, but Vulkan is
not throughput parity. The public results record argmax-exact decode and
prefill-logit cosine 1.0, while decode throughput remains an order-of-magnitude
class gap versus llama.cpp depending on precision and baseline. Treat a Vulkan
smoke pass as "the backend works on this AMD GPU", not "this is fast enough for
production serving".

## NVIDIA / CUDA

Run on Linux or WSL with CUDA and `nvcc` available:

```bash
examples/gpu-smoke/run-cuda.sh
```

The script calls `internal/compute/build_cuda.sh binary` for the two binaries and
defaults `FAK_CUDA_GRAPH=1`. On the documented RTX 4070 SmolLM2 run, that graph
path reaches about 119-120 decode tok/s, matching llama.cpp Q8_0 for that small
model. This does not claim large-model parity: the chosen model must fit the GPU,
and larger f32 checkpoints can exceed RAM or VRAM.

## Source results

- [Vulkan AMD results](../../docs/benchmarks/VULKAN-AMD-RESULTS.md)
- [CUDA GPU results](../../GPU.md)
- [CLAIMS.md Engine section](../../CLAIMS.md#engine)
- [Example output](EXAMPLE-OUTPUT.md)
