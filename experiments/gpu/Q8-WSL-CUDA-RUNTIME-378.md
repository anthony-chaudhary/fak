# Q8 CUDA on WSL2: the missing toolkit runtime is a no-sudo provisioning step (#378)

> **Resolution of [#378](https://github.com/anthony-chaudhary/fak/issues/378)** —
> *"WSL2 missing CUDA runtime libraries for Q8 quantization."* The blocker is real but it
> is **already solved in this tree** by a no-sudo user-space toolkit installer
> ([`internal/compute/setup_cuda_wsl.sh`](../../internal/compute/setup_cuda_wsl.sh)) — none of
> the issue's three "Options to Resolve" are needed. The proof the blocker is cleared is the
> Q8 CUDA benchmark the issue says *cannot run*, already committed in this directory and
> captured **on this exact RTX 4070 / WSL2 box**: [`qwen2.5-3b-q8-cuda-4070.json`](qwen2.5-3b-q8-cuda-4070.json).
> No new measurement is claimed.

## The root cause, stated precisely

WSL2 GPU passthrough mounts the NVIDIA **driver** at `/usr/lib/wsl/lib` — `libcuda.so`
(the driver API), `libnvidia-ml.so`, and `nvidia-smi`. That is the *driver*, not the CUDA
**toolkit**. `libcudart.so` (the CUDA runtime) and `libcublas.so` ship with the toolkit, and
WSL2 does **not** install the toolkit. So `nvidia-smi` sees the RTX 4070 and `libcuda.so`
links, but the `-tags cuda` build — which needs `cudart` + `cublas` to compile and run fak's
own GEMV — has nothing to link against. The issue's symptom (CUDA backend fails to initialize
under `FAK_CUDA_Q8=1`) is exactly this: driver present, toolkit runtime absent.

This is precisely the environment [`setup_cuda_wsl.sh`](../../internal/compute/setup_cuda_wsl.sh)
documents in its header: *"working WSL2 GPU passthrough … and a C compiler, but no system CUDA
toolkit and no passwordless sudo."*

## Why the issue's three options were all heavier than necessary

| Issue option | Why it is not the right resolution here |
|---|---|
| 1. `sudo apt install nvidia-cuda-toolkit` | Needs **sudo**, which this box does not have (no passwordless sudo); installs a system-wide toolkit on a passthrough WSL where one is unnecessary. |
| 2. Pre-quantized GGUF via llama.cpp | **Abandons fak's own kernel.** The whole point of this lane is fak's own Q8 GEMV (`fcuda_matvec_q8`), not a `libllama`/`libggml`/`gguf` path. A GGUF run would not witness fak's engine at all. |
| 3. Run on a native Linux node | Moves off the very box the issue is about; the RTX 4070 Laptop / WSL2 is the target environment, not an obstacle to route around. |

The tree implements a **fourth, lighter** option the issue did not list: a complete
**user-space** CUDA toolkit, no sudo, no system change, consumed by the build with zero edits.

## The implemented resolution (already in the tree)

[`internal/compute/setup_cuda_wsl.sh`](../../internal/compute/setup_cuda_wsl.sh) installs a
self-contained CUDA 12.6 toolkit into `~/cudaenv` via micromamba — **no sudo, no apt, nothing
written outside `$HOME`**:

```bash
micromamba create -y -p ~/cudaenv -c nvidia -c conda-forge \
  cuda-nvcc=12.6 cuda-cudart-dev=12.6 cuda-nvrtc-dev=12.6 libcublas-dev=12.6 cuda-cccl=12.6 \
  cmake ninja make
```

That line provides the two libraries the issue names as missing — `cuda-cudart-dev`
(`libcudart`) and `libcublas-dev` (`libcublas`) — plus the `nvcc` front-end and headers needed
to compile fak's `cuda_kernels.cu`. It then writes `~/cudaenv.env`, which **prepends the toolkit
libs to `LD_LIBRARY_PATH` while keeping the WSL driver path**:

```
LD_LIBRARY_PATH=~/cudaenv/lib:~/cudaenv/targets/x86_64-linux/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
```

So `libcudart`/`libcublas` resolve from `~/cudaenv` and `libcuda.so` (the driver) still resolves
from `/usr/lib/wsl/lib`. The script ends with an on-box witness that compiles and runs an `axpy`
kernel on the GPU and prints `SETUP_OK`.

The build consumes this with no edits: [`internal/compute/build_cuda.sh`](../../internal/compute/build_cuda.sh)
defaults `CUDA_HOME=~/cudaenv` (and falls back to a system `nvcc` on a datacenter image), so the
**same** script builds on WSL, the GPU server, and a GCP GPU VM.

## The number that closes it (committed witness, this box)

The Q8 CUDA run the issue reports as blocked is already in this directory —
[`qwen2.5-3b-q8-cuda-4070.json`](qwen2.5-3b-q8-cuda-4070.json):

| field | value |
|---|---|
| `backend.selected` | `cuda` |
| `backend.tier` | `sm_89` (Ada — RTX 4070 Laptop) |
| `precision` | `Q8_0` |
| `decode.tok_per_sec` | **25.14** (39.77 ms/tok, 128 steps × 5 reps) |

Per [`README.md` §Conditions](README.md#conditions-this-run), that run was captured on the
**RTX 4070 Laptop GPU, 8188 MiB, WSL2**, with the toolchain `~/cudaenv` (no-sudo micromamba) —
the exact box and environment of #378. It also passes the correctness gate: `gpucheck -lean`
is **argmax-exact** vs the CPU-Q8 reference over 12 tokens
([`README.md` §Correctness](README.md#correctness--argmax-exact-the-gate-not-a-vibe)). So Q8
quantization through the CUDA backend on WSL2 is **empirically working**, not a theoretical
maybe — once the toolkit is provisioned by the script above.

## Reproduce (provision once, then run)

```bash
# 1. provision the no-sudo user-space CUDA toolkit (cudart + cublas + nvcc), one time:
bash internal/compute/setup_cuda_wsl.sh        # ends with SETUP_OK
source ~/cudaenv.env                           # puts libcudart/libcublas on LD_LIBRARY_PATH

# 2. build fak's CUDA kernels and run the Q8 witness the issue says is blocked:
bash internal/compute/build_cuda.sh build
FAK_CUDA_Q8=1 go run -tags cuda ./cmd/gpucheck  -hf <qwen2.5-3b-dir> -lean -backend cuda -n 12
FAK_CUDA_Q8=1 go run -tags cuda ./cmd/modelbench -hf <qwen2.5-3b-dir> -lean -backend cuda \
    -decode-steps 128 -decode-reps 5 -decode-prompt 16 \
    -out experiments/gpu/qwen2.5-3b-q8-cuda-4070.json
```

## What is genuinely still open (not this issue)

Nothing for the **runtime-provisioning** blocker of #378: the missing `libcudart`/`libcublas`
are supplied by `setup_cuda_wsl.sh`, the build consumes them, and the committed witness proves
the path runs. The residual items on this box are separate and already tracked: the GPU-clock
state that caps a base-clock run at 9.2 tok/s is [#379](Q8-CLOCK-RESOLUTION-379.md), and the
WSL per-launch tax vs `llama.cpp` Q8 is the capture-safe CUDA-graph lever analyzed in
[`GPU-QWEN-RESULTS.md` §4](../../docs/benchmarks/GPU-QWEN-RESULTS.md). Neither is a missing
toolkit runtime.
