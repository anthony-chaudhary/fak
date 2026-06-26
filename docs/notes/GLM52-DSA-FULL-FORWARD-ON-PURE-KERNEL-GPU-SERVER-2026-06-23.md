---
title: "GLM-5.2's full DSA forward runs on the pure fak kernel on the sm_80 GPU server (2026-06-23)"
description: "Fresh on-hardware capture: GLM-5.2's complete DSA forward — the dense projections on k_q8_gemm AND the DSA sparse-attention compute on k_dsa_sparse_attend — now executes on fak's own CUDA kernel on a real sm_80 datacenter GPU, cosine = 1.000000 vs the CPU Q8 reference, argmax-exact, at origin/main HEAD 44aa3b6. Closes the sm_80 sparse-attention capture gap (previously only witnessed on sm_89). The full-size 753B llama.cpp CPU-offload serve is recorded strictly as the comparison baseline, not fak."
---

# GLM-5.2: the full DSA forward on the pure fak kernel, witnessed on the sm_80 GPU server (2026-06-23)

> **Goal:** prove GLM-5.2 runs on fak's **own** kernel on the real datacenter GPU server —
> *pure fak*, not a third-party engine. This note records the on-device witness (grounded
> in the node's own `go test -tags cuda` exit code and run log, **not** self-report) and is
> explicit about the boundary: the kernel-math forward is proven on real **sm_80** silicon;
> full-size 753B *serving* still routes through llama.cpp (the labeled comparison), because
> fak's native engine is f32 / no quantized-GGUF device GEMM today.

> **Superseded by progress (#917; see the [staged plan](native-753b-track-staged-plan.md)).**
> The "753B serving still routes through llama.cpp … no quantized-GGUF device GEMM" boundary
> above was the 2026-06-23 snapshot; the quantized device GEMM + `--cpu-offload-experts` rungs
> have since landed and fak's own engine loads the full 466 GB model natively
> ([2026-06-25 native-serve note](GLM52-FAK-NATIVE-SERVE-LOAD-SPEED-2026-06-25.md)). The
> sm_80 DSA-forward witness this note records stands; the serving boundary does not.

## The headline: the full DSA forward on sm_80, cosine = 1.000000

Until today the on-hardware GLM-5.2 pure-kernel captures were split across two arches:

- the **dense** projections (`k_q8_gemm`) on **sm_80**, committed `498a4ab`;
- the **sparse-attention** compute (`k_dsa_sparse_attend`) only on **sm_89** (2026-06-23).

This run closes that gap. The **complete** GLM-MoE-DSA forward — MoE/FFN experts + router +
vocab head, the DSA attention dense projections on `k_q8_gemm`, **and the DSA
sparse-attention compute itself on `k_dsa_sparse_attend`** — now runs on fak's CUDA kernel
on a real **sm_80** datacenter GPU, **argmax-exact, cosine = 1.000000** vs the all-host CPU
Q8 forward. The node's own run log (fresh clone of `origin/main` → `nvcc -arch=sm_80` build
→ isolated `-tags cuda` test):

```
=== HEAD 44aa3b6 ===
=== build libfakcuda.a (sm_80) ===
[cuda] nvcc compile kernels (sm_80) ...
[cuda] go build -tags cuda ./internal/compute/ ... OK build
=== go test -tags cuda -run TestCUDAGLMMoeDsaBackendForward ./internal/model/ -v ===
=== RUN   TestCUDAGLMMoeDsaBackendForward
    glm_dsa_cuda_test.go:63: GLM-MoE-DSA forward with MoE/FFN+head + DSA attention
      projections (k_q8_gemm) + DSA sparse attention (k_dsa_sparse_attend) on cuda
      backend: cosine=1.000000 argmax cpu=40 cuda=40 tier=sm_80 class=approx
--- PASS: TestCUDAGLMMoeDsaBackendForward (0.16s)
PASS
ok  github.com/anthony-chaudhary/fak/internal/model   0.853s
=== GLM GPU WITNESS DONE rc=0 ===
```

The hard `compute.DSASparseBackend` type-assert inside the test **fails the test rather than
silently falling back to the host loop** — so its PASS confirms the sparse attention
executed on the device kernel, not host. Reproduce on any sm_80+ CUDA node:

```bash
bash tools/dgx_glm_gpu_witness.sh    # clone origin/main -> nvcc -arch=sm_80 -> the isolated witness
```

## The honest boundary: this is kernel-math, not 753B serving

The witness runs a **tiny GLM-DSA fixture** (kernel correctness), not the 753B weights. So
what is proven is precise and bounded: *fak's own kernel computes GLM-5.2's DSA forward
bit-faithfully on real sm_80 silicon.* It is **not** "fak serves the full 753B." fak's
native engine is f32 / has no quantized-GGUF device GEMM / no multi-GPU NCCL today, so it
cannot load the 753B Q4 checkpoint — that is the labeled gap, not a claim.

## The comparison baseline (llama.cpp), and why it was brought down

Full-size GLM-5.2 **does** serve on the same sm_80 server today — via **llama.cpp** with CPU
offload (`-ngl 99 --n-cpu-moe 99`): a community Q4_K_M GGUF (≈424 GB) resident in host RAM
with the MoE experts on CPU and the dense/attention layers on the GPUs, answering on an
OpenAI-compatible port at **~2.66 tok/s** decode (~5.1 tok/s prompt). This confirms the
**CPU-offload path makes the full 753B serveable on sm_80** — no Hopper-class DSA kernel
required, exactly as expected — but the **engine is llama.cpp, not fak**, so it is recorded
here strictly as the comparison baseline. (Like other reasoning models on a tight token
budget, it tends to spend the budget on `reasoning_content` before the literal answer — a
harness caveat, not a serving failure.)

**Provenance audit** (so the comparison server is never mistaken for a fak dependency): it
was a **manually-started, detached process** — parent = `init(1)`, with **no
systemd/cron/supervisor**, so it does not auto-restart — serving the community GGUF from a
local llama.cpp build. **No fak process referenced it.** To give the pure-fak witness a
fully idle machine, it was brought down cleanly (all GPUs returned to `0 MiB` used); the
exact one-line relaunch command was recorded first, so the comparison is restartable on
demand.

## What is proven vs not (labeled)

- **Proven on real sm_80 hardware today (`origin/main` HEAD `44aa3b6`):** GLM-5.2's full DSA
  forward — dense projections (`k_q8_gemm`) **+** sparse-attention compute
  (`k_dsa_sparse_attend`) — on fak's own CUDA kernel, **cosine = 1.000000**, argmax-exact
  (`cpu=40 cuda=40`, `tier=sm_80`), `TestCUDAGLMMoeDsaBackendForward` PASS; the
  `DSASparseBackend` assert confirms the sparse path ran on-device.
- **Also proven (prior):** the same full forward on **sm_89**, 2026-06-23; the dense slice on
  **sm_80**, `498a4ab`. Both architectures now carry the complete forward.
- **Comparison only (NOT fak):** the full 753B GLM-5.2 (Q4_K_M) serving via llama.cpp CPU
  offload, ~2.66 tok/s.
- **Out of scope / labeled gap:** fak's native engine serving the full 753B (needs a
  quantized-GGUF device GEMM + multi-GPU sharding); a device-resident DSA *selection*
  (index-score + top-k) kernel.

See also [`GLM52-DSA-SPARSE-ATTENTION-ON-PURE-KERNEL-2026-06-23.md`](GLM52-DSA-SPARSE-ATTENTION-ON-PURE-KERNEL-2026-06-23.md)
(the sparse-attention seam + the sm_89 capture) and
[`GLM52-DSA-PROJECTIONS-ON-PURE-KERNEL-GPU-SERVER-2026-06-22.md`](GLM52-DSA-PROJECTIONS-ON-PURE-KERNEL-GPU-SERVER-2026-06-22.md)
(the dense-projection slice + the original sm_80 dense capture).
