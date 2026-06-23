---
title: "SWE-bench Verified on the pure fak kernel — the GPU-server runbook"
description: "The exact, end-to-end command sequence to resolve SWE-bench Verified instances with a coding model served by fak's native CUDA engine (no SGLang in the path), and the honest residual that only a GPU run can close."
---

# SWE-bench Verified on the pure fak kernel — runbook

> **What this is.** The executable path to a *pure-kernel* SWE-bench Verified solve: a
> coding model served by fak's **own** CUDA forward pass (`fak serve --gguf --engine
> inkernel --backend cuda`), driven by fak's **own** coding agent (`fak swebench run
> --agent fleet`), graded by the **official** SWE-bench harness. No SGLang, no vLLM, no
> external agent in the loop — the differentiator the [Qwen3.6-27B results](QWEN36-27B-GPU-SERVER-RESULTS.md)
> deliberately do *not* claim (that run is SGLang-serves + fak-adjudicates).
>
> **Status: the path is ASSEMBLED in code and the harness is witnessed; the resolve-rate
> is the GPU residual.** Every step below runs on the GPU server; the only thing this dev
> box cannot produce is the number itself (no GPU here). Nothing in this doc invents a
> resolve-rate.

---

## 1. What is shipped vs. what the GPU run still owns

The kernel levers and the harness are landed and tested; the remaining work is to *run*
them on a GPU and record the number.

| Piece | State | Evidence |
|---|---|---|
| Native quant GEMM (Q4_K/Q8_0), fp16 HGEMM, fused flash attention | **built + GPU-gated** | `internal/compute/cuda.go:291` `Caps{UploadDtype:true, FusedAttn:true}`; gates `cuda_quant_test.go` / `cuda_fp16_test.go` / `cuda_flash_test.go`; [parity tracker §3.1](../notes/gpu-parity-tracking-480.md) |
| Q4_K GGUF load + device-resident path (no f32 materialization) | **landed** | `internal/ggufload/quant_q4k_loader.go` (`AddResidentQ4K`); `fak serve --gguf` + `FAK_Q4K=1` |
| In-kernel serving from the native engine | **landed** | `fak serve --gguf --engine inkernel --backend cuda` (no `--base-url` ⇒ the model serves from fak's own decode) |
| fak-native coding agent (the "fleet" runner) | **landed + witnessed (no model)** | `internal/swebench/fleet.go`; `fleet_test.go` proves the loop mechanics on a temp git repo |
| Coding policy that gates the agent's tools | **landed + validated** | `examples/swebench-coding-agent-policy.json` (`fak serve … --policy-check` OK) |
| Official-harness grading | **landed (Docker-gated)** | `fak swebench eval` prints the exact `python -m swebench.harness.run_evaluation` command |
| **Resolve-rate on the pure kernel** | **`pending GPU run`** | this runbook |

The honest fence: the device kernels are proven *argmax-exact against the CPU reference*
(`cuda_test.go`), and the agent loop is proven mechanically — but **no real model has yet
solved an instance through this path.** Do not claim "the pure kernel solves SWE-bench"
until step 5 below returns a resolve-rate > 0 with `--engine inkernel` and no proxy.

---

## 2. Prerequisites on the GPU server

1. A CUDA toolkit + a reachable GPU. Build for the card's arch via `FAK_CUDA_ARCH`
   (e.g. `sm_90`); the build fails loud if `--backend cuda` is named but no GPU is reachable,
   so a typo never silently runs on CPU.
2. A coding model as a **Q4_K GGUF** with an embedded tokenizer. Qwen2.5-Coder-7B-Instruct
   at `q4_k_m` is the obvious pick — the GGUF loader + the Qwen BPE tokenizer are oracle-proven
   (`internal/tokenizer/oracle_qwen_test.go`), and the Q4_K weight is ≈4.7 GB resident, well
   inside a single datacenter GPU.
3. Docker + the `swebench` Python module on the grading box (for step 5).

## 3. First, prove the device kernels (a skip is not a pass)

These exit non-zero on a SKIP, so a green run is real device evidence:

```bash
bash tools/run_485_acceptance_on_gpu.sh   # Q8_0/Q4_K device GEMM cosine + VRAM witness
bash tools/run_484_acceptance_on_gpu.sh   # fp16 HGEMM cosine
bash tools/run_486_acceptance_on_gpu.sh   # fused flash attention cosine
```

## 4. Serve the model from the pure kernel, then solve

Serve from fak's own engine (note: **no `--base-url`** — that is what makes it pure-kernel;
`FAK_Q4K=1` selects the direct-resident-Q4_K path, and `--backend cuda` runs prefill+decode
through the GPU HAL):

```bash
FAK_Q4K=1 fak serve \
  --gguf /srv/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
  --engine inkernel --backend cuda \
  --addr 127.0.0.1:8080 \
  --policy examples/swebench-coding-agent-policy.json
```

Drive the fak coding agent against it on a smoke slice. `--allow-exec` enables the agent's
`bash` tool (the policy's `bash.command` deny-regex guards — `rm -rf`, `sudo`, `curl|sh`,
`git push` — bind to it, and the gateway drops any denied call before the agent executes it):

```bash
fak swebench run --agent fleet \
  --gateway 127.0.0.1:8080 \
  --model qwen2.5-coder-7b-instruct \
  --filter smoke \
  --difficulty <bench>/data/swebench_verified_difficulty.json \
  --allow-exec \
  --output run-pure-kernel
```

`run-pure-kernel/predictions.json` is a real `git diff` per instance, in the exact shape the
official harness applies. (Per-instance, the fleet runner clones the repo at `base_commit`,
runs the read/edit/bash loop to `--max-steps`, and captures the diff.)

## 5. Grade with the official harness

```bash
fak swebench eval --predictions run-pure-kernel/predictions.json
```

On a box with Docker + the `swebench` module this produces the resolve-rate directly;
elsewhere it prints the exact `python -m swebench.harness.run_evaluation …` command to run
on the grading box. **That number, with `--engine inkernel` and no proxy in the path, is the
pure-kernel proof.**

## 6. The smallest honest win, and the operational long pole

- **Smallest provable slice.** One instance, end to end, producing a *real* (non-empty) diff
  from fak's own forward pass — even a non-resolving patch the harness *attempts to apply*
  proves the path works. Resolve-rate > 0 is the second milestone; full SWE-bench Verified
  is the goal.
- **The long pole is operational, not algorithmic.** Reaching the GPU server today goes
  through the Slack control bridge, whose transcript readback (`!dump`) is unreliable
  (tracked in private lab tooling). A direct shell, or fixing that readback, is the
  highest-risk GPU-server item — everything in steps 3–5 is blocked on *observing* the run, not on
  the kernel.

---

## 7. Provenance

- Serving flags: `cmd/fak/main.go` (`--gguf` / `--engine` / `--backend` / `FAK_Q4K`).
- Coding agent: `internal/swebench/fleet.go` (+ `fleet_test.go`); the gateway adjudication
  proxy that drops denied calls: `internal/gateway/http.go`.
- Policy: `examples/swebench-coding-agent-policy.json` (validated with `fak serve --policy-check`).
- Kernel levers + acceptance gates: [`gpu-parity-tracking-480.md`](../notes/gpu-parity-tracking-480.md);
  `tools/run_48{4,5,6}_acceptance_on_gpu.sh`.
- Written on a host with no GPU; the resolve-rate cell is `pending GPU run` by design.
