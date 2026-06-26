---
title: "fak in-kernel CUDA serve on an L4: first live serve tok/s, and a concurrency crash fixed"
description: "The first warm steady-state decode tok/s measured against fak's LIVE in-kernel CUDA serve (a real Q8 0.5B checkpoint on a datacenter L4), and the single-stream-serialization bug a concurrent request burst exposed on the live serve — root-caused, fixed, and covered by a proven GPU-free test."
---

# fak in-kernel CUDA serve on an L4: live tok/s + a concurrency crash fixed

_2026-06-25._ A companion to
[DGX2-CROSS-ENGINE-DATA](DGX2-CROSS-ENGINE-DATA-2026-06-25.md) and
[GLM52-NATIVE-THROUGHPUT-AND-BENCHMARK-PLAN](GLM52-NATIVE-THROUGHPUT-AND-BENCHMARK-PLAN-2026-06-25.md).
Those notes record native-kernel and 753B-serving work on the DGX-class A100 nodes. This one
records what could be measured **right now, on the one GPU node actually reachable** — the
`fak-realmodel` GCP `g2-standard-8` (a single **NVIDIA L4**, sm_89, 23 GB) that already runs
fak's in-kernel CUDA serve live — and the real serving bug that live traffic exposed there.

> **Scope & honesty.** The number below is fak's **own in-kernel CUDA forward** decoding a
> **real Q8_0 0.5B checkpoint** through the live `/v1/messages` serve. It is a *small-model
> device-decode serving rate*, not a 753B rate, not a DGX2/A100 number, and not a stock-engine
> comparison. The DGX2/DGX3 A100×8 nodes were reachable only through a control-bridge that is
> down; the L4 is the reachable hardware, and one GPU is all the pure kernel needs.

## 1. The live serve

`fak serve --gguf qwen2.5-0.5b-instruct-q8_0.gguf --backend cuda` runs in the `fak-gpu`
container on the L4, answering on `:8082`. `GET /healthz` returns
`{"engine":"inkernel","model":"qwen2.5-0.5b-gpu","ok":true}` and `/v1/messages` returns real
assistant text — this is a genuine in-kernel device decode, not a proxy.

## 2. First warm decode tok/s (single stream)

**fak in-kernel CUDA serve · real Qwen2.5-0.5B Q8_0 · NVIDIA L4 (sm_89) · 128-token completions:**

| metric | value |
|---|---:|
| warm single-stream decode | **466 tok/s** (median of 8; range 437–510) |
| latency (128 tok) | ~0.27 s |
| resident VRAM | ~2.25 GB of 23 GB |

**Warmup caveat (do not misquote):** the *first* decode after a cold container start measures
~66 tok/s — it carries the CUDA-context + first-graph warmup. The **warm steady-state** rate is
~466 tok/s; that is the number to quote, with the warmup discarded.

This is consistent with [GPU-QWEN-RESULTS](../benchmarks/GPU-QWEN-RESULTS.md)'s central finding:
the laptop Q8 numbers (11–15 tok/s) were **WSL-launch-bound**, not kernel-bound. On a
native-Linux datacenter L4 with no per-call launch tax, the same Q8 device path decodes a
0.5B model an order of magnitude faster.

## 3. The bug live traffic exposed: the device serve was not concurrency-safe

A 2-way (and 4-way) concurrent `/v1/messages` burst **took the GPU serve down**: one stream
dropped the connection, the other returned garbage (`!!!!`), and the CUDA context was poisoned
with thousands of sticky `fak-cuda: cuda_kernels.cu:81 an illegal memory access was
encountered` errors that propagated to **every later request** until the container was
restarted. The sibling **CPU** serve (`:8081`, same model, no backend) answered correctly
throughout — isolating the fault to the CUDA backend path.

**Root cause.** The CUDA backend is single-stream by construction (one `g_stream`, one cuBLAS
handle, a shared size-bucketed device free-list); its Go-side `cudaMu` makes each *individual*
op atomic but not a whole multi-op forward. The gateway drives `agent.InKernelPlanner.Complete`
concurrently, and on the device path that function held **no** serializing lock across the
forward (the existing `p.mu` guards only the radix prefix tree, which is a CPU-session path that
never engages with a backend wired). Two concurrent forwards interleaved their per-token op
sequences on the shared device state and stomped each other's activation/KV buffers → an
out-of-bounds kernel access → poisoned context.

**Fix** (`internal/agent/inkernel_planner.go`). A dedicated `devMu` held across the entire
forward when `backend != nil`, serializing concurrent device requests into safe queuing —
correct for a single-stream device. The CPU path (`backend == nil`) is untouched (the hold is
gated on the backend), so the radix-reuse / CPU-session behavior is byte-for-byte unchanged.
Batched multi-user device decode (real concurrent throughput, not just safety) is the separate
follow-up (`internal/model/batch.go`), not a correctness fix.

**Test** (`internal/agent/inkernel_planner_concurrency_test.go`). A GPU-free paired honesty
test wraps the CPU reference backend in an overlap detector and drives 4 concurrent `Complete`
calls through a tiny synthetic model. It is **proven non-vacuous**: without the lock the four
forwards overlapped **3,719 times** (the live crash, made deterministic); with it, **0**
overlaps, race-clean, every call decodes its tokens. A second test confirms the CPU path still
completes (the fix never engages there).

## 4. Status / next steps

- **Shipped to the tree:** the `devMu` serialization fix + the proven concurrency test.
- **Not yet on the live serve:** the running `fak-gpu` container is the 2026-06-21 image (it
  predates the fix). The single-stream number in §2 is unaffected by the fix; redeploying the
  CUDA image (its build pipeline is out-of-band of this repo) is what carries the concurrency
  safety to the live box. Until then, the live serve must be driven one request at a time.
- **Follow-up — real concurrent throughput.** Serialization makes concurrency *safe*; batched
  device decode (`internal/model/batch.go`, the Q8 tile-GEMM lane) is what would make it *fast*.
  That is the L4 analogue of the DGX "throughput@concurrency" rung.

## 5. Reproduce

```sh
# Against the live serve (single stream, warm — discard the first rep):
curl -s http://127.0.0.1:8082/v1/messages -H 'content-type: application/json' \
  -d '{"model":"qwen2.5-0.5b-gpu","max_tokens":128,"messages":[{"role":"user","content":"..."}]}'

# The concurrency fix + its proof (no GPU needed):
go test ./internal/agent/ -run TestInKernelConcurrentDeviceCompleteSerializes -race -count=1 -v
```
