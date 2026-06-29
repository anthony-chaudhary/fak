---
title: "On-device M3 Pro parity witness (2026-06-29): Metal correct (cosine 1.0) but 27B q4_k still launch-bound (~0.10× of llama.cpp-Metal)"
description: "A fresh, this-session on-device re-witness of fak's Metal path on the Apple M3 Pro bench node, driven over Tailscale SSH from the Windows agent-host. Re-confirms kernel-level correctness parity (q4k GEMV cosine 1.000000, greedy decode token-exact) and re-measures the 27B q4_k throughput gap on a clean GPU (decode 0.70 tok/s, prefill 5.59 tok/s) — both ~0.10× of the cited llama.cpp-Metal bars. Also records the #62 build-tag retirement confirmed on-device, and a trunk build hole that blocks the fak serve resident-decode path."
date: 2026-06-29
---

# On-device M3 Pro parity witness (2026-06-29)

Driven over Tailscale SSH from the `windows/amd64` agent-host onto the **Apple M3 Pro**
bench node (`node-macos-a`, 18-core Metal GPU, 36 GB unified, macOS 26.5). Source fetched
from public `origin/main` and built at commit `4fb75496` (the build-tag retirement +
runtime auto-select are in this tree); `CGO_ENABLED=1`, Metal linked automatically with
**no `-tags fakmetal`** (the #62 build-tag flip, re-confirmed below). GPU was clean —
no llama-server / lmstudio / fak serve resident, ~27 GB free.

## 1. Correctness parity — PROVEN on-device (kernel level)

`go test ./internal/metalgemm ./internal/model` (Metal, real GPU): both **PASS**.

| witness | result |
|---|---|
| `TestMetalQ4KGemvMatchesCPU` | q4k GEMV **cosine = 1.000000**, maxRel ≤ 1.2e-6 at [256,256], [512,1024], [5120,5120] |
| `TestMetalQ4KDecodeMatchesCPU` | greedy decode token sequence **bit-identical to CPU** = `[433 92 166 106]` |
| `internal/metalgemm` suite | `ok` (MatMul/Forward/Reset vs f32 reference) |

The Metal kernels compute the same math as the CPU reference on real Apple Silicon. This is
the architecture/kernel-level correctness gate, re-witnessed this session.

## 2. Speed — NOT at parity for the 27B q4_k headline (re-measured, clean GPU)

`modelbench -gguf Qwen3.6-27B.q4_k_m.gguf -q4k -metal` (resident Q4_K/Q8 hybrid + MetalQ4K;
`-decode-steps 16 -decode-reps 1 -decode-prompt 8`, `-prefill-sizes 32 -prefill-reps 1`):

| phase | fak-Metal (this run) | llama.cpp-Metal bar (cited) | ratio |
|---|---:|---:|---:|
| decode | **0.70 tok/s** (1438 ms/token) | 7.29 tok/s | **~0.10×** |
| prefill | **5.59 tok/s** @P32 | 51.55 tok/s @pp22 | **~0.11×** |

Resident footprint: Q4_K 184 tensors / 7.94 GiB, Q8 313 / 11.60 GiB, f32 354 / 4.86 GiB,
total **24.4 GiB**; decode reads **19.09 GiB/token**. At the device's ~150 GB/s that is a
~127 ms/token bandwidth floor (~7.9 tok/s ceiling), so the measured 1438 ms/token leaves
~90% of the wall in per-call command-buffer launch overhead — the documented wall, not a
wrong kernel. The closing lever (the one-command-buffer GDN-hybrid resident decode forward)
is Mac-gated `.m` engineering tracked by #64/#67/#69/#70/#71.

## 2b. Dense Qwen2.5-7B-Q8 — decode at near-parity (measured this session)

`modelbench -gguf qwen2.5-7b-instruct-q8_0 -metal -lean` with **`FAK_METAL_MPS=1`**
(without it the box reports *"MPS unavailable … the f16 prefill is disabled"* and the full
prefill bench emits nothing — the f16 GPU GEMM goes through MetalPerformanceShaders, which
is off by default here). Engine: *"fak-in-kernel Metal prefill (MPS f16 GEMM on GPU; CPU Q8
decode)"*. Decode was stable across two runs (16.0, 16.4 tok/s).

| phase | fak (this run) | llama.cpp-Metal bar (cited) | ratio |
|---|---:|---:|---:|
| decode | **16.4 tok/s** (61.1 ms/token, 48 steps × 5 reps) | 17.27 tok/s | **0.95×** |
| prefill @P256 | **94.4 tok/s** (2712 ms) | 192.9 tok/s | **0.49×** |

So for the dense-Q8 parity-class backend, **decode is essentially at parity (0.95×)** and
Metal f16 prefill is ~6× faster than the old CPU prefill (0.083× → 0.49×).

> **Honesty flag — do NOT overwrite the authority on this run alone.** The committed
> `QWEN25-7B-RESULTS.md` / `BENCHMARK-AUTHORITY.md` figure is fak **CPU-Q8 decode 8.7 tok/s
> (0.50×)**. This session's 16.4 tok/s is ~**1.9×** higher on the *same* CPU-Q8 decode path
> (engine label confirms decode is CPU, not GPU-resident). The most likely cause is the
> parallel Q8-decode workers (this run: 6 workers, GOMAXPROCS=12) vs the older measurement,
> but that is **not reconciled here** — until it is (re-measure at workers=1; confirm the
> older artifact's config), the conservative committed 0.50× stands and 0.95× is a
> *pending, unratified* datum. This is the BENCHMARK-AUTHORITY discipline: a single
> favorable run does not move the number of record.

## 3. #62 build-tag retirement — confirmed on-device

`fakchat -metal` help on the freshly built binary reads *"requires Apple Silicon+cgo"*
(previously *"requires -tags fakmetal"*), and `CGO_ENABLED=1 go build` linked the Metal
backend with no special tag. The build-tag half of #62 (`881b7daf`) and the runtime
auto-select (`dfe9de9b`) are both live in the built tree.

## 4. Honest fences

- Single-rep modelbench decode (0.70 tok/s) is an order-of-magnitude datum, not a
  multi-rep authority row; it is consistent with the 64-token fakchat real-generation
  figure of **1.2 tok/s** (`MAC-QWEN36-27B-Q4K-METAL-PERF-DIAGNOSIS-2026-06-26.md`) — both
  put 27B q4_k decode well under 0.16× of the bar.
- The llama.cpp-Metal **7.29 / 51.55** bars are cited (committed reference
  `QWEN36-PARITY-RESULTS.md`), not re-measured here.
- The dense-7B-Q8 numbers in §2b are the **`modelbench` path** (Metal MPS f16 prefill +
  **CPU** Q8 decode). The **GPU-resident** Q8 decode (#67, the ~0.99× claim) runs through
  `fak serve`, which does not build from clean trunk right now (§5) — so the §2b decode is
  the CPU-Q8 path at 0.95×, not the GPU-resident path. Both are decode of the same weights;
  the resident-vs-CPU distinction is why the §2b honesty flag matters.

## 5. Trunk build hole (blocks the fak serve resident-decode path)

A clean `CGO_ENABLED=1 go build ./cmd/fak` at `origin/main` **fails**: committed call sites
in `cmd/fak/main.go` (and siblings) reference functions whose definition files are
**untracked** and were never committed — `chatrelay.go` (`cmdChatRelay`) + `internal/chatrelay/`,
`audit_diagnose.go` (`cmdAuditDiagnose`), `claude_mac_split.go`, `guard_child.go`,
`guard_provider.go`. It is masked on any working tree that already holds those untracked
files (e.g. the box that authored them), but every fresh clone — including this Mac node —
fails to build `cmd/fak`. This needs the authoring sessions to commit the missing definition
files; it is multi-session WIP, not swept here.
