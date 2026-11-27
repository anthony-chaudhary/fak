# The native Inkling Metal path: how it is built and why it is shaped this way

**Audience:** maintainers who have to change, debug or extend the Apple GPU path for Inkling-2-Mini. This page maps the design. It is not a benchmark sheet.

**Evidence labels used on this page.** *Measured* means a number recorded from a run. *Read from code* means the statement comes from reading the sources named next to it and has not been confirmed by a profile or an ablation.

## Summary

- On an Apple GPU, public fak runs Inkling-2-Mini only on its own Metal code: the resident device (`compute.Inkling2Device` implemented over `internal/metalgemm`), with the host-driven Metal forward (`ggufload.OpenInkling2LoopMetal`) as the fallback where the resident device is not linked.
- The resident session and prefill drivers are the same backend-neutral host code in `internal/ggufload` that the CUDA path runs; the backend only supplies the device interfaces (section 2).
- An earlier vendored C++/Metal runtime from `incoai/splash` (Apache-2.0) has been removed from this repo. Why it left, what replaced it, and the last recorded comparison numbers are in [`inkling-metal-vendored-retirement.md`](inkling-metal-vendored-retirement.md). Techniques fak adapted from it carry a provenance comment in the adapting file and are covered by the Splash entry in [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

## 1. The CUDA Inkling path, for contrast

The CUDA path was written in fak from the start, and it is structured as additions to shared code:

- Device kernels are headers in the shared CUDA library. `internal/compute/cuda_kernels.cu` includes five directly: `internal/compute/cuda/inkling2_loop.cuh`, `internal/compute/cuda/inkling2_prefill.cuh`, `internal/compute/cuda/inkling2_dspark.cuh`, `internal/compute/cuda/inkling2_dspark_markov.cuh`, and `internal/compute/cuda/inkling2_verify_rows.cuh`. Two more, `internal/compute/cuda/inkling2_attend_split.cuh` and `internal/compute/cuda/inkling2_expert_gemv.cuh`, come in transitively through the first of those. The entry points are declared in `internal/compute/cuda_backend.h`.
- The Go seam is `internal/compute/cuda_inkling2.go`; the device session is `internal/ggufload/inkling2loop_cuda_device.go`.
- The forward's structure (which layer runs when, loop passes, tree rows) is shared host code in `internal/model/inkling2loop_forward.go`, `internal/model/inkling2loop_passes.go`, `internal/model/inkling2loop_spine.go`, and `internal/model/inkling2loop_tree.go`, which the CPU path also runs.

*Measured:* about 5.4k fak-authored lines; decode climbed from 6.3 to 151 tok/s over 22 commits, with three small cited adaptations from other projects.

| | CUDA path | Former vendored Mac path |
|---|---|---|
| Origin | Written in fak | Vendored |
| Size | about 5.4k lines | about 63k upstream-derived plus 1.7k adapter |
| Shares the backend library with other models | Yes | No |
| Shares the host forward with the CPU path | Yes | No; it had its own forward |
| Time to first fast result | 22 commits | One commit |
| fak engineers can explain each kernel | Yes | No |

Neither column was the right answer by itself. Vendoring reached about 40 tok/s 25 minutes after the first home-grown Metal attempt logged 7–9. The CUDA path took longer and left fak with code it understands and can extend to the next model. The native Metal path follows the CUDA shape.

## 2. The backend-neutral seam

*Read from code.* Neither GPU path calls its backend from the model code. They go through Go interfaces in `internal/compute`:

| Interface | Defined in | Covers |
|---|---|---|
| `Inkling2Device` | `internal/compute/inkling2_device.go` | The op set of the resident decode forward: every op enqueues on the device and nothing returns to the host except through explicit read calls. |
| `Inkling2PrefillDevice` | `internal/compute/inkling2_prefill_device.go` | The same ops batched along the token axis for a prompt chunk. |
| `Inkling2VerifyDevice` | `internal/compute/inkling2_prefill_device.go` | What a speculative verify chunk adds: logged convolution history and the commit that rewinds to the kept prefix. |
| `Inkling2TreeVerifyDevice` | `internal/compute/inkling2_tree_verify_device.go` | Tree-shaped verify of a candidate tree in one forward. |
| `Inkling2DraftDevice` | `internal/compute/inkling2_prefill_device.go` | The DSpark draft's forward. |

The CUDA backend implements them in `internal/compute/cuda_inkling2.go` and `internal/compute/cuda_inkling2_prefill.go`. The Metal backend implements `Inkling2Device`, `Inkling2PrefillDevice`, `Inkling2TreeVerifyDevice` and the DFlash2 draft devices in `internal/compute/metal_inkling2*.go`, over kernels in `internal/metalgemm` (`ink2_*.metal`). The drivers are backend-neutral host code in `internal/ggufload`: `internal/ggufload/inkling2loop_resident.go` opens a resident session on any backend that satisfies `Inkling2Device` and upgrades to batched prefill when it also satisfies `Inkling2PrefillDevice`; `internal/ggufload/inkling2loop_prefill.go`, `internal/ggufload/inkling2loop_spec.go`, and `internal/ggufload/inkling2loop_draft_dev.go` use the prefill, verify, and draft interfaces the same way. An improvement to a shared driver therefore lands on both GPUs.

## 3. Lessons for the next port

These follow from the comparison in section 1 and from the vendored runtime's time in this repo, and apply beyond this model.

1. **Look at what exists before building, in both directions.** Before the first home-grown Metal wiring was written, a runtime that already ran this exact model fast on this exact hardware was available under a permissive license. Before it was vendored, fak's own Metal library already had packed-GGUF kernels, command batching and device KV. As far as the commit history shows, neither was consulted against the other; that is an inference from the order and content of the commits, not a recorded fact. The check is cheap: for each technique the target needs, name the upstream file and the fak file that does it.
2. **Vendoring buys a result; it does not buy understanding.** One commit moved the number from 7–9 to about 40 tok/s. It also added a second Metal stack that shared nothing with the first. Budget the understanding work as part of any vendoring decision, not as optional follow-up.
3. **Index borrowed code by technique, not only by file.** A provenance record that says "275 files from revision X" answers the license question. It does not answer "where is the int8 KV" or "do we already have this". An index keyed by technique, with the upstream location, the fak equivalent and whether they are connected, answers both and makes duplication visible.
4. **The reason a port is slow is usually structural.** fak's first Metal wiring was not slow because its kernels were bad; it was slow because it offloaded matrix products and kept the control flow on the host. The vendored runtime's advantage was where the boundary sat: the whole cycle on the device, one wait. That lesson transferred to `internal/metalgemm` without copying the runtime.
5. **Note what was left behind.** Upstream's serving engine was never vendored, and with it went batching, prefix caching and KV tiering policy. When borrowing part of a system, record which capabilities lived in the part that was dropped.
6. **Keep the fast path as the bar while replacing it.** A correct, fast reference on real hardware is what every replacement is measured against on the same file and prompt. Moving that reference out of the shipped product (into a separate comparison build) keeps the bar without shipping it.
7. **A calibration table is data, not code.** Kernel-selection constants measured on someone else's devices need their own check on local hardware; byte-identity with upstream says nothing about whether they are right here.
