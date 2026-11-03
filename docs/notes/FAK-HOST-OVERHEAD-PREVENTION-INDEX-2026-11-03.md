# FAK host-overhead prevention index (2026-11-03)

One read for the whole detection stack that keeps **small-batch decode launch-bound**
on fak's own engine. Every row below is a landed improvement against the issue class
established by the originating evidence:

- **[GLM-5.2 decode: the path to 10 tok/s on the pure fak kernel](GLM52-DECODE-PATH-TO-10-TOKS-2026-06-27.md)** —
  localizes the real 753B cpu-offload serve's <0.1 tok/s decode to structural device glue
  (~1100 synchronous device↔host round-trips/token), quantifies the host expert-kernel
  ceiling, and names the levers. That note is *prose measured once*; the ten items below
  convert it into a version-controlled, recomputable, gated detection stack.

## The ten landed improvements

| # | Verb / mechanism | Repo | SHA | What it detects / prevents | Rationale |
|---|---|---|---|---|---|
| 1 | `fak-dev hostshare` | fak (public) | `c5c4c193f` | Reads `host_share` (host-phase share of decode-step time) from perf-receipt-v2 JSON + profile notes; gates launch-bound decode at `--threshold` (default 0.30) | Turns single prose readings (PHL-051 host_share 0.130; PHL-127 GPU busy split) into a re-runnable gate over committed receipts; a new launch-bound regression can no longer stay invisible until it surfaces as tok/s loss |
| 2 | `fak-dev regime` | fak (public) | `c4b76756` | Multi-signal decode-regime classifier (launch / compute / mem / unknown) from receipt signals | Prevents attributing a decode slowdown to the wrong lever (kernel vs host vs memory); gives every other gate in this table a shared, typed regime vocabulary instead of free-text |
| 3 | `fak-dev glue-census` | fak (public) | `5510d8a5c` | Verified sync round-trips/token census for the GLM-DSA decode path; go/ast verification against live source, vanished/added seam = DRIFT (exit 1) | The ~1100 round-trips/token figure becomes recomputable after any refactor; a NEW sync site added to the decode path fails the run instead of silently rotting the metric |
| 4 | `fak-dev graph-audit` | fak (public) | `1ae9aa785` | Presence-vs-invokability audit of CUDA/Vulkan graph capture/replay on decode paths; classifies `PER_LAYER_HOST_SETUP` paths; `--fail-on uncovered` | Per-layer host-setup paths that lose graph coverage (or were never capturable) are named and gated; converts "graphs exist somewhere" into a per-path coverage table with a committed snapshot |
| 5 | perf-receipt v3 host-overhead schema | fak (public) | `694a0c323` | Native capture of the host-overhead decomposition (`host_phase_us`, kernel-launch count, …) inside perf-receipt-v3 receipts | Host-share data stops living in ad-hoc prose: every future hardware receipt carries the decomposition natively, which is what rows 1, 2, and 6 consume |
| 6 | `fak-dev hostregress` | fak (public) | `b8b5757b` | Cross-receipt host-overhead regression trend gate: ≥50% relative delta vs oldest-K baseline or a regime flip fails | Catches SLOW host-overhead drift that no single receipt flags — the trend itself is the defect, and the gate makes the trend a first-class CI input |
| 7 | `fak-dev h2d-inventory` | fak (public) | `cc9316efe` | Per-token host→device copy inventory on decode state paths (sampler / mask / KV-append metadata): copies/token + bytes/token by model shape + source-drift gate; zero-copy witness rows | A NEW H2D memcpy on a state path shows up as drift or an uncovered marker; converting a zero-copy seam into a memcpy regresses loudly instead of quietly adding launch latency |
| 8 | `fak-dev launch-census` | fak (public) | `59df3efc1` | Launches/token census of decode-path kernel-launch / parFor sites (looped vs batched MoE expert dispatch), pinned to file+function+pattern with drift gate | The "~24 tiny parFors/layer" hand-wave becomes a computed delta between dispatch variants; a batching change that retires looped rows drifts loudly, and a new dispatch site is discoverable |
| 9 | bringup-preflight (day-1 host-share/regime gate) | fak-private (companion) | `78baf5741` | Day-1 L3 preflight executes `fak-dev hostshare` + `fak-dev regime`, folds exit codes into a fail-closed verdict (LAUNCH_BOUND or host-share breach = FAIL; missing tooling = unblock event unless waived), appends an HOP ledger row | A model cannot be brought up without a deterministic host-overhead regime reading; months-later prose notes are replaced by a day-1 recorded verdict |
| 10 | `fak-dev hostsetup-bench` | fak (public) | this commit | Direct per-layer HOST SETUP vs KERNEL microbench at batch 1 on the compute `Planner` seam (`PrepareMatMul` vs `MatMulPlan.Run`): reports `setup_us_per_layer`, `kernel_us_per_layer`, `host_fraction`; gate `host_fraction > 0.30 => LAUNCH_SETUP_DOMINANT` (exit 1 under `--gate`). CPU fallback labeled `[SW-VERIFIED]`; CUDA wired via the backend registry and skipped with a reported reason when unavailable | Rows 1–9 all INFER the host side (receipt fractions, counted round-trips, counted dispatch sites). This is the first DIRECT measurement of the setup-vs-kernel ratio for identical math kernels, so the launch-bound verdict rests on a measured number with a regression gate, not on inference |

## How the stack layers

```
direct measurement      hostsetup-bench   (this row) — setup/kernel ratio on the Planner seam
trend gates             hostregress       — cross-receipt drift + regime-flip gate
single-receipt gates    hostshare, regime — host_share threshold + regime classification
static censuses         glue-census, launch-census, h2d-inventory, graph-audit —
                        recomputable counts pinned to live source, drift-gated
native data plane       perf-receipt v3   — the decomposition every gate above consumes
bring-up enforcement    bringup-preflight (fak-private) — day-1 fail-closed regime gate
```

Reading order for a suspected regression: run `hostsetup-bench` on the affected shape to
get the measured ratio → check `regime`/`hostshare` on the latest receipts → diff
`glue-census` / `launch-census` / `h2d-inventory` snapshots against trunk to find WHICH
seam moved → confirm with `graph-audit` whether a capturable path lost coverage.

## Device posture

`hostsetup-bench` measures on whatever backend the compute registry resolves: `auto`
prefers `cuda` when registered in the build, else falls back to `cpu-ref` and labels the
reading `[SW-VERIFIED]`. A requested-but-unavailable backend is reported in `skipped[]`
with its reason (e.g. `not registered in this build (CUDA needs -tags cuda +
libfakcuda.a)`). Physical-silicon readings require a `-tags cuda` build on a GPU host
(fleet node per [`docs/fleet-compute-nodes.md`](../fleet-compute-nodes.md)); the bench
never fabricates a device result. Each measured sample auto-repeats the N-layer walk until
it spans ≥ 2 ms on the local clock (coarse host timers quantize smaller intervals to
zero); `reps_per_sample` reports the factor actually used, per arm.
