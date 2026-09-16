---
loop: goal
witness: commit-audit
budget: { max_iters: 30 }
---

# Objective

Default every OSS study to actually running the claimed thing on our hardware.
Reading code/docs alone no longer closes a study. Each study ships a hardware
section answering: 1) reproduce, 2) limitations, 3) config deltas, 4) levers in
their world view (comparables + soft hints on the OSS Overton window).

# Hardware section template (mandatory per study note)

1. **Reproduce:** exact command run on strix1, pinned claim revision, receipt
   path in `docs/benchmarks/receipts/`. Quote date + full SHA + receipt only.
2. **Limitations:** what could NOT run here (wrong OS/arch/GPU count, closed
   engine, missing weights) and why. Single-node cannot witness dual-APU
   claims; that non-reproduction IS the finding.
3. **Config deltas:** their config vs ours (model, quant, backend, context,
   power/thermal state) as a table. No timeless "X does Y".
4. **Levers + Overton:** their tunable levers mapped to our seams
   (`path:line@sha` both sides), portfolio disposition
   (DEFAULT/OPTIONAL-MODULE/RECIPE/WATCH/EXCLUDE), and what the claim implies
   about the OSS Overton window (what is now normal to expect).

# Operating rules (already binding, restated)

- Token-compact only: `fak-strix perf --compact`, `fak-strix bench run
  --profile <p> --compact`. Never dump raw `/metrics`/`journalctl`/`sensors`.
- Moment-in-time receipts: physics floors (TTFT ceilings, tok/s floors), P50/P90
  over outliers, regime disclosure, zero hardcoded fallbacks.
- Bipartite discipline: sim/mock = `[SW-VERIFIED]` only; physical criteria stay
  `[ ]` until witnessed live (`[HW-WITNESSED]`).
- Leak guard: private notes stay in `docs/research/`|`docs/notes/`; scrub with
  `tools/issue_scrub.py` before any public filing; update
  `docs/research/monitored-repositories.json` per study.
- Non-runnable targets still owe sections 2–4 (limitation + deltas + levers);
  section 1 records the attempted command and the blocking fact.

# Runnable queue (ranked by single-strix1 feasibility)

1. `Nathanw1014/strix-halo-llamacpp` (MIT, single-node Vulkan) — first tracer:
   run their pinned Vulkan path on strix1, compare vs our in-kernel baseline.
2. `peonist-ai/halogen-flash-server` (closed engine, INSPIRE-ONLY) — behavioral
   comparison only: flags/configs/docs vs our receipts; no source port.
3. Dual-APU claims (`davidcanar/vllm-strix-halo`, `wkljohn/ds4-strix-halo-tp-odinlink`)
   — expected limitation finding on one appliance; record topology delta.
4. Foreign-silicon stacks (`mtplx`, `slotstream` Metal; `FreeToken`,
   `syv-ai/qwen38-27b-rtx3090` CUDA) — config-delta + lever-mapping only.

# Baseline evidence (2026-09-10, no new run claimed)

- `perf --compact`: MODEL Qwen3.8-27B-Q4_K_M | STAT ONLINE | PWR 8.2W |
  TEMP 48.6C | SCLK 2900MHz | RAM 42.7G/62.4G.
- Latest receipt `docs/benchmarks/receipts/latest.json`
  (`bench_quick_1789014878985514401`, 2026-09-10T04:34:38Z): quick profile,
  verdict WARN, cold TTFT 6.647s client-latency only, native phase timings
  unavailable. Unchanged checks are not re-run; next study run re-baselines.

# Done condition

- [ ] Tracer 1 landed: study note with all four hardware sections + receipt.
- [ ] Template enforced on the next three studies (or explicit limitation record).
- [ ] Registry (`monitored-repositories.json`) updated per completed study.
