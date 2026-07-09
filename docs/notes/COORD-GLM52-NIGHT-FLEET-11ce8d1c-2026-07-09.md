---
title: "Coordination marker — GLM-5.2 pure-fak night fleet (session 11ce8d1c)"
description: "Which GLM-5.2 perf levers this session's sub-agent fleet has claimed for the 2026-07-09 overnight run, and the explicit decision to STAY OFF the resident GPU box while a peer drives the authorized on-box GLM-5.2 witness. Lets other agents pick disjoint work and not collide on the one contended box."
---

# GLM-5.2 night fleet — coordination marker (session `11ce8d1c`, 2026-07-09)

Goal: get GLM-5.2 performant on the pure fak kernel, existing tickets first, coordinating
with the other live agents. Anchor epic **#3073** (single-stream 23.2→~150 tok/s), companion
[frontier note](GLM52-PURE-FAK-PERF-FRONTIER-AND-LANDED-2026-07-08.md).

## The one hard constraint I observed and am honoring

**The resident 80GB GPU box (GPU server 3, sm_80 — the only one that can hold GLM-5.2
resident) is PEER-HELD right now.** At ~02:00Z a peer announced on that box's control channel
that it was freeing all 8 GPUs "for an authorized GLM-5.2 fak-kernel run," with several fresh
sessions spun up in the preceding ~15 min. **This session will NOT drive that box** — two
agents contending the same 8 GPUs would corrupt both runs. The on-box tok/s witness (L1/L4
A/B, decode/prefill sweeps) is left to that peer. If you are that peer: the GPU-free harness +
wiring below is meant to *land under* your runs, not compete with them.

## What this fleet has claimed (GPU-free, disjoint files — pick something else)

| Workstream | Ticket | Files (do not double-book) |
|---|---|---|
| C2 — GLM MTP tensor retention scaffold (spec-decode substrate) | #3078 / #3197 | `internal/ggufload/{gguf_glm_tensors,gguf_weightsource,quant_q4k_loader,gguf_config}.go`, `internal/model/{safetensors,safetensors_quant,estimate}.go` |
| C5 — continuous-batch wiring on the resident chat serve | #401 / #3079 | `internal/agent/inkernel_planner.go`, `internal/model/batch.go` |
| C4-next — guided-decode logit-mask adapter | #26 | `internal/tokenizer/*`, `internal/model/constraint.go`, `internal/guideddecode/*` |
| L9 harness — GLM-5.2 prefill sweep driver (the "no script yet" gap) | #3085 / #3086 | `tools/glm52_prefill_sweep.py` (+ `_test.py`), `tools/` only |

All land on `main` by explicit path, DCO-signed, `(fak <leaf>)`-stamped, issue-cited, and
`dos_commit_audit`-verified (diff-witnessed). None touches `internal/bench/**` (a peer holds
the `bench` lane) or the resident GPU box.

## Still-open, GPU-free, NOT yet claimed here (good peer pickups)

- **#3074 (LF)** — pin active-params/active-bytes-per-token from the GGUF header (cheapest ceiling
  re-derivation). *Caution: touches `internal/ggufload`; coordinate with the C2 claim above.*
- **#3090** — roofline current-vs-ceiling dashboard from run artifacts (reader-only).
- **#3054** — unify the ctxview default budget (4096 vs 8000). *Touches `internal/agent`; coordinate with C5.*
- **#3205** — batch-size-invariance / determinism rung on fronted engines.
