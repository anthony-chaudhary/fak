---
title: "Microharness — native in-process Go microagent runtime (thesis, seam map, honest fences)"
description: "Design note for epic #2000: why 2 OS processes per agent is optional weight, the in-process host + ToolExec isolation dial, rate-limit-as-real-ceiling, and the quality/security fences with the child that witnesses each."
---

# Microharness — native in-process Go microagent runtime

_Design note for epic #2000 (this note is M33, issue #2032). Current-state claims are
witnessed against the files/lines cited (read 2026-07-18); the runtime core has landed
since the epic was filed, so this note records both the thesis and the shipped seam._

## Thesis

The fak production fleet is **heavy per agent**. A dispatched agent is not run by fak at
all — fak spawns **one detached OS process per agent**
(`spawnDispatchIssueWorker`, `cmd/fak/dispatch_tick_worker.go:246`), and that process is
a full external coding CLI (`claude -p ...` / `opencode run`, spawned by
`tools/dispatch_worker.py`; Go twin `cmd/dispatchworker/worker.go`) wrapped by a
**second** `fak manage` process acting as the policy reverse-proxy (`cmd/fak/guard.go`).
Steady state is therefore **~2 heavyweight processes per agent**, plus a per-process
`internal/registrations` init tax and a per-process hash-chained audit JSONL.

The SOTA "microharness" results say most of that per-agent weight is now optional:

- **mini-SWE-agent** (~100 lines: bash-only, stateless per-action subprocess, linear
  history) scores **>74% on SWE-bench Verified**.
- **smolagents** (~1000 lines, code-as-action) gets most of the way on typical tasks but
  trails on complex ones (**49% vs LangGraph's 62%** on 8+-step tasks) — thinness has a
  measurable quality edge-cost.
- The operating rule both demonstrate: **the model is the ceiling; the scaffold
  determines how close you get.**

Go is the right language for this shape (goroutines, tiny static binary, low RSS), and
fak already owns exactly the parts minimal harnesses deliberately skip: the in-process
policy gateway, a shared LRU session control plane
(`DefaultTableLimit = 8192`, `internal/session/table.go:15`), and token/budget/pace
primitives (`internal/session/envelope.go`). So the epic's move is: run **N agent loops
as goroutines inside one process behind ONE shared kernel gateway**, instead of 2 OS
processes per agent.

## Seam map (shipped in `internal/microagent`)

The `fak agent` benchmark loop (`cmd/fak/main.go:974`) was promoted into the
`internal/microagent` runtime; the host is reachable via the `fak micro` verb
(`cmd/fak/main.go:272`). One file per seam:

| Seam | File | Child |
|------|------|-------|
| Microagent value type — bounded linear-history context + model client, no per-agent process | `microagent.go`, `context.go` | #2001, #2004 |
| In-process host — N goroutine agents fan into ONE kernel gateway; per-agent control plane is the shared session table | `sessiongateway.go` | #2002, #2005 |
| **ToolExec seam** — the isolation dial (below) | `toolexec.go`, `toolexec_backend.go` | #2003, #2013, #2014 |
| Token-aware admission + scheduling — TPM/ITPM/OTPM budget queue, bounded in-flight model calls | `budgetqueue.go`, `slotsched.go` | #2019, #2006 |
| Quality layers — feedback-grounded retry, active compaction, pluggable verification | `retry.go`, `compact.go`, `verifier.go` | #2023, #2024, #2025 |
| Density — idle-agent hibernation, warm reserve, shared journal sink (not per-process JSONL) | `hibernate.go`, `warmreserve.go`, `journalsink.go` | #2012, #2011 |

## The ToolExec / isolation dial

`ToolExec` is swappable by trust level: **goroutine → subprocess (#2014) → container
(#2015) → microVM / remote sandbox (#2016)**. The invariant that makes this a dial
rather than a downgrade: **the kernel adjudication floor stays in-process at every
level** — adjudication happens at the seam BEFORE dispatch, and every construction path
(`NewToolExec`, `NewToolExecBackend`, the by-name registry) requires the kernel floor
(`toolexec.go` package doc; pinned by `toolexec_floor_conformance_test.go`). That is the
fak differentiator over minimal harnesses, whose local executors are explicitly "not a
security boundary".

## Rate-limit is the real ceiling

Guarded-dispatch concurrency is capped by distinct provider account **seats** (~2), not
by local CPU/RAM. Spinning 1000s of local agents cannot exceed provider
TPM/ITPM/OTPM + concurrency caps. So the *actual* scaling layer is the token-aware
admission control (`budgetqueue.go`, with the seat pool + runaway kill-switch of #2020
and backpressure of #2021) — built before optimizing per-agent bytes. The honest framing
of the win: **density per seat and per dollar** plus lower local orchestration
overhead — never "more agents against one provider".

## Honest fences — and the child that witnesses each

Every fence points at the child whose evidence retires it; none is retired by this note.

| Fence | Witnessing child | State (2026-07-18) |
|-------|------------------|--------------------|
| **Quality:** a thin native loop can trail a full CLI on complex tasks (smolagents 49% vs 62%; bash-only beats structured tools only sometimes). No real lane migrates until the gap is measured. | **M29 = #2028** (quality ablation vs the guarded-CLI baseline on SWE-bench-style tasks) and **M34 = #2033** (benchmark plan: density, $/task, wall-clock, quality) | Both OPEN — the migration gate is not yet witnessed |
| **Security:** thin harnesses drop isolation; the floor must not thin with the scaffold. | **M18 = #2018** (kernel adjudication floor at EVERY isolation level) | Landed; enforced by construction + `toolexec_floor_conformance_test.go` |

## What this does NOT do

- **Not a rate-limit escape.** Provider TPM/ITPM/OTPM and concurrency caps bind exactly
  as before; the runtime only raises density per seat / per dollar under them.
- **Does not replace #1951** (universal harness *profiles*, which wrap external
  sub-harnesses — see `UNIVERSAL-HARNESS-PROFILES-2026-07-01.md`). Profiles are the
  external-wrap plane; this is the complementary **native** runtime.
- **Does not replace #1911** (agentic-first scheduling) or the relay perpetual-session
  epic; the in-process host is a target those layers can drive (#2030 lets
  dispatch/wave target it).
- **Not a claim that thin wins on quality.** That claim belongs to #2028/#2033 and is
  open until they land numbers.

## Related

- #2000 — the epic (child map M1–M34).
- #1951 — universal harness profiles (external sub-harness config + rotation).
- #1911 — agentic-first scheduling.
