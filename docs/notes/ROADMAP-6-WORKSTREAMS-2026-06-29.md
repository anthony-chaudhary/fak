# Roadmap: the 6 live-but-not-quite-there workstreams (2026-06-29)

A reconciliation of the operator's six long-running "live but not quite there yet"
workstreams against the GitHub milestone/roadmap structure. The goal of the pass was to
**clear** the roadmap: leave zero floating issues, map every theme onto a real milestone,
and state each workstream's status as a **witnessed** `not yet` (evidence + the missing
witness + the next checkable step) rather than a self-reported "done".

The six themes are **cross-cutting** — each spans several of the 11 milestones — so they are
not themselves milestones (creating six duplicate milestones would make the board *less*
clear, the opposite of the ask). Instead each theme below names the milestone(s) that carry
it and the gating epic/issue that moves it.

GitHub-native single pane: **epic #1457** mirrors this doc as a tracking issue.

## What this pass changed on GitHub

- **38 open issues had no milestone** → now **0**. Every floating issue was routed to its
  owning milestone (mapping table at the end of this doc). The milestone view is now a
  complete partition of the open backlog.
- No milestones were created or deleted; no issues were closed (these are *open*
  workstreams, not finished ones).

Verify: `gh issue list --state open --json milestone -q '[.[]|select(.milestone==null)]|length'` → `0`.

---

## 1 — Datacenter-GPU pure-fak kernel (GLM-5.2 + Qwen-3.6 on H/A100×8)

**"There" =** one real ticket solved end-to-end through fak's *own* CUDA kernel + cache on
8×datacenter-GPU, serving GLM-5.2 and Qwen-3.6 at witnessed correctness — no llama.cpp,
no vLLM in the path.

**Milestones:** M#4 Decode parity (GPU/CUDA) · M#3 Disaggregated serving · M#2 KV cache
value · M#10 Model support & routing · M#5 (epic #1010).

**Status — `not yet` (primitives landed, no live e2e witness):**
- Shipped: expert-parallel MoE FFN sharding for GLM-5.2 (`internal/model/expert_parallel.go`,
  `f713391c`); native CUDA engine code for the pure-kernel SWE-bench path (#484–486 built/gated).
- Missing witnesses (three rungs): (a) a real cross-**device** NCCL/RCCL collective — CUDA is
  still `cudaSetDevice(0)`, `Caps.Collective=false`; (b) MLA-aware TP + EP **live-wired** into
  the forward pass; (c) the compute HAL is **f32-only** — `NewBackendSession` cannot serve
  GGUF-quant weights, so a quantized GLM/Qwen run is not yet possible through the pure kernel.
  Capacity is proven (8×80GB = 640 GiB fits GLM-5.2 UD-Q4_K_M at 433.82 GiB), but resident-EP
  tok/s is unwitnessed.

**Next checkable step:** epic **#1010** — drive the Claude harness over one real solved ticket
through fak's kernel+cache on the dgx and capture the tok/s + correctness witness. Until then
the CPU-offload baseline (0.2324 tok/s native vs 0.89 tok/s llama.cpp CPU) stands.

## 2 — Mac Qwen-3.6 working

**"There" =** Qwen-3.6 reaches M4Correct (CI-runnable correctness oracle) on the Metal
backend on the M3 Pro node.

**Milestones:** M#4 Decode parity (Metal) · M#10 Model support.

**Status — `not yet`:**
- The M3 Pro bench node (`node-macos-a`) is reachable over Tailscale; the Metal backend exists.
- Per `docs/milestones/STATUS.md` the climb is at **M4 (35.5%, 19/56 cells)** with most Metal
  cells still M1-fenced/M3-runs — Qwen-3.6×metal is **not** yet witnessed correct, and the
  f32-only HAL limit (workstream 1) applies to quantized Qwen on Metal too.

**Next checkable step:** raise Qwen-3.6×metal from its current rung to **M4Correct** with a
correctness oracle runnable on the Mac node (the metal cell in `internal/covmatrix`).

## 3 — Dogfooded, validated end-to-end fak value (api-only · pure-fak · half-fak)

**"There" =** a witnessed end-to-end value run for each of the three serving postures.

**Milestones:** M#1 Durable sessions (api-only) · M#2 KV cache value · M#3 Disaggregated
serving (pure-fak) · M#11 Substrate/interop (half-fak) · dogfood scorecard (#1422).

**Status — `not yet`, by track:**
- **(a) api-only** — *closest.* `fak guard -- claude` is the productized front door (in-process
  loopback, passthrough); the real witnessed value is **eating 429s + safety**, and the
  cache-value P&L roll-up (#1301, **closed**) gives kernel-WITNESSED + API-OBSERVED $. But
  milestone **M#1 has 0 of 23 issues closed** — the durable-session value is demonstrated, not
  yet *milestone-landed*.
- **(b) pure-fak** — gated on workstream 1 (no live kernel e2e). `not yet`.
- **(c) half-fak (sglang/vllm/plugin)** — SGLang serving + fak gateway bridges to the dgx
  Qwen3.6 box, but the EngineDriver adapters are blocked: **#39 (SGLang) is code-complete but
  uncommittable**, blocked on sibling **#38 (vLLM)** WIP.

**Next checkable step:** drive the **dogfood** KPI to A (#1422) by capturing one witnessed
e2e value run per track (api-only is one commit away; half-fak needs #38/#39 unblocked).

## 4 — RSI (recursive self-improvement)

**"There" =** the self-improvement loop synthesizes, replays, and gate-locks its own
improvements; the scorecard portfolio is all-A and gate-locked.

**Milestones:** M#6 10x agentic loop (witnessed self-correction) · scorecard epics
#1414 / #1418–1423 · #1444 (milestone report as an RSI scorecard).

**Status — `not yet` (active, real motion):**
- Shipped/closed: adjudicator-rule synthesis from the refusal log (#537, **closed**); the
  optimization fuser — RSI from hand-wired harnesses toward auto-discovered targets (#1279,
  **closed**).
- Open: the **portfolio burndown to Excellent** (#1414) — not every scorecard is grade-A
  gate-locked yet (B2 code-quality, B3 code-slop, B4 disambiguation, B5 doc-appeal, B6 tail,
  B7 gate-lock are the open children #1418–1423).

**Next checkable step:** epic **#1414** — burn each open scorecard to grade A and wire it into
`make ci` (#1423), so the portfolio cannot silently regress.

## 5 — Slack / ongoing dev

**"There" =** the Slack surfaces are durable and self-posting (no manual `fak <surface> post`).

**Milestones:** M#8 Fleet observability (Slack/grafana feeds) · M#7 Ship releases.

**Status — ongoing:**
- 10 Slack surfaces live on 1 shared bot; the slack-control bridge deploys over Slack and the
  slack-helpers durability race (`_save_state`) is fixed (v0.20.1).
- Gap: **no scheduled posts** — surfaces are posted manually today.

**Next checkable step:** the **scheduled milestone tick** (#1440) — a recurring post + durable
trend accrual — is the first surface to flip from manual to self-posting.

## 6 — Agentic dev (dispatch fleet + loop)

**"There" =** a guarded fan-out that scales the worker fleet with a queryable, witnessed loop,
driven natively in Go.

**Milestones:** M#6 10x agentic loop · M#9 Scale the dispatch fleet to 100 workers · dispatch
epic #1413.

**Status — ongoing:**
- The dispatch fleet runs (3 cron tasks, oldest-first picker `94092b2b`, guard-by-default Go
  worker); but concurrency is **seat-bound at 3** (claude), so the true 2x lever is *adding a
  seat*, not more workers.
- The Python→Go dispatch port is **in flight** (#1402–1406: account/seat routing, lane routing,
  worker prompt/picker, resolve-mode scheduling, status/witnessed-close arms).

**Next checkable step:** finish the native Go **dispatch-tick** port (#1404) and add a claude
seat to lift the concurrency cap; epic #1413 tracks deeper gh/project-management defaults.

---

## Floating-issue → milestone mapping (the clearing)

The 38 previously-unmilestoned issues, by destination:

| Milestone | Issues | Theme |
|---|---|---|
| M#9 Scale the dispatch fleet | 1395, 1396, 1397, 1398, 1402, 1403, 1404, 1405, 1406, 1410, 1413, 1415, 1451, 1452, 1453 | dispatch picker/routing/Go-port/cap-gates |
| M#6 10x agentic loop | 1414, 1418, 1419, 1420, 1421, 1422, 1423, 1449 | scorecard portfolio burndown (RSI) |
| M#8 Fleet observability | 1436, 1437, 1439, 1440, 1443, 1444 | milestone-report substrate + scheduled tick |
| M#11 Substrate/interop | 1431, 1433, 1435, 1446, 1455 | codex/memq interop + hardware-tell grammar gate |
| M#2 KV cache value | 1407, 1408, 1409 | history-compaction / ctxview firing |
| M#1 Durable sessions | 1434 | task handoff at session completion |

The mapping is a judgment call per issue and is fully reversible (`gh issue edit <n>
--milestone "<title>"`); it is recorded here so the routing is reviewable, not silent.
