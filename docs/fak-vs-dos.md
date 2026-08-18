---
title: "FAK and DOS: which layer owns what"
description: "The practical boundary between the Fused Agent Kernel and the DOS decision/operations kernel, with command routing and a live-usage audit."
---

# FAK and DOS: which layer owns what

**FAK runs agents. DOS admits work and verifies results.**

## TL;DR

Use FAK for tool calls, context, models, cache, and policy. Use DOS for leases, evidence, liveness, and decisions. They are complementary kernels, not two names for one product.

## The 10-second routing rule

| Your question | Owner | Start here |
|---|---|---|
| How do I manage an agent's tool calls, context, tokens, model route, cache, or capability policy? | **FAK** | `fak manage`, `fak serve`, `fak session`, `fak preflight` |
| May this worker touch these files while peers are active? | **DOS** | `dos arbitrate`, then `dos lease-lane acquire` |
| Did the claimed commit, phase, or external effect really happen? | **DOS** | `dos verify`, `dos commit-audit`, `dos witness` |
| What is the fleet doing, stuck on, or waiting for a human to decide? | **DOS** | `dos top`, `dos decisions`, `dos observe` |
| How do I launch the FAK repository's issue workers and operating loops? | **FAK workflow, DOS-grounded** | use the FAK skill/`fak-dev` workflow; it calls DOS for admission, leases, and witnesses |
| Where does this repository declare DOS lanes and policy? | **FAK repository configuration consumed by DOS** | `dos.toml`; generated runtime evidence is under gitignored `.dos/` |

A useful sentence to remember is:

> FAK is in the agent's execution path; DOS is in the work's decision and proof path.

## The seam, not an overlap

### FAK owns agent execution

FAK is the shipped Go product in this repository. It wraps an agent harness and adjudicates tool calls. It also manages context, turn budgets, model routes, provider/cache state, capability policy, and runtime controls. A user adopting those outcomes installs and invokes `fak`.

Repository-specific orchestration also lives here: FAK's skills and `fak-dev` workflows know this project's issue taxonomy, test gates, worker launchers, and landing rules. They may *compose* DOS, but that does not move DOS's primitives into FAK.

### DOS owns work admission and evidence

DOS is the separately installed `dos-kernel` package and `dos` CLI. It provides repository-agnostic syscalls. Those syscalls cover admission, leases, claim verification, liveness, decisions, observation, and witnessed completion. This repository supplies `dos.toml`; DOS reads it and journals local state under `.dos/`.

Use the DOS primitive directly when the question is itself a DOS question. Do not invent FAK aliases for DOS primitives merely because the work happens in this repository. Call `dos arbitrate`, `dos verify`, `dos witness`, and `dos decisions` directly.

### Why FAK contains DOS-shaped code and docs

Three integrations are intentional:

1. **FAK workflows call DOS.** A FAK dispatcher understands the local backlog; DOS independently decides whether its proposed file tree is safe and later witnesses the result.
2. **FAK installs a headless-safe DOS hook fallback.** Some headless seats do not load the user-scope plugin. The project hook keeps DOS adjudication present there; plugin-aware wiring must avoid double firing. The recorded decision is in [`notes/DECISION-fak-dos-hook-boundary-2026-07-07.md`](notes/DECISION-fak-dos-hook-boundary-2026-07-07.md).
3. **FAK reports DOS evidence.** FAK operator views may summarize lease, hook, or decision state. Rendering DOS evidence is integration, not ownership of the DOS verdict.

If code starts reimplementing a DOS verdict rather than invoking or projecting it, the boundary has been crossed.

## Is this checkout actually using DOS?

**Yes. Its explicit primitives work, but the current hook signal is noisy.** A live audit on 2026-08-17 found:

- `dos doctor` resolved workspace `C:\work\fak`, loaded `dos.toml`, found the workspace skill pack, and identified the installed distribution as `dos-kernel`.
- The executable reports DOS `v0.29.0` from a local editable checkout. Codex discovered plugin skills cached from `dos-kernel` `0.30.0`. This is environment drift, not an ownership change.
- `dos helped` saw **1,577** hook-adjudicated calls in its current window: **573** passed untouched, **1,000** produced admission advice, and **0** were refused in that window.
- All 1,000 interventions were admission cautions, overwhelmingly on generic shell calls whose hook input exposed an empty/unknown file tree. DOS allowed those calls, but repeating the same advisory does not provide 1,000 units of value.
- This session followed the intended path. `dos arbitrate` proved the four documentation paths disjoint. Then `dos lease-lane acquire` journaled the grant before editing.
- The audit first found stale and missing lane declarations in `dos.toml`. This change reconciled them; `dos doctor --check --wiring` now exits 0 with no findings. Its inventory still labels 23 optional workspace host configs `NOT_WIRED`; none is a previously wired host that drifted.

That yields a precise verdict:

| Question | Verdict |
|---|---|
| Is DOS installed and reading this workspace? | **Yes** |
| Are hooks adjudicating calls? | **Yes** |
| Are explicit admission + lease primitives usable? | **Yes** |
| Is every hook advisory useful? | **No — unknown-footprint shell calls dominate** |
| Is the complete doctor/wiring gate green? | **Yes, after reconciling `dos.toml`** |
| Should FAK duplicate DOS to hide those rough edges? | **No — fix wiring/footprint quality at the DOS seam** |

## The correct operating pattern

```powershell
# 1. Diagnose the separately installed DOS layer.
dos doctor --check --wiring

# 2. Before concurrent edits, declare the real file footprint.
dos arbitrate --workspace . --lane my-work --kind keyword `
  --tree docs/example.md README.md --explain

# 3. A GO verdict is only a decision. Journal the hold before editing.
dos --workspace . lease-lane acquire --lane my-work --kind keyword `
  --tree docs/example.md README.md --owner <session-id>

# 4. Run the FAK-specific workflow and its tests/build gates.
#    Use fak/fak-dev for agent runtime and repository workflow concerns.

# 5. Use DOS to verify claims, then release the lease.
dos verify --help
dos --workspace . lease-lane release --help
```

The distinction between step 2 and step 3 matters: `dos arbitrate` answers “would this be safe now?”; it does not reserve anything. The lease command records the reservation so later workers can see it.

## Anti-confusion checks

- Agent-runtime questions should normally begin with `fak`: tokens, context, models, tools, policy, or serving.
- Work-control questions should normally begin with `dos`: collision, lease, truth, liveness, witness, or operator decision.
- A FAK skill named `dos-*` is an operating recipe that uses DOS; it is not a second DOS implementation.
- `.dos/` is derived local DOS state. `dos.toml` is this repository's committed DOS policy. Neither is the FAK runtime product.
- The FAK headless hook fallback exists for host coverage. It must call the stable DOS hook contract rather than importing private DOS internals.

For DOS's own verb map, run `dos start-here`. For the low-level hook audit, see the [DOS kernel transfer playbook](dos-kernel-transfer-playbook.md).
