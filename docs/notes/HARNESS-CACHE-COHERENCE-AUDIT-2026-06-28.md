---
title: "Two cache stacks, one wire: how fak's cache concepts intersect the agent harness's (and do we disable Claude Code auto-compaction?) (#1131)"
description: "Audit of the coherence between fak's cache/context machinery (cache-preserving compaction, budget reset, prompt-MMU, ctxmmu quarantine, paged-KV evict, provider-prefix stability/coherence) and the agent harness's OWN cache/context machinery (Claude Code auto-compaction, cache_control prompt-cache, ~5-min TTL). Names every interference point, answers the do-we-disable-auto-compaction question with evidence (you can't via settings; only a PreCompact hook exit-2), and lands the spine: internal/compactcohere, a pure sensor+policy that attributes a per-turn prefix event and a standing block/allow posture."
---

# Two cache stacks, one wire (#1131)

_Audit + the shipped spine for the fak<->harness cache-coherence question. Companion code:
[`internal/compactcohere`](../../internal/compactcohere/compactcohere.go). Adjacent epics:
cache-preserving compaction [#745](https://github.com/anthony-chaudhary/fak/issues/745),
cut-vs-reset hybrid [#774](https://github.com/anthony-chaudhary/fak/issues/774), inbound
prompt-MMU [#751](https://github.com/anthony-chaudhary/fak/issues/751), owned-KV value
[#1072](https://github.com/anthony-chaudhary/fak/issues/1072)._

## TL;DR

When you run `fak guard -- claude`, **two context/cache managers are stacked on the same
wire, blind to each other**:

- **fak** (the kernel manager) is **cache-PRESERVING**: it sheds old turns by byte-splicing
  on the original bytes so the provider `cache_control` prefix survives verbatim
  (`agent.CompactAnthropicHistory`, #745), resets the session when the cache goes cold
  (#774), curates the inbound surface cache-prefix-safely (`promptmmu`, #751), and
  quarantines poisoned spans (`ctxmmu`).
- **Claude Code** (the harness manager) is **cache-DESTROYING**: when the conversation
  nears the context window it auto-compacts — summarizes the history and re-emits a
  **new, shorter `messages[]`**, which **rewrites the `cache_control` prefix** and bursts
  the provider cache (a `cache_creation` event, the opposite of what fak just worked to
  avoid).

They are not coordinated. The harness's auto-compaction can silently undo fak's
cache-preserving compaction, defeat fak's quarantine (by folding a sealed span into its
summary), and double-manage the window. **Do we disable Claude Code auto-compaction while
fak's concepts run?** Short answer: *you cannot disable it via settings/env* (those toggles
are silently ignored), and you *shouldn't disable it unconditionally* (it is the only net
against a hard context overflow when fak's own compaction bails). The right answer is a
**coherence protocol**: a `PreCompact` hook that fak's policy drives — **block** the
harness's auto-compaction while fak is the cache-preserving manager and coping, **allow** it
the moment fak stops coping. The pure sensor + policy for that decision shipped this pass as
`internal/compactcohere`; wiring the actuators is the epic.

## 1. The two cache stacks, side by side

| Concern | fak (kernel manager) | Claude Code (harness manager) |
|---|---|---|
| Shed old turns | `agent.CompactAnthropicHistory` — byte-splice, **cache prefix preserved** (#745) | auto-compaction — summarize + **re-emit new `messages[]`**, **prefix rewritten** |
| Trigger | budget tokens (`--compact-history-budget`) / cut-vs-reset score (#774) | context-window fraction (~clamped near 83%, see §3) |
| Reset / fresh start | `sessionreset` + `gateway.ResetOnBudget`, cold-prefix detection (#774) | `/clear`; a new summary effectively starts a fresh prefix |
| Inbound curation | `promptmmu` — drop denied tool defs past the breakpoint, cache-safe (#751) | n/a (the harness composes the body) |
| Quarantine poison | `ctxmmu` seal/tombstone; result-admit floor; `SegSealed` never re-served | n/a (the harness keeps its own full transcript on disk) |
| Provider prompt-cache | preserve `cache_control` prefix byte-identical; break it only on world-witness staleness (`cachemeta` prefix_stability / prefix_coherence) | places `cache_control` breakpoints on the static head AND recent turns; relies on the ~5-min ephemeral TTL |
| Local KV | bit-exact middle-span evict on paged KV (#33/#34/#277) | n/a — the model is upstream; no local KV on the proxy seat |

The asymmetry is the whole story: **fak's levers are designed to keep the cache; the
harness's lever is designed to keep the context window under a cap, and it pays for that
with the cache.** Neither knows the other exists.

## 2. The interference matrix (where they collide)

1. **Double compaction / fighting compactors.** Both shed old turns. fak's preserves the
   prefix; the harness's rewrites it. When the harness auto-compacts, fak's next-turn cut
   sees a changed inbound prefix and **bails** (`prefix_mismatch` / `no_breakpoint`), and
   the provider re-bills the whole new prefix at `cache_creation`. fak did the careful work
   for nothing that turn.

2. **The harness is blind to fak's shed.** Claude Code derives each request from its OWN
   local transcript; a proxy edit (fak dropping a middle turn) is invisible to it — **next
   turn it re-sends its full history anyway**. So fak's compaction is a per-turn reshaping
   of the wire, never a reduction the harness sees; and the harness's auto-compaction
   trigger counts tokens fak already drops on the wire, so the two disagree on "how big is
   the context."

3. **TTL <-> cadence.** Anthropic's ephemeral prompt cache expires ~5 min idle. An agent
   loop that idles past the TTL (waiting on a tool, a human, a `ScheduleWakeup`) returns to
   a cold cache; the next turn pays `cache_creation` on a byte-identical prefix. #774
   already reasons about this for cut-vs-reset; the harness's auto-compaction adds a second
   cold-cache source (it bursts the prefix on its own schedule).

4. **Quarantine defeated by the harness summary (a TRUST hole).** fak seals a poisoned
   tool result so the model never reads it (`ctxmmu`, `SegSealed`). But Claude Code keeps
   its own transcript and, on auto-compaction, **summarizes it** — if the poisoned span was
   in the harness transcript when it summarized, the poison can be **folded into the summary
   text**, which then rides every subsequent request as ordinary prose the kernel's seal no
   longer covers. fak controls the wire bytes; it does not control the harness's on-disk
   transcript or its summarizer. (Direct prior evidence the two layers already interfere: a
   guard quarantine of a mid-conversation tool call can orphan a `tool_use` block under
   `claude --resume`.)

5. **Who owns the window contract.** Today it is implicit and uncoordinated. The fix is to
   make it explicit: *detect* a harness compaction on the wire, *attribute* it (not fak's
   doing), and *act* (block it, or yield to it, or reset).

## 3. The explicit question: do we disable Claude Code auto-compaction?

**Finding (provenance: public Claude Code issues + community docs, 2026-06; verify against
the installed version):** auto-compaction **cannot be reliably disabled via configuration.**

- `"autoCompactEnabled": false` in `~/.claude/settings.json` is **silently ignored** — the
  key is not in the settings schema (the real state lives in `~/.claude.json`).
- `claude config set -g autoCompactEnabled false`, a `DISABLE_AUTO_COMPACT` env, and the
  `/config` toggle are unofficial and **reported unreliable** (compaction still fires).
- The trigger threshold is **clamped near ~83%** (`Math.min`), so you cannot push it out.
- There is **no `--no-auto-compact` flag**.
- Open issues track exactly this gap: anthropics/claude-code
  [#38483](https://github.com/anthropics/claude-code/issues/38483),
  [#42817](https://github.com/anthropics/claude-code/issues/42817),
  [#42149](https://github.com/anthropics/claude-code/issues/42149),
  [#24589](https://github.com/anthropics/claude-code/issues/24589),
  [#6689](https://github.com/anthropics/claude-code/issues/6689).
- **The one dependable lever is a `PreCompact` hook** that exits non-zero (exit code `2`)
  to block the pending compaction at the event level.

So the answer is not a flag flip; it is a **conditional coherence protocol**, and
`fak guard` is uniquely positioned to run it because it already controls the child Claude
process's environment and config:

> **Block** the harness's auto-compaction (PreCompact exit 2) **while fak's cache-preserving
> compaction is wired and coping** — so there is ONE cache-aware context manager, not two
> fighting. **Allow** it (PreCompact exit 0) the moment fak's own compaction has **bailed for
> a sustained streak** — because suppressing the harness when fak can't keep the window under
> the cap would strand the session into a hard context overflow. Past the cache TTL, prefer a
> **RESET** over either cut (#774).

Suppressing unconditionally is wrong (it removes the safety net); never suppressing is wrong
(the harness silently bursts the cache fak preserves and can defeat a quarantine). The
protocol is the middle path, and its decision is a pure function — which is the spine.

## 4. The spine shipped this pass: `internal/compactcohere`

A tier-1, stdlib-only, off-hot-path leaf — the **sensor + policy**, kept pure so the whole
decision is unit-tested with no gateway, provider, child process, or clock. It deliberately
does **not** re-implement what already exists: `cachemeta.Diverge` / `AnalyzeStability`
already detect THAT a prefix broke, `cachemeta.EvaluatePrefixCoherence` breaks it on purpose
for world-staleness, and `gateway.ResetScore` chooses cut-vs-reset within fak's own levers.
`compactcohere` adds the missing layer: **attribute a break to the harness-as-second-
compactor, fold in the wall-clock TTL, and decide the suppress/yield posture.**

- `Classify(prev, cur, ttl) PrefixEvent` attributes one served turn from content-free facts
  (inbound protected-prefix digest delta, fak's own `CompactOutcome`, provider
  `cache_read`/`cache_creation`, idle gap):
  `stable` / `fak_cut` / `fak_world_break` / **`harness_rewrite`** / **`cold_ttl`**. The
  load-bearing inference: **fak never changes the inbound protected prefix** (it forwards it
  verbatim), so a changed inbound-prefix digest can only be the harness rewriting its own
  history.
- `Coordinator.Observe(cur) Decision` carries rolling state and emits a per-turn `Decision`
  plus a **standing `Posture`** (block/allow). `PreCompactExitCode(posture)` maps it to the
  hook's exit code (block→2, allow→0). It blocks by default, yields after a configurable
  fak-bail streak, and raises **`QuarantineAtRisk`** when a fak seal precedes a harness
  rewrite (the §2.4 trust hole made observable) and **`BurstObserved`** when a turn costs a
  `cache_creation` burst.

What it does NOT do (the epic's actuator rungs): install/drive the `PreCompact` hook from
`fak guard`; compute the inbound-prefix digest on the gateway passthrough and feed
observations in; route `recommend_reset` into `ResetOnBudget`; surface a
`harness_rewrite` / `quarantine_at_risk` line in the guard banner + `/metrics`.

## 5. Open work -> epic #1131

The epic's Definition of Done and child tickets are filed under #1131. Sequenced by
leverage: (A) the spine [shipped], (B) gateway wiring to capture the inbound-prefix digest +
feed `compactcohere` and emit metrics, (C) the `fak guard` PreCompact-hook actuator driven by
the standing posture, (D) the quarantine-survives-summary witness (prove §2.4 and that the
warning fires), (E) the operator observability line.

## Honest fences

- The Claude Code behavior in §3 is from **public issues + community docs**, not a
  fak-witnessed run; it is labeled and dated and should be re-verified against the installed
  Claude Code version (the contested-setting situation is actively changing). The
  architectural conclusions (auto-compaction rewrites the prefix; a PreCompact hook is the
  reliable lever) do not depend on the exact setting name.
- `compactcohere` is the **decision surface only**. No actuator is wired yet, so this pass
  ships a *proven policy*, not a *realized suppression*. The block/allow recommendation is
  inert until rung C lands — stated so silence is not read as "already coordinated."
- The quarantine-survival hole (§2.4) is a **named risk with a shipped detector flag**
  (`QuarantineAtRisk`), not yet a demonstrated exploit-and-block; rung D owes the witness.
