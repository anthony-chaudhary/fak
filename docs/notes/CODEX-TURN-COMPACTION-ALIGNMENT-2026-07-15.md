---
title: "fak manage ↔ Codex integrated compaction: runtime alignment"
description: "How fak manage aligns with Codex's built-in auto-compaction in practice — two wires, one budget — plus the 2026-07-15 fleet turn/compaction audit and the monitoring + best-practice defaults it motivates."
date: 2026-07-15
status: research-note
---

# fak manage ↔ Codex integrated compaction: runtime alignment

Operational companion to [BUILT-IN-COMPACTION-AUDIT-2026-07-06](./BUILT-IN-COMPACTION-AUDIT-2026-07-06.md)
(which covers *why* built-in compaction is conceptually weak). This note covers
*how fak manage actually aligns with Codex's integrated compaction at runtime*,
backed by a fleet audit of the local Codex rollout store, and the monitoring +
best-practice defaults that follow.

## Bottom line

fak manage enforces **one resident-token budget (~96K)** for a guarded session,
but it has to implement that budget **two different ways** because the model is
reached over two different wires:

| Wire | Who compacts | Mechanism | Budget knob |
|---|---|---|---|
| **Anthropic Messages** (`/v1/messages`) | **fak itself** | cache-prefix-preserving **suffix shed** — drops the uncached middle, keeps the byte-identical cached prefix; **no summary, lossless prefix** | `DefaultCompactHistoryBudget=48000` (interactive) / `HeadlessCompactHistoryBudget=96000` (headless) — `internal/gateway/gateway.go` |
| **OpenAI Responses** (Codex) | **Codex, natively** | Codex's built-in **summarize-and-discard** auto-compaction (lossy) | fak sets `model_auto_compact_token_limit=96000` at launch — `cmd/fak/codex_launcher.go`, `cmd/dispatchworker/guard.go` |

On the wire fak controls (Anthropic) it uses the *stronger* lossless cut. On
Codex's wire it cannot run that cut (wrong protocol), so it **delegates to
Codex's native compaction and configures its trigger to the same 96K budget**.
That single launch override (`codex_launcher.go:244`) is the whole alignment
seam.

## Why fak pins Codex to 96K (and why compaction "looks light")

Codex's out-of-the-box `model_auto_compact_token_limit` sits near the model
window (~245K of a 258K effective window). Left alone, two things break for a
guarded fleet:

1. Guarded sessions **appear never to compact** — they ride at ~245K until the
   window is nearly full, which is exactly the "fires too late / context rot"
   failure the conceptual note warns about (§"It often fires too late").
2. Headless dispatch workers sit **far above the 96K cache-economics budget**
   the rest of fak is tuned to (#4253), destroying cross-worker prefix reuse.

So `fak codex` / the dispatch worker inject `-c model_auto_compact_token_limit=96000`.
A later operator `-c` remains authoritative.

**This directly explains the operator-visible "compaction even when the context
looks light."** The Codex TUI shows occupancy as a fraction of the **258K model
window**; compaction fires at the **96K budget** = ~37% of that window. It is
not premature — it is the intended budget. The gap is a *display* mismatch
(window shown, budget enforced), not a bug. See the fleet evidence below.

## Fleet evidence (2026-07-15 audit)

Audited the full local Codex rollout store (`~/.codex/sessions/**/*.jsonl`,
~2,967 files) with the new monitor `tools/codex_turn_health.py`:

- **5,746 turns, ~292K tool calls, ~51 tool calls/turn, ~93% `shell_command`.**
  Volume is tool-call loop length in autonomous sessions, not user turns.
- **Compaction is trimodal, and the two large modes reconcile the alignment:**
  - **~973 events fire ≥200K** — sessions **without** the 96K override (Codex
    native default, or a user `-c`, or pre-#4253 launches). These are the
    "never compacts until nearly full" case.
  - **~141 events fire near 96K** (81K–106K) — sessions launched through
    `fak codex` with the override. **This is the intended budget**, and the
    cluster the operator sees as "compacting when light."
  - **32 events fire <40K** — genuinely premature, and concentrated in stuck
    no-op loops (see below), i.e. a *symptom*, not a threshold bug.
- **26% of all turns call zero tools** — turn inflation:
  `silent/talk-only ≈ 1,368`, `guard_refused = 133`, `preamble_noop = 12`.
  Worst case session `019f2ae9`: **44 turns, 0 tool calls** — every turn
  re-emits a "Using `super-loop` because…" preamble and yields.
- **15 guard-refusal-loop sessions** (mostly `gpt-5.6-sol`): the model proposes
  an irreversible `Remove-Item …*.pid`, the fak kernel refuses it
  (`REQUIRE_WITNESS` / preview-confirm gate: "re-propose byte-identical if
  deliberate"), and the model re-proposes variants that keep getting refused —
  up to 20 wasted turns in one session.

OpenAI's own Codex prompting guide corroborates the turn-inflation half:
preambles/plan/status narration **"can cause the model to stop abruptly before
the rollout is complete"** — precisely the talk-not-act and preamble-noop turns.

## Best-practice defaults

Posture: **additive and reversible on the shared trunk; keep coding-quality
knobs high.**

**Keep as-is (already correct):**
- `model_reasoning_effort` stays **high by default** — coding tasks want it.
  Do *not* lower it fleet-wide. (Only ever scope a reduction to purely
  mechanical dispatch loops, never coding.)
- The 96K auto-compact delegation — it is the right budget; the fix is *display*
  (show budget, not window), not a value change.

**Safe, additive monitoring default (this change):**
- `tools/codex_turn_health.py` — a privacy-preserving health monitor over the
  rollout store; exits non-zero when a red flag fires so a nightrun/CI job can
  gate on it. No prompts, args, tool results, or model text are read.

**Recommended, but gated behind review (do not flip unilaterally):**
- **`tool_output_token_limit` (e.g. 12000)** at launch, parallel to the existing
  `model_auto_compact_token_limit` override. *Tradeoff:* it caps per-turn tool
  output (stabilises prompt cache, cuts input growth) but can truncate large log
  reads a headless worker depends on. Wants its own test + fleet soak before
  becoming a default.
- **Preamble suppression** in the autonomous-loop prompt surface (super-loop /
  dispatch). Highest-leverage turn-inflation fix and OpenAI-endorsed, but it
  edits a shared prompt — treat as its own reviewed change, not a drive-by.
- **Guard-refusal alternative surfacing:** when the kernel refuses an
  irreversible delete, hand the model the *sanctioned* form up front (e.g.
  `Remove-Item <file>` without `-Recurse/-Force`, or the witness path) so it
  stops re-proposing blocked calls 16–20×.

## Monitoring — how to watch this going forward

Run the monitor (defaults to `~/.codex/sessions`, honours `CODEX_HOME`):

```
python3 tools/codex_turn_health.py            # full report + exit 1 on any flag
python3 tools/codex_turn_health.py --limit 200   # last 200 rollout files
```

Signals it emits (all fail-closed to exit 1):

- `HIGH_ZERO_TOOL_RATE` — >20% of turns call no tool (turn inflation).
- `GUARD_REFUSAL_LOOPS` — sessions re-proposing kernel-refused tool calls.
- `PREMATURE_COMPACTION` — compactions firing <40K occupancy (stuck loops).

Plus a compaction histogram (`near_budget_96k`, `near_window_200k_plus`,
`premature_lt40k`) that makes the two-wire budget posture visible at a glance —
a healthy guarded fleet should sit in `near_budget_96k`, not `200k_plus`.

## Forward opportunities

1. **Close the gap on Codex's wire with its own hooks.** The conceptual note
   catalogs Codex `PreCompact`/`PostCompact` hooks. fak already witnesses/pins
   state elsewhere; wiring a PreCompact hook to record must-keep pointers and a
   PostCompact probe to verify they survived would give Codex's *lossy* native
   compaction some of the *lossless* discipline fak applies on the Anthropic
   wire — directly answering the conceptual note's "verify compaction" and
   "keep raw evidence addressable" targets.
2. **Fix the display mismatch** so a 96K compaction on a 258K window reads as
   "at budget," not "compacting when nearly empty."
3. **Stuck-loop kill switch:** the monitor already identifies ≥N consecutive
   zero-tool turns; a live detector could halt a session at turn 3 instead of
   letting `019f2ae9` burn 44.

## Cross-links

- Conceptual: [BUILT-IN-COMPACTION-AUDIT-2026-07-06](./BUILT-IN-COMPACTION-AUDIT-2026-07-06.md)
- Cache economics: [long-session-economics](../explainers/long-session-economics.md)
- Launch seam: `cmd/fak/codex_launcher.go`, `cmd/dispatchworker/guard.go` (#4253)
- fak-side compactor: `internal/gateway/gateway.go` (`DefaultCompactHistoryBudget`, `HeadlessCompactHistoryBudget`)
- Monitor: `tools/codex_turn_health.py`
