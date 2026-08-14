---
title: "You Never Manage the Context Window"
description: "The context window is the agent era's physical RAM, and in 2026 everyone is still writing overlay code by hand: humans decide when to /compact, agents are told not to read big files, sessions die of length. fak's doctrine is that context placement is a kernel job, not a user job — residency (which tokens are in the window) and warmth (which bytes are cached) are two projections of one placement decision the kernel owns. This page states the property, shows what is live on the default path today, and is honest about where fak is still only a referee."
slug: you-never-manage-the-context-window
keywords:
  - automatic context management
  - context window management
  - agent context window
  - context compaction
  - prompt cache
  - demand paging for LLM context
  - long agent sessions
  - zero-knob context
date: 2026-07-07
---

# You Never Manage the Context Window

**Short answer:** managing the context window is placement work — deciding which
facts are resident this turn and which cached bytes stay warm — and placement work
belongs to a kernel, not to the person or agent using it. fak's default path already
does a large slice of it automatically. This page is precise about which slice is
live today and which is still a referee-only measurement, because the honest version
is the one worth citing.

## The window is the agent era's physical RAM

Programmers used to manage physical memory by hand: overlays, segment registers,
"keep this routine under 4K so it fits." The MMU and demand paging made physical
placement invisible; allocators and garbage collection made even allocation
invisible. Today nobody *manages RAM* — they couldn't if they wanted to, and the
programs are better for it.

The context window is the same resource one abstraction layer up, and in 2026
everyone is back to writing overlay code by hand:

- humans decide when to `/compact`, `/clear`, or start a fresh session;
- humans diet their `CLAUDE.md` and memory files so the base context stays small;
- agents are *instructed by their own harness prompts* not to read large files
  because "it will overflow your context";
- agents fan out subagents whose only real purpose is protecting the parent window,
  and hand-write `HANDOFF.md` batons when a session dies of length;
- operators place cache breakpoints, pick 5-minute-vs-1-hour TTLs, and keep prefixes
  byte-stable by discipline instead of by mechanism.

Every one of those is a **manual overlay** — a human or agent doing placement work a
kernel should own. The doctrine, in its falsifiable form:

> A user- or agent-facing instruction, habit, flag, or skill whose *only* purpose is
> context management is a defect. Count them; drive the count toward zero. The knobs
> may survive as operator/debug surfaces; the default path must never require one.

"Nobody manages the window" is a product property, like "no bad tool call gets in"
(the capability floor) and "no good value silently gets lost" (context safety). It is
checkable: enumerate the manual overlays and watch the counter.

## Residency and warmth are two projections of one thing

The user side of context is **residency**: which tokens are in the rendered window
this turn. The server side is **warmth**: which bytes are warm in which cache —
provider prompt cache, in-kernel KV prefix, engine radix tree, disk CAS. These are
not two problems. They are two projections of one placement problem over one address
space:

- Every fact in a session is a **cell** with an address (fak already has the
  addresses: `ctxplan`'s lossless store, `recall`'s session-as-core-dump CAS,
  `cachemeta`'s tiered entries).
- The window is a **rendered view** of some cells (the residency projection).
- A cache line is a **warm copy** of some cells (the warmth projection).
- Every residency action has a warmth price, and vice versa. A page-out that rewrites
  the middle of the prefix busts the provider cache from that point; the compaction
  shed is only near-free *because it is suffix-shaped and byte-splices the protected
  prefix verbatim*. Upgrading a cache TTL changes the break-even for keeping a long
  prefix resident at all.

The rule that falls out: **one placement decision, one ledger.** No residency change
ships without its warmth price, and exactly one context manager owns any given wire —
everything else is sensed and made coherent, or suppressed.

## What is live on the default path today

This is the load-bearing honesty section. From a wiring survey at HEAD (a component is
"live" when a non-test file in `cmd/fak` or `internal/gateway` imports it):

- **Live, default-on, on the `fak manage`/`serve` wire** — the
  `internal/gateway/messages.go` pipeline: a 1-hour cache-TTL upgrade → an `ctxplan`
  O(1) planned resident view → a cache-prefix-preserving compaction shed (the shed is
  suffix-shaped so the warm prefix splices through byte-for-byte, and each dropped span
  leaves a `fak_context_restore` handle — plus a verbatim `[fak:goal]` pin — so nothing
  load-bearing is lost to the trim) → oversized
  `tool_result` elision → a `promptmmu` inbound tool/system prune. Alongside it:
  `ctxmmu` write-time result admission (quarantine or page-out to a CAS pointer),
  `compactcohere` coherence sensing, `cacheobs` observation, `rehydrate` on resume,
  and `recall`/`contextq` demand paging for finished sessions and MCP resources.
- **Built and CLI-reachable, but off the default path:** `fak session
  budget|envelope|reset-diff`, `fak vcache prove|observe|score`, `fak headroom`, and
  `fak debug --cmd context-plan-preview`. These are the operator/debug surfaces the
  doctrine allows to exist; they are not required to get the automatic behavior.

So a long session on the default path already gets a planned resident view, a
cache-preserving shed instead of a cache-busting truncation, write-time quarantine of
poisoned results, and demand-paging back in on resume — without anyone passing a flag.

## Where fak is still only a referee (the honest gap)

Two caveats keep this page citable rather than hype:

1. **On the Claude Code wire, fak is a referee, not the owner.** The harness still
   fires its *own* auto-compaction, and fak's actuator against it (a PreCompact
   suppression, tracked as [#1133](https://github.com/anthony-chaudhary/fak/issues/1133))
   is not yet wired. Today fak *measures and senses* the harness compacting the window
   — and keeps its own levers cache-preserving — but it does not yet prevent the
   harness from butchering context. `compactcohere` exists precisely because two
   context managers share one wire and fight; the end state is one manager per wire.
2. **Some rungs are designed, not shipped.** Relay-by-default for headless fleet
   workers, an auto-derived context envelope (retiring `--ctx-view-budget`), and the
   unconditional tool-result windowing that would let the "don't read big files"
   instruction be deleted are filed rungs under the automatic-context epic
   ([#2198](https://github.com/anthony-chaudhary/fak/issues/2198)), not live defaults.
   Where the kernel cannot place a fact automatically, the contract is to abstain with
   a structured reason — never a silent degrade.

## What you still control

Zero-knob context does not mean fak decides your goal. What stays yours:

- the actual **objective**, explicit facts, and pins;
- hard **budget limits** (token, turn, wall-clock) when you want a ceiling;
- **layout preferences** and any approval to promote or delete durable memory.

What you stop doing: saying "summarize our context," "keep that in the prompt,"
"remember the old reset had my real goal," or "don't read that file, it'll overflow."
Those are placement responsibilities the kernel carries — choosing the bounded
resident view, keeping dropped spans recoverable by digest, carrying the pinned
objective and remaining budget through a hidden reset, and keeping context-only facts
out of durable memory unless promotion is earned.

## The one-line version

The context window is RAM for agents, and asking a user or an agent to manage it is
asking them to hand-write overlays in 2026. fak treats residency and warmth as one
placement decision a kernel owns: on its own wire the automatic pipeline is the
default, and on a borrowed wire it is — for now — an honest referee working toward
being the owner.

## Where to go deeper

- The doctrine, the full manual-overlay inventory, and the per-surface infusion map:
  [`CONCEPT-AUTOMATIC-CONTEXT-2026-07-01.md`](../notes/CONCEPT-AUTOMATIC-CONTEXT-2026-07-01.md)
- The product contract for long sessions that cross hidden resets:
  [`managed-context-continuous-usage.md`](../managed-context-continuous-usage.md)
- Why a bounded resident view is cheaper than carrying a full transcript forever:
  [`o1-context-window-economics.md`](o1-context-window-economics.md)
- The cache-preserving compaction shed, how the prefix survives it, and what the drop
  leaves behind (the goal pin, the originating-task tombstone, and `fak_context_restore`):
  [`context-shedding.md`](context-shedding.md)
- The exact-span-removal primitive behind write-time quarantine:
  [`addressable-kv-cache.md`](addressable-kv-cache.md)
- Why context survival and durable-memory promotion are separate decisions:
  [`../CONTEXT-IS-NOT-MEMORY.md`](../CONTEXT-IS-NOT-MEMORY.md)

- How a large external source can be named, faulted, filtered, cached, and
  admitted without putting all of it in every model turn:
  [`context-as-a-variable.md`](context-as-a-variable.md)
