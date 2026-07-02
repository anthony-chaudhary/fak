---
title: "Automatic context: nobody manages the window"
description: "The zero-knob doctrine for context: every fact is a cell in one address space; the window (user side) and the caches (server side) are two projections of it; the kernel decides residency and warmth together, and a user- or agent-facing instruction whose only purpose is context management is a defect to file. Binds the managed-context, relay, cache-default, prompt-MMU, and ctxplan programs to one falsifiable product property."
date: 2026-07-01
---

# Automatic context: nobody manages the window

Status: concept note + doctrine + epic spine for
[#2198](https://github.com/anthony-chaudhary/fak/issues/2198) (children
#2199–#2206). Nothing new ships from this note; it binds programs that already
exist (managed-context #1570, perpetual-session relays #1860, cache-default
#1490, inbound prompt-MMU #751, system-prompt MMU #1258, ctxplan reachability
#844) to one falsifiable property and files the rungs that are missing between
them.

## The operator's ask

> The broader point is you don't have to manage the context window yourself.
> The user side of it and the server side are two sides of the same coin.
> Think about how to infuse this in every aspect — super automatic context
> management. The user (agent or human) should NOT have to think about
> managing context ever again.

## The thesis

Programmers used to manage physical memory by hand: overlays, segment
registers, "keep this routine under 4K so it fits." The MMU and demand paging
made physical placement invisible; allocators and GC made even allocation
invisible. Today nobody *manages RAM* — they couldn't if they wanted to, and
the programs are better for it.

The context window is the agent era's physical RAM, and in 2026 everyone is
writing overlay code by hand:

- humans decide when to `/compact`, `/clear`, or start a fresh session;
- humans diet their `CLAUDE.md` and memory files so the base context stays small;
- agents are *instructed by their own harness prompts* not to read large files
  because "it will overflow your context";
- agents fan out subagents whose only real purpose is protecting the parent
  window, and write `HANDOFF.md` batons by hand when a session dies of length;
- operators place cache breakpoints, pick 5m-vs-1h TTLs, and keep prefixes
  byte-stable by discipline instead of by mechanism.

Every one of those is a **manual overlay** — a human or agent doing placement
work a kernel should own. The doctrine of this note, in its falsifiable form:

> **A user- or agent-facing instruction, habit, flag, or skill whose only
> purpose is context management is a defect.** Count them. Drive the count
> toward zero. The knobs may survive as operator/debug surfaces; the *default
> path* must never require one.

"Nobody manages the window" is a product property, like "no bad call gets in"
(the security floor) and "no good value silently gets lost" (context safety,
#1217). It is checkable: enumerate the manual overlays, watch the counter.

## Two sides of the same coin

The user side of context is **residency**: which tokens are in the rendered
window this turn. The server side is **warmth**: which bytes are warm in which
cache — provider prompt cache, in-kernel KV prefix, engine radix tree, disk
CAS. The operator's point is that these are not two problems. They are two
projections of one placement problem over one address space:

- Every fact in a session is a **cell** with an address. fak already has the
  addresses: `ctxplan`'s lossless store (the turn is an O(1) *view* over it),
  `recall`'s session-as-core-dump CAS, `cachemeta`'s tiered entries, `memq`
  cells.
- The window is a **rendered view** of some cells (residency projection).
- A cache line is a **warm copy** of some cells (warmth projection).
- Every residency action has a warmth price: a page-out that rewrites the
  middle of the prefix busts the provider cache from that point; the #555
  compaction shed is only near-free *because it is suffix-shaped and
  byte-splices the protected prefix verbatim*. Every warmth action has a
  residency price: upgrading to a 1h TTL (#1850) changes the break-even for
  keeping a long prefix resident at all (the cache-hit vanity note shows the
  incentive runs the wrong way when the two are scored separately).

Today fak runs these as two programs with two ledgers: managed-context
(#1570: plans, resets, envelopes, assumption ledgers) prices residency;
cache-default (#1490: attribution, gates, vCache) prices warmth. The seams
where they already touch — breakpoint planning as part of context planning
(#1603), cache affinity across continuations (#1609), `fak_changes`
invalidation into ctxplan (#1561), `fak vcache context-join` — are exactly the
places the tree already knows they are one thing. The doctrine makes it a law:
**one placement decision, one ledger.** No residency change ships without its
warmth price, and vice versa.

The tree also already contains the negative proof. `compactcohere` exists
because *two* context managers share one wire — fak's cache-preserving levers
under Claude Code's cache-destroying auto-compaction — and fight. Generalized:
**exactly one context manager owns a wire.** Everything else is sensed and
made coherent, or suppressed.

## What exists today (the honest map)

From a wiring survey at HEAD (imported by a non-test file in `cmd/fak` or
`internal/gateway` = live):

- **Live on the guard/serve wire, default-on:** the `internal/gateway/messages.go`
  pipeline — 1h TTL upgrade → `ctxplan` O(1) planned view (`maybePlanMessages`)
  → #555 cache-prefix-preserving compaction shed → oversized tool_result
  elision → `promptmmu` inbound tool/system prune; `ctxmmu` write-time result
  admission (quarantine / page-out to CAS pointer) on
  `adjudicate_proposed.go`; `compactcohere` coherence sensing on
  `harness_coherence.go`; `cacheobs`/`cachevalueledger` observation;
  `headroom` compression seam; `rehydrate` on resume; `recall`/`contextq`
  demand paging for finished sessions and MCP resources.
- **Built, CLI-reachable, off the default path:** `fak session
  budget|envelope|reset-diff`, `fak vcache` prove/observe/score,
  `fak headroom`, `fak debug --cmd context-plan-preview`, `fak guard
  --managed-cache` posture (AUTO only for API-key-billed Anthropic).
- **Dormant (no live importer):** `syspromptmmu` (fak authoring its own base
  context — the authorship dual of promptmmu), `memview` (derived memory
  views), `ctxresidency` (the witnessable residency read), `radixkv`
  (automatic KV reuse comparison).
- **Gated off:** `vcachechain` (M4 replay/rebuild), the `vcachegov` live
  control loop (M5 records decisions; does not act).

Two honesty caveats. First, the 54 managed-context children #1571–#1624 were
bulk-closed in late June; per the epic close-out method a CLOSED box is a
self-report, so treat those as *design-shipped* — the still-open operator
readout (#1918) is what would witness the product contract. Second, on the
guard wire fak is a referee, not the owner: the harness still fires its own
auto-compaction, and fak's actuator against it (the PreCompact suppression,
#1133) is not yet wired — today fak *measures* the harness butchering context
but does not prevent it.

## The manual-overlay inventory (what a user still manages)

Human overlays:

- **H1 — when to cut.** `/compact`, `/clear`, "this session is getting long."
  The relay (#1860) is the designed answer; the incentive layer (the cache-hit
  vanity metric note) shows why no one cuts voluntarily. Not yet automatic
  anywhere.
- **H2 — budget arithmetic.** `--ctx-view-budget` is a flat 8000 (below the
  agentic floor, #1142); `fak session budget --context-tokens` is a hand
  knob; the envelope (#1573) is user-typed syntax.
- **H3 — cache posture.** `--managed-cache auto` covers exactly one billing
  arrangement on one provider; everyone else chooses.
- **H4 — memory hygiene.** What to promote into memory files, when to compact
  them (a `memory-compact` *skill* exists — by this doctrine, a defect with a
  name), whether a recalled fact is stale (#2077 open).
- **H5 — feeding the window.** Pasting repo maps and file contents into
  prompts instead of letting the kernel demand-page them (`contextq` faults
  exist for MCP resources #1619; nothing makes this the default motion).

Agent overlays (the goal says *agent or human* — an agent that must think
about its window is the same defect one level down):

- **A1 — "don't read that, it will overflow."** Harness prompt templates warn
  agents off large files and subagent transcripts. The wire-level answer
  (unconditional tool-result windowing: head+tail+pointer, fault on demand)
  exists in `ctxview`/`ctxmmu` for the proxied path; the instruction survives
  because coverage isn't total. The instruction, not the file size, is the bug.
- **A2 — fan-out to protect the parent.** Delegating to a subagent *because
  of window pressure* is a placement decision the planner should make (or at
  least price) — today it's agent folklore.
- **A3 — hand-written batons.** `HANDOFF.md` habits; the relay's typed baton
  (#1863) is designed, not shipped.
- **A4 — post-reset re-reading.** Deciding what to re-read after a
  compaction/reset instead of a page-fault protocol doing it (#1587,
  design-closed).

Server-side overlays: breakpoint placement (#1603, design-closed; live wiring
unproven), TTL selection (H3), prefix stability as discipline rather than
contract (promptmmu covers `tools[]`; the base context is still
harness-authored while `syspromptmmu` sleeps).

## The doctrine

- **L1 — One manager per wire.** Exactly one context manager owns a wire; every
  other manager present is sensed and made coherent or suppressed
  (`compactcohere`, generalized from sensor to law).
- **L2 — The window is a view.** No component may treat a resident token as
  the only copy. Everything resident re-derives from an addressed cell
  (`ctxplan` Faithful; the relay's "transcript is disposable").
- **L3 — Removal is placement.** Page-out, evict, shed, elide all demote to a
  named tier with a fault path and a witness. Silent loss is the only sin
  (context safety #1217 is the enforcement plane).
- **L4 — One coin.** Residency and warmth are decided against one ledger.
  A residency change carries its warmth price; a warmth change carries its
  residency price. No separate scorekeeping.
- **L5 — Zero knobs on the default path.** Context knobs are operator/debug
  surfaces. A user- or agent-facing instruction whose only purpose is context
  management is a filed defect, and the count of them is a ratchet that only
  goes down.
- **L6 — Abstain honestly.** Where the kernel cannot manage automatically it
  says so with a structured reason from the closed vocabulary — never a silent
  degrade, never a quiet inherited habit.
- **L7 — Every surface declares its context plan.** A new verb, skill, or
  surface states at design time what enters the window, what pages out, and
  what warms — the way the security floor is non-optional. Advisory first,
  gate later. This is "infuse it in every aspect" as a mechanism instead of a
  slogan.

## Per-surface infusion map

| Surface | Automatic looks like | Exists at HEAD | Gap |
|---|---|---|---|
| `fak guard` (proxy) | full pipeline fires; harness compaction coherent or suppressed | pipeline live; compactcohere senses | PreCompact actuator #1133; dynamic budget #1142 |
| `fak serve` (native) | KV auto-sized; kvmmu evicts; no window error reaches a user | kvmmu bridges; #1045 designed | auto-fit live wiring; `RunArm` live caller #1316 |
| dispatch/fleet workers | headless goal ⇒ relay by default; workers never die of window | relay epic #1860 filed | tracks E–H, then default-on admission |
| `fak session` verbs | envelope derived, not typed | budget/envelope/reset-diff verbs | auto-envelope from model window + task class |
| memory/recall | recalled facts re-verified at injection; stores self-compact | memq/recall live; write gate #82 | #2077; memview wiring; retire manual compaction skills |
| subagents/workflows | delegation priced by forecast window cost | callavoid amplification data | forecast-based auto-delegation advice |
| resume/rehydrate | resume splices warm KV; no re-read ritual | rehydrate live; WaitResume built | WaitResume zero live callers |
| base context | fak authors its own spine, paged + versioned | syspromptmmu built | first live splice (epic #1258 residual) |
| docs/AEO | one canonical "you never manage context" page | explainers scattered | the page, citing this doctrine |
| CI/bench | flat-context + zero-loss witnessed per release | soak bench #1623 designed | wire into nightrun/ci |

## The rungs (new, not duplicating #1570 / #1860 / #1490)

Filed as epic [#2198](https://github.com/anthony-chaudhary/fak/issues/2198):
R1 #2199 · R2 #2200 · R3 #2201 · R4 #2202 · R5 #2203 · R6 #2204 · R7 #2205 ·
R8 #2206.

- **R1 — The manual-overlay counter.** A generated inventory of every
  context-touching flag, env var, verb, skill, and harness-prompt instruction,
  each classified `operator-debug` (fine) or `user-required` (defect). The
  `user-required` count is a ratchet. This is the doctrine's witness.
- **R2 — The one-ledger join.** One per-turn record joining residency actions
  (plan/shed/elide/page-out from ctxplan, ctxmmu, gateway) with warmth deltas
  (cacheobs KV reuse, vcachesnapshot provider window, cache_pricing dollars),
  provenance-labeled WITNESSED vs OBSERVED, never blended. #1607 designed the
  join; this rung is the live single record both programs read.
- **R3 — Placement dual-write.** Cache breakpoint placement consumes the
  residency plan: `SegStable` segments and `cache_control` positions derive
  from one structure, so prefix stability is a contract in code, not
  discipline (binds promptmmu/syspromptmmu to #1603's design).
- **R4 — The context-plan-required gate.** Advisory lint: every new `cmd/fak`
  verb or skill carries a declared context plan (enters / pages / warms), in
  the maturity-ladder style — evidence the author did not write. L7 as code.
- **R5 — Auto-envelope.** Derive the session context envelope from the model's
  window and task class; nobody passes `--ctx-view-budget` (folds #1142
  into a zero-knob wrapper).
- **R6 — Retire the "don't read big files" instruction.** Make tool-result
  windowing unconditional across proxied *and* subagent/transcript paths, then
  delete the warning from harness templates. The deleted instruction is the
  witness (L5's first scalp; the A1 overlay).
- **R7 — Relay-by-default admission.** Once #1860 E–H land: a headless session
  with a goal gets a relay policy with no operator opt-in; interactive
  sessions keep opt-in. The H1 overlay retired for the fleet.
- **R8 — The readout that proves it.** Extend the #1918 operator readout with
  the R1 counter and the R2 join so "nobody managed context this period and
  nothing was silently lost" is one witnessed screen, not a vibe.

## Honesty fences

- WITNESSED (fak's own shed/plan/evict) and OBSERVED (provider `cache_read`)
  are never summed; the vanity-metric failure mode is the canonical warning.
- The bulk-closed managed-context children are design-shipped until #1918
  witnesses the product contract; this note does not count them as product.
- KV prefix elision is not bit-exact (`exact=false`; a re-RoPE cannot un-see
  attended history) — L3's "named tier with witness" includes naming that.
- On the guard wire fak is a referee: until #1133 wires the PreCompact
  actuator, harness compaction is sensed, attributed, and *not* prevented.
- This note ships no code. Every claim about wiring is from the HEAD survey
  summarized above; every mechanism claim cites its issue or file.

## Next checkable step

Epic #2198 and children #2199–#2206 are filed (lane and witness in each
body). Build R1 (#2199) first — the counter is the cheapest rung and
everything else is graded against it. Check: `gh issue view 2199`.
