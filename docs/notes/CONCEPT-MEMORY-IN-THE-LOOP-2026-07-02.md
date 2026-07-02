# Concept: memory in the loop — the loop reads, verifies, and writes memory by default

Date: 2026-07-02 · Status: concept + first rung in flight · Track: cache-optimization
(agent memory + reuse, `internal/worktype`)

## The gap in one sentence

fak has shipped every memory *primitive* — a deterministic memory-query algebra
(`internal/memq`), read-time artifact re-verification (`internal/recall/reverify.go`,
the `STALE_RECALL` discipline), a committed fleet-memory mirror reader
(`internal/memoryread`), write-time durability + promotion gates (`internal/ctxmmu`,
`memq/promotion.go`) — but the agent loop never uses them: a loop turn starts
memory-blind (the child sees one `FAK_GOAL_LAST_REFUSAL` scratch line plus the GOAL.md
plan) and ends without distilling anything, so every weakness here is a **missing
composition, not a missing primitive** (same verdict as
`RESEARCH-DURABLE-SESSION-STATE-SHARED-MEMORY-2026-07-01.md`).

## What the field does (SOTA scan, 2026-07)

- **The write–manage–read loop.** The 2026 surveys formalize agent memory as a
  write→manage→read cycle coupled to the action loop (arXiv 2603.07670), an evolution
  from storage → reflection → experience (arXiv 2605.06716), over the standard
  episodic/semantic/procedural taxonomy (arXiv 2602.06052). fak's durability thesis
  (`docs/CONTEXT-IS-NOT-MEMORY.md`) already encodes the episodic/semantic split as the
  `{turn, session, bounded, durable}` axis.
- **Distill lessons from trajectories, including failures.** ReasoningBank (arXiv
  2509.25140) distills *why* a trajectory succeeded or failed into small structured
  strategy items injected at task start — up to 20% relative effectiveness gain and 16%
  fewer steps on WebArena/SWE-Bench. ACE (arXiv 2510.04618) splits the roles
  (Generator / Reflector / Curator) and merges *delta* entries with deterministic
  non-LLM logic to avoid context collapse. Memp (arXiv 2508.06433) frames procedural
  memory as Build/Retrieve/Update. The deterministic-curation half of these is exactly
  fak-shaped; the loop's NOT_YET scratch lines + closed refusal tokens are ready-made
  "failure lessons" nobody is harvesting.
- **Platform memory is file-based and client-side.** The Anthropic memory tool +
  context editing (2025-09) give agents a file-based store outside the window (reported
  39% lift combined on agentic search evals); Claude Code auto-memory and the committed
  `.claude/memory` mirror are instances of the same shape. Store frameworks (Mem0,
  Letta/MemGPT, Zep/Graphiti, LangMem, A-MEM's Zettelkasten notes) own recall quality;
  benchmarks are consolidating (LongMemEval, EvoMemBench).
- **What nobody does: re-verify recalled memory against ground truth at injection.**
  The security literature warns that accumulated stale/poisoned memory drifts behavior
  (arXiv 2604.16548, OWASP memory-poisoning T1), and `docs/integrations/agent-memory.md`
  already positions fak as the trust boundary under a store. fak has a *shipped*
  read-time verifier (git SHA ancestry / worktree path / flag grep →
  fresh/stale/unverifiable) — the field has nothing equivalent. The expansion should
  lead with that differentiator rather than re-implement recall quality.

## The ladder (each rung is a composition of shipped parts)

- **R1 — a loop-facing verified-recall verb** (the first spin, in flight). A memq
  `Backend` over the markdown memory store (MEMORY.md index + per-fact files, the
  `internal/memoryread` grammar) so the SAME algebra that runs over recall images and
  Codex memories runs over the fleet memory mirror — with `Materialize` re-verifying
  each note's concrete artifact claims (`recall.ExtractArtifactClaims` +
  `DefaultArtifactVerifier`) so a stale note is *refused with evidence*, never rendered
  as fact. Sugar verb: `fak memory recall --intent <task>` emits a budget-bounded,
  freshness-tagged orientation block. Grounds #2077's "done when"; gives #1559 its
  loop-contract verb; the read half of #421's cold-start fix.
- **R2 — inject verified recall at loop-turn start.** The drive loop
  (`cmd/fak/loop_drive.go`, the `loopDriveEnv` seam) and the dispatch skills hand the
  child a verified recall block instead of (or alongside) the raw MEMORY.md read, ranked
  by the turn's plan item as intent. "Query old session memory" becomes part of the
  default loop contract (#1559) rather than a discipline.
- **R3 — distill a lesson at the witnessed turn end.** At the handoff gate
  (`loopdrive.HandoffGate`), fold the turn's NOT_YET scratch + refusal tokens + witness
  outcome into a *proposed* structured lesson (trigger context + fix), routed through
  the existing promotion/disposition gates (`memq/promotion.go`,
  `ctxmmu.GateDisposition`) — never auto-written durable. The ReasoningBank/ACE move,
  under fak's evidence rules; the write-side feeder for the fleet lessons ledger
  (#2141–#2145).
- **R4 — a recall-value witness.** A fixture-store benchmark + loop-ledger metric: does
  verified recall reduce re-discovery turns / repeated refusals? Report under the
  net-true-value standard (measured against the real alternative: the unranked,
  unverified MEMORY.md dump). No witness ⇒ the rung stays `not yet`.

Later rungs (tracked separately, not this ladder's scope): the memory tree as an
arbitrated `dos` lane (P5), healing the store fork (P6, `internal/memorycotravel`),
MEMORY.md load-cap lint as a loop preflight, scratchpad carry across session re-homes
(#2344/#2345).

## Honest fences

- R1 verifies **concrete artifact claims only** (SHA / path / flag). A note whose claim
  is prose ("the team prefers X") is `unverifiable`, rendered hedged — the verifier is
  not an oracle and must not be sold as one.
- Relevance is deterministic token overlap (the memq ranker), not semantic search. That
  is a feature at this layer (reproducible, no embedder on the hot path) and a fence:
  recall *quality* stays the store's job (`docs/integrations/agent-memory.md`).
- A rendered memory is still untrusted text; it rides the same result-admission floor
  as any tool output. Verification reduces staleness, not injection risk.

## Cross-references

Issues: #1559 (loop contract), #2077 (re-verify at injection), #421 (cold workers),
#2141/#2142/#2145 (fleet lessons ledger), #1494 (self-use epic), #2063/#2136 (working
conditions), #82 (S7 durability), #782 (memory ECC), #1433/#1431 (Codex memories),
#1860 (perpetual sessions). Docs: `CONTEXT-IS-NOT-MEMORY.md`,
`MEMORY-LAYERS-EXPLAINER.md`, `docs/integrations/agent-memory.md`,
`RESEARCH-DURABLE-SESSION-STATE-SHARED-MEMORY-2026-07-01.md`.

External anchors: arXiv 2603.07670 · 2605.06716 · 2602.06052 · 2509.25140
(ReasoningBank) · 2510.04618 (ACE) · 2508.06433 (Memp) · 2604.16548 (memory security) ·
Anthropic context-management announcement (2025-09).
