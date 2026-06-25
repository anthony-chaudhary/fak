# Inbound prompt-MMU — curate WHAT enters the context window (2026-06-25)

**Status:** spine SHIPPED (`internal/promptmmu`, tier-1, green). Wiring + generalization rungs OPEN.
**Epic issue:** filed as `epic(promptmmu)` (see GitHub). Sibling of #745 (cache-preserving
compaction), #746 (direct-MCP value), #747 (guard value).

## Crux

fak already controls the **outbound / result** half of the context window end-to-end:

- `ctxmmu` gates what comes BACK — tool results are quarantined (poison/secret) or paged out
  to a <2KB CAS pointer.
- the adjudicator gates tool CALLS — allow / repair / deny.
- `agent.CompactAnthropicHistory` (#555) byte-splices OLD turns out of `messages[]` while
  preserving the cached prefix to the byte.

But the **inbound composition** of the window is forwarded **byte-identical** on the flagship
`fak guard -- claude` passthrough. Verified: `req.Raw` is mutated in exactly two places —
`max_tokens` (messages.go:269) and old-turn compaction (messages.go:284). The `tools` array
(tool DEFINITIONS the model is offered), the `system` prompt, and any harness-injected skill
or memory text all pass through untouched. Nobody curates **what goes in** per turn — only
what comes back and what falls off the end.

This epic builds the inbound prompt-MMU: the INGRESS dual of `ctxmmu`. It curates what the
model is offered, starting with tool definitions the kernel has **already adjudicated as
denied** — a denied tool can never be called, so dropping its definition is pure upside (the
model loses nothing it could do; the request carries fewer uncached bytes upstream).

## Default posture: on by default, both directions; flags are for opinionated overrides

The egress half is already on by default (ctxmmu, adjudication always run; compaction is the
one opt-in). The ingress half should match: **curate both directions out of the box.** A
feature flag exists for the opinionated, deployment-specific case (a user who wants a wider or
narrower tool surface than the policy-denied default), NOT as the on-switch. This raises the
bar on fail-safe identity — an on-by-default pruner that is wrong is far more dangerous than an
opt-in one — which is exactly why the spine's floor is "prove cache-safe + well-formed, or ship
the original verbatim."

## Load-bearing invariants (every rung holds ALL five)

1. **Cache-prefix preserved — proven, not hoped.** Every byte from offset 0 through the end of
   the last `cache_control`-bearing element is copied VERBATIM (memcpy), never re-serialized.
   The result re-decodes AND `bytes.Equal(raw[:prefixEnd], out[:prefixEnd])`. No partially-cached
   element is ever re-marshalled, so JSON key reordering can never bust the prefix.
2. **Fail-safe identity, named.** On ANY ambiguity return the input slice UNCHANGED with a
   closed-set `SkipReason`. When the transform cannot PROVE it is cache-safe and well-formed it
   ships the original. An identity result is auditable, never a silent no-op.
3. **The pruned thing is NAMED.** A removed tool is reported in `PruneResult.Pruned`. No tool
   silently vanishes.
4. **Reversible.** Pure function of `(raw, plan)`; an empty plan reproduces the input bit-for-bit.
5. **Decoded kernel view untouched — trust boundary unchanged.** Pruning only narrows what the
   model SEES, and only by the denied set; it never alters the decoded request the kernel
   adjudicates. The kernel polices the same tool set before and after.

## Cache geometry (read before changing the splice)

In Claude Code's wire shape `tools[]` sits structurally BEFORE `messages[]`, and the
`cache_control` breakpoint typically lands on the LAST tool (the whole static tool block is the
cached head). Two consequences the spine encodes:

- The boundary is the **last TOOL-level breakpoint** — derived by the spine as an index, NOT a
  trusted caller flag and NOT the global last breakpoint. (The shipped messages[] compactor
  anchors on the FIRST breakpoint because in a growing conversation the last is a recent
  message; tools[] is a static block, so last-tool-breakpoint is the correct anchor. Two
  independent design judges flagged this inversion — it is the one place a naive port goes
  wrong.)
- Therefore the set of tools strictly after the breakpoint is usually **empty mid-session** —
  and that is correct: pruning a tool at/before the breakpoint would move the cache boundary
  and bust the whole session. The spine fires only when the tool block is rebuilt anyway
  (session start / post-RESET). Harvesting denied-tool headroom across a RESET is a later rung.

This means the v1 spine is **green but largely inert on live mid-session flagship traffic** —
shipped honestly as the proven core that the RESET-time and session-start rungs build on, not
as an immediate live win. The win lands when Rung 3 wires it at the prefix-rebuild boundary.

## Rungs

| # | Rung | Gate |
|---|---|---|
| 1 | `internal/promptmmu`: `CompactInboundTools` byte-splice spine (tool-def pruning, denied-first) | **SHIPPED** — independent |
| 2 | adjudication → `ToolPlan` adapter (project the kernel's per-tool DENIAL onto `Drop`) | needs Rung 1 |
| 3 | gateway wiring on the passthrough, **session-start / RESET only** (the only `req.Raw` mutation) | green `internal/gateway` |
| 4 | flags + observability: **default-on** curate, `off`/`shadow` override; log `Pruned` | follows Rung 3 |
| 5 | agent self-control over its own advertised MCP tool surface (self-imposed `Drop`) | needs Rung 1+2 |
| 6 | generalize: skills demotion / memory budgeting / system pruning (same splice, per-block breakpoint) | needs Rung 1; lowest priority, highest blast-radius |
| 7 | SAFETY witness: prove over real bodies that every drop is named, reversible, kernel-view byte-unchanged | independent; gates 3–6 going live |

## Non-goals / honesty caveats

- **No silent lobotomy.** Only stops advertising tools the kernel ALREADY denied — never a
  heuristic "probably won't need this" pruner. Not provably deniable ⇒ stays.
- **No outbound / security-floor change.** Touches only the advertised inbound surface; the
  decoded view the kernel adjudicates is byte-unchanged.
- **Mid-session pre-breakpoint pruning is intentionally refused** (it would bust the cache).
- **System-prompt pruning is last + most conservative** — the system block is the most
  cache-sensitive and behavior-critical surface.
- **No model-visible "withheld tools" note in v1** — a denied tool is invisible by design;
  legibility is for the operator via `Pruned`.
