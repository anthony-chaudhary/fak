# Generation Classification — ctxmmu Tool-Catalog Paging (issue #2440)

Grooming artifact for issue #2440
(`feat(ctxmmu): tool schemas as content-hashed read-only pages with a
deterministic resident set`). It classifies the generation horizon from issue
evidence so a later, implementation-authorized worker starts warm. It is **not**
a resolution: #2440's witnessable acceptance (three `TestToolPages_*` tests plus
the `tool_schema_resident_bytes` / `tool_page_dedup_hits_total` metrics) is not
built, so #2440 stays **open**.

Parent: extends the cache/context grooming map in
[`GENERATION-CACHE-CONTEXT-PROGRAM-MAP-2026-06-30.md`](./GENERATION-CACHE-CONTEXT-PROGRAM-MAP-2026-06-30.md).
Snapshot date: 2026-07-04.

## Classification

| Field | Value |
|---|---|
| Issue | #2440 |
| Stream | `gen/next` |
| Milestone | `Generation G1 - Next Gen` (milestone #13) |
| Also carries | `generation` |
| Parent epic | #2398 (`epic(ctxpages)`), itself under harness-native program #2387 |

### Why `gen/next` (from issue evidence)

`gen/next` is the program map's "near-term cache/context integration … runnable
soon after gates, handoffs, and visibility exist" bucket, and it names
**"context page faults"** and **"live provider/cache metrics"** explicitly.
#2440 is precisely that shape:

- It is a **context page-fault** mechanism: each tool schema becomes a
  content-hashed read-only page that faults in on demand and evicts
  re-faultably, reusing the existing pager
  (`internal/ctxmmu/capbody.go` `PageOutBody`/`PageInBody`, the `abi.PageOut`
  codec seam in `internal/ctxmmu/mmu.go`).
- It adds **live metrics** (`tool_schema_resident_bytes`,
  `tool_page_dedup_hits_total`) — the gen/next "visibility" gate.
- It attaches at an **existing live seam** — `maybeCompactInboundTools` /
  `compactInboundToolsWithDecision` in `internal/gateway/messages.go` (today it
  only *prunes* floor-denied tool defs) — so it is runnable soon, not a research
  bet.
- It reuses an **in-repo** deterministic ranker
  (`internal/selfquery` `rankCards`, the discipline behind `fak_capabilities`),
  never a model guess, so the resident-set choice is witnessable.

### Why not the other streams

- **Not `gen/now`**: this is a new subsystem (the tool catalog moves out of the
  transcript into the page table), not current-product trunk hygiene with an
  immediate captured witness. It needs the test gate + metrics before it is
  safe-by-default.
- **Not `gen/second-next`**: it needs no cross-engine compatibility policy,
  adapter conformance, or simulation — it composes existing single-repo seams.
- **Not `gen/future`**: the next action is code against named seams, not
  research, a hardware witness, or a market/standards analogue.

## Promotion evidence (what moves #2440 toward `gen/now`)

Per the program map's promotion rules for cache/context work:

- The three named tests green:
  `TestToolPages_ByteIdenticalAfterCompaction` (compaction can only evict a
  schema page re-faultably, never lose it),
  `TestToolPages_ResidentSetDeterministic` (rankCards picks the same resident
  set for the same intent), `TestToolPages_PrefixHashStable` (the outbound
  splice preserves the prompt-prefix hash so paging never bursts the provider
  cache).
- `/metrics` exposes real, non-zero-by-construction
  `tool_schema_resident_bytes` and `tool_page_dedup_hits_total` backed by the
  page table (dedup counted on identical-page CAS hits across sessions).
- Gate-off / no-op behavior witnessed (the seam is identity unless enabled and
  fronting the real Anthropic passthrough), and gate-on behavior bounded to the
  tool-def path — the same discipline the inbound prune already proves.

## Demotion / retirement evidence (what pushes #2440 back or closes it)

- The prefix-hash-stable splice cannot be proven — paging tool defs bursts the
  provider prompt cache in practice — invalidating the "paging never taxes the
  cache" premise. → demote / rescope.
- The deterministic resident set regresses reachable-tool recall (a needed tool
  is paged out and the model cannot fault it back), with no task-success
  witness. → demote.
- #751 (inbound prompt-MMU) or #1258 ships a shared page-table owner that
  subsumes the outbound tool-def case, making #2440 a duplicate. → retire into
  that owner.

## Invalidating assumption

This classification assumes the observed bug trail cited in the issue —
"schemas lost after compaction, oversized-deferred-input resume hangs, ranking
regressions" — reflects the *current* transcript-resident tool-catalog path and
has not already been mitigated by the existing `maybeCompactInboundTools` prune
or a sibling ctxmmu change landed after 2026-07-04. If the bug trail is already
closed elsewhere, #2440's value is option-preservation only and it should be
re-checked against live behavior (and possibly demoted) before implementation.
A future worker must refresh `gh issue view 2440` and re-run the acceptance
`go test` selector before dispatching from this note.

## Smallest next increment (advisory, for an implementation-authorized worker)

The full feature is an epic child, not a one-file change. A defensible smallest
honest slice, deferring the rest:

1. A ctxmmu-side tool-page store: content-hash each tool schema and page it
   through the existing CAS (`PageOutBody`), deduping by digest so identical
   schemas share one page across sessions — plus
   `TestToolPages_PrefixHashStable` and `TestToolPages_ResidentSetDeterministic`
   as the first two gates (resident-set chosen by `selfquery.rankCards`).
2. A second increment wires compaction survival
   (`compactAnthropicRawWithReason` evicts a page re-faultably instead of losing
   it) with `TestToolPages_ByteIdenticalAfterCompaction`, and adds the two
   `/metrics` gauges next to `observeInboundToolPrune` in
   `internal/gateway/metrics.go`.

Do not ship the metrics as hardcoded zeros ahead of the backing store — a gauge
with no page table behind it is a misleading witness, worse than an absent one.

## Grooming action still owed (host-gated here)

The GitHub label/milestone binding is the generation intake repair this note
records but could **not apply**: `gh issue edit` is refused by the
preview-confirm gate in this worker's environment (the confirm token re-mints on
every attempt). A human or a peer with write access should apply to #2440:
`generation`, `gen/next`, and milestone `Generation G1 - Next Gen` (#13) — and
should classify the still-ungroomed parent epic #2398 (no generation label, no
milestone) for the whole harness-native ctxpages program.
