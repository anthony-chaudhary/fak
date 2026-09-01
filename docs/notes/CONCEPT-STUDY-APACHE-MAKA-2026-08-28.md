# Apache Maka: durable runtime authority and crash-safe orchestration

> Studied 2026-08-28. Source: [apache/maka](https://github.com/apache/maka), pinned at [`2c066316545e432459e327734c634dd2cb61fa40`](https://github.com/apache/maka/tree/2c066316545e432459e327734c634dd2cb61fa40). Study tracker: [#9880](https://github.com/anthony-chaudhary/fak/issues/9880). Durable receipt: `study_5f14cbe409dee0c4aac7d910fa50e0ac33207d2194aa8a2f550a11fc8352dd22`.

## Verdict

Maka's most transferable idea is not one feature. It is a consistency model for agent software: one durable authority owns each transition; runtime events remain immutable facts; context, UI, graph, evaluation, and recovery are bounded projections; uncertain external effects are typed instead of retried optimistically.

FAK already has stronger structural policy enforcement and several matching pieces: hash-chained session evidence, admit-once RPC, content-addressed tool-result restoration, lazy tool discovery, task leases, and late-result quarantine. Three narrower gaps survived source-level comparison and deduplication:

- [#9872](https://github.com/anthony-chaudhary/fak/issues/9872): bind compaction projections to exact source-event coverage, digest, high water, and policy.
- [#9870](https://github.com/anthony-chaudhary/fak/issues/9870): derive reference-only graph records and stable activation intents from committed runtime facts.
- [#9871](https://github.com/anthony-chaudhary/fak/issues/9871): let protocol-aware tools settle during a bounded cancel grace before process reap.

All three are **inspire-only**, not direct ports. Maka's root is Apache-2.0, but its pinned [`DISCLAIMER-WIP`](https://github.com/apache/maka/blob/2c066316545e432459e327734c634dd2cb61fa40/DISCLAIMER-WIP) says the software grant and committer ICLAs are incomplete and directs downstream incorporators to perform a thorough licensing review. FAK should re-derive the Go contracts and copy no Maka source until ASF IP clearance and exact-file review change that disposition.

## Value frame

- **For:** operators running long-lived, tool-using agents across restarts and context boundaries.
- **Problem:** a process can recover while its summaries, graph state, or external-effect fate no longer correspond exactly to durable execution facts.
- **Today:** FAK has strong independent ledgers and gates, but several projections are not yet cryptographically or transactionally tied back to their precise fact set.
- **Better because:** source-bound projections and staged effect handling make recovery reconstructive instead of speculative.
- **Witness:** each filed leaf names a fail-before/pass-after artifact at the exact FAK seam.

Centrality is **Core** for source-bound compaction and **Enabling** for graph projection and cancellation settlement. P1 is the single-authority/replay-safety invariant; P2 counts the complete admission and failure path; P3 retains present FAK defaults while new behavior is explicit and bounded; P4 exposes identities, coverage, and typed uncertainty in receipts.

## The system Maka is building

Maka is an Apache Incubator, local-first personal agent workspace spanning a desktop application, CLI/TUI, one-shot runs, evaluation, MCP, computer use, and remote Runtime Hosts. At the pin it had 3,889 stars and 364 forks. The repository was created 2026-05-27 and moved exceptionally quickly: the pin was 427 commits ahead of v0.1.11, released only ten days earlier. Mutable conclusions here therefore expire on the next relevant release, merge, or ASF IP-status change.

Its architecture uses four load-bearing boundaries:

1. **Runtime facts:** the Runtime Event Log is semantic truth. Provider/model/tool stepping commits ordered facts before terminal run projections.
2. **Host authority:** one Runtime Host owns a State Root. Clients contribute capabilities but do not become storage or execution authority.
3. **Bounded projections:** transcript views, model context, graph records, and recovery plans are derived, limited, and rebuildable.
4. **Typed uncertainty:** a lost connection after dispatch is not silently retried. Unknown tool effects park continuation or remain `outcome_unknown`.

This worldview is explicit in the [Runtime core architecture](https://github.com/apache/maka/blob/2c066316545e432459e327734c634dd2cb61fa40/docs/architecture/runtime-core-architecture-draft.md#L67), [Runtime Host architecture](https://github.com/apache/maka/blob/2c066316545e432459e327734c634dd2cb61fa40/docs/architecture/runtime-host-architecture.md#L56-L66), and [Agent Graph architecture](https://github.com/apache/maka/blob/2c066316545e432459e327734c634dd2cb61fa40/docs/architecture/agent-graph-stream-scheduling-draft.md#L33-L78).

## Shipped mechanisms that carried the study

### Runtime, storage, and recovery

- Runtime events are ordered by invocation and event sequence; terminal facts cross a durability barrier before terminal run headers.
- Streaming partials are mutable presentation snapshots and are replaced by immutable finals rather than becoming permanent replay history.
- Safe continuation starts a new run only after fail-closed replay, workspace, tool-catalog, lineage, and unsettled-work checks. Unknown effects park.
- Compaction preserves source history and stores rolling V2/V3 checkpoints with exact event coverage, terminal source identity, source digest, and projection-policy version. Prefix or hash mismatch refuses reuse in [`history-compact-checkpoint.ts:34-50,275-287,507-533,549-611`](https://github.com/apache/maka/blob/2c066316545e432459e327734c634dd2cb61fa40/packages/runtime/src/history-compact-checkpoint.ts#L34-L50).
- Large tool results can become session-bound `maka://archive/...` references with a bound read/query/page capability.
- State Roots use realpath/filesystem identity plus OS-handle leases; close drains admitted work.

### Graph and multi-client operation

- Agent Graph is a schedule over existing Sessions, AgentRuns, and committed RuntimeEvents, not a second execution runtime.
- Only committed non-partial events become bounded reference-only graph records. Readiness is deterministic; admission is an atomic SQLite decision; retries inspect or recover the exact preallocated run.
- Runtime Host transcript continuity combines a bounded durable tail with an immutable active overlay, coalesces deltas, and evicts slow subscribers independently.
- Protocol evolution uses strict codecs, version ranges, compatibility epochs, composition revisions, frame/request limits, and released-peer fixtures.
- Read-only operations may reconnect; a control lost after dispatch is reconciled against authority rather than replayed.

### Tools, cancellation, and evaluation

- Computer-use requests distinguish queued, writing, delivered, and settled stages. A delivered request that loses its executor has unknown outcome.
- Protocol-aware cancellation sends `$/cancel`, waits a bounded two seconds for the action's real fate, then tears down an unresponsive executor. The code documents why cancel-and-kill in one tick can prevent the cancel write from flushing: [`maka-cu-service.ts:69-86,709-763`](https://github.com/apache/maka/blob/2c066316545e432459e327734c634dd2cb61fa40/packages/computer-use/src/maka-cu-service.ts#L69-L86).
- Eval runs task × repetition × subject cells, appends immutable attempts under an exclusive writer lock, and selects the earliest non-replaceable attempt. Comparable arms start as one task group; unmatched arms are explicitly called a bias risk.
- Live UI redaction retains a bounded private suffix so secrets split across stream deltas remain masked after display truncation.
- MCP discovery has one immutable callable snapshot per generation; remote redirects shed credentials irreversibly after crossing origin.

## FAK seam and candidate matrix

FAK self-query used `fak capabilities`, `fak dev index`, raw source searches, and open/closed GitHub issue searches. Broad capability cards were not accepted as proof of a specific mechanism.

| Capability axis | FAK status | Disposition | Decision |
|---|---|---|---|
| Exact source-bound compaction projection | **ABSENT** | **WATCH / inspire-only** | File #9872. `CompactionContract` records survival and replayable page digests but not exact source coverage/digest/policy. |
| Runtime-fact graph projection + stable activation intent | **PARTIAL** | **WATCH / inspire-only** | File #9870. `taskgraph` has durable task/lease/blocker state, but no reference-only projection from committed execution facts. |
| Cancel request → bounded settle → revoke/reap | **PARTIAL** | **WATCH / inspire-only** | File #9871. `toolprocgate.Supervisor.Tick` invokes cancel and optional OS reaper in the same tick. |
| Content-addressed restoration of shed tool results | **PRESENT** | **DEFAULT** | Keep `internal/headroom` restore handles and #2423; do not duplicate with Maka's archive URI. |
| Stable request identity and out-of-band result | **PRESENT** | **DEFAULT** | Keep `internal/fakrpc` nonce idempotency/result file; broad reconnect work would duplicate it. |
| Lazy tool-schema discovery | **PRESENT** | **DEFAULT** | FAK already has MCP registry search and Anthropic cold-tool deferral. Maka reinforces the rule that final executable binding, not a catalog, decides callability. |
| Earliest-valid benchmark retry selection | **PARTIAL** | **RECIPE** | FAK's journal flake policy already prevents last-green replacement. Revisit only if a benchmark store can still overwrite an earlier valid attempt. |
| Bounded live transcript overlay / slow-client eviction | **PARTIAL** | **WATCH** | Useful when #8586 grows a multi-client host surface; no isolated leaf now because the target product seam is still moving. |
| Cross-delta streaming display redaction | **PARTIAL** | **WATCH** | FAK has redaction and streaming assembly, but this peripheral UI technique needs a concrete display seam and captured leak witness before filing. |
| In-process permission heuristics as security boundary | **ABSENT** | **EXCLUDE** | Preserve FAK's structural default-deny capability floor. Borrow only explicit guarantee-versus-heuristic labeling. |

## Project-history lessons

Selected merged PRs expose design decisions that source snapshots alone do not:

- [#3935](https://github.com/apache/maka/pull/3935) replaced a renderer/Host multi-state delegation saga with one SQLite transaction that persists target admission plus coordinator linkage, then consumes the already-admitted message. Review caught a wake path that re-admitted the message and could drain the Host.
- [#4049](https://github.com/apache/maka/pull/4049) made managed Runtime Host lifecycle one State-Root transaction authority with exact package/generation fencing and rollback as a fresh revision.
- [#4098](https://github.com/apache/maka/pull/4098) made final executable binding the sole tool-availability authority and moved non-direct tools behind local discovery.
- [#3751](https://github.com/apache/maka/pull/3751) retained strict compaction validation while allowing one bounded repair; its failure circuit is invalidated by every causal input and does not arm on abort.

The unmerged history is equally useful:

- [#1394](https://github.com/apache/maka/pull/1394) abandoned provider-native tool search as the semantic path because schemas could still cross the provider boundary; local discovery became canonical.
- [#1541](https://github.com/apache/maka/pull/1541) did not revive an obsolete permission evaluator merely to fix its identity collision.
- [#657](https://github.com/apache/maka/pull/657) rejected heavy adversarial self-check DAGs as a strong-model default because their attention, token, and tool overhead could outweigh their value.
- [#2979](https://github.com/apache/maka/pull/2979) established that cancellation must not wait behind the hook it cancels, catalog mutation must be effect-free, and revision identity must survive through cleanup.

Open discussions [#4030](https://github.com/apache/maka/discussions/4030) and [#3630](https://github.com/apache/maka/discussions/3630) separate immutable audit history from model-serving blobs while converging retrieval on one bounded resource-reference seam. Discussion [#3286](https://github.com/apache/maka/discussions/3286) frames WorkHub as a stable coordination Session that references child execution rather than copying it.

## Negative knowledge and divergence

- Semantic replay is not bit-exact provider-wire replay; Maka does not snapshot every request byte, prompt, tool, and option.
- Agent Graph v1 is monotonic DAG-only: no deletion, rewiring, cycles, global fairness, distributed leases, or implicit graph completion.
- Runtime Host does not promise exactly-once arbitrary external effects.
- A compaction digest proves correspondence to source, not semantic completeness.
- The computer-use executor is not distributed; several product paths are robust handling around an unavailable binary.
- Eval egress filtering is a known-domain blocklist, not a complete exfiltration boundary.
- Windows sandbox work is partly experimental; a nonprivileged token prototype is explicitly disconnected from product.
- Maka's [`SECURITY.md`](https://github.com/apache/maka/blob/2c066316545e432459e327734c634dd2cb61fa40/SECURITY.md) treats the OS account as its only adversarial enforcement boundary and in-process permissions/redaction/filtering as heuristics. FAK deliberately diverges: its policy gate remains structurally load-bearing.

## License and provenance

The root license is Apache-2.0 and the NOTICE identifies Apache Maka (incubating), copyright 2026 ASF. Root provenance also identifies adapted or vendored Vercel AI SDK, Astryx, node-pty, trycua/cua, opencode, cline, gemini-cli, models.dev, DeepSeek Harness, and renderer assets under their respective Apache-2.0 or MIT terms. There are no Git submodules at the pin.

This inventory does not override the `DISCLAIMER-WIP` fence. Release bundles, renderer third-party notices, every dependency, and every file header were not independently audited. No Maka source was copied into FAK.

## Coverage, omissions, and refresh boundary

The study covered root product/architecture/security/contribution/license material; the core, runtime, Runtime Host, storage, MCP, CLI, UI, desktop, computer-use, evaluation, native Git/peer helpers, and Windows sandbox architecture/source/test seams; release/tag/compare history; selected material open/closed issues, merged/unmerged PRs, and discussions; FAK self-query; deduplication; license disposition; and three independently read-back issue packets.

Delegated coverage was independently witnessed against the pin for **9 confirmed mechanism claims, 0 refuted, 0 unwitnessed, and 0 no-claim**. The source-inventory helper hung before emitting an artifact, so this note does not claim a persisted exhaustive file manifest.

Scoped omissions: dev mailing-list archives and the Apache Incubator status record; private advisories; release-binary/source-bundle checksum verification; exhaustive dependency and per-file license audit; 21 of 26 discussion bodies; every body/review across 953 issues and 3,128 PRs; binary presentation assets, translations, lockfiles, generated bindings, and repetitive peripheral tests. Root `ROADMAP.md`, the root unfinished-work list file, and `.gitmodules` are absent; GitHub Projects and wiki are disabled.

Refresh on a new Maka release, a relevant compaction/graph/cancellation change, a source pin change used for implementation, ASF IP-clearance movement, or material movement in FAK's compaction, graph, tool supervision, archive, or multi-client host seams. This note does not claim Maka code was run, that the filed FAK issues are implemented, or that Maka's operational guarantees reproduce in FAK.
