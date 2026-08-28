# TencentDB Agent Memory: borrow the durability seams, not the product stack

> Studied 2026-08-28. Source: [TencentCloud/TencentDB-Agent-Memory](https://github.com/TencentCloud/TencentDB-Agent-Memory), pinned at [`5299c00aaf65481703c180fd69df066d11254eb7`](https://github.com/TencentCloud/TencentDB-Agent-Memory/commit/5299c00aaf65481703c180fd69df066d11254eb7). Tracker: [#9851](https://github.com/anthony-chaudhary/fak/issues/9851). Durable receipt: `study_94cd1dad91c0ac9eccb06ae666e7ac0d161223e6161d9282ff1e433b8988fb04`.

## Verdict

TencentDB Agent Memory is a useful implementation atlas for long-running agent memory: raw-turn capture, progressively distilled memories, hybrid retrieval, knowledge graphs, governed memory/skill bridges, and multiple harness adapters. FAK should not import that product stack. Its strongest transferable pieces are four narrower correctness gaps at existing FAK seams:

1. couple extraction-cursor progress to durable memory artifacts ([#9865](https://github.com/anthony-chaudhary/fak/issues/9865));
2. remove reinjected recall and harness noise before turn-end distillation ([#9866](https://github.com/anthony-chaudhary/fak/issues/9866));
3. pin skill versions per live session across global hot-swaps ([#9867](https://github.com/anthony-chaudhary/fak/issues/9867));
4. add bounded CJK bigrams to lexical memory recall ([#9868](https://github.com/anthony-chaudhary/fak/issues/9868)).

The graph-neighbor recall and first turn-end lesson spine were already tracked by [#2621](https://github.com/anthony-chaudhary/fak/issues/2621) and [#2349](https://github.com/anthony-chaudhary/fak/issues/2349), so this pass did not duplicate them. FAK's stable-prefix placement and default-deny policy gate are already stronger on their respective axes.

Direct code or deployment reuse is **INSPIRE-ONLY**. The root `LICENSE` says MIT, while `README.docker.md` says `Proprietary — Tencent Cloud`; GitHub reports `NOASSERTION`, and upstream clarification [#1073](https://github.com/TencentCloud/TencentDB-Agent-Memory/issues/1073) remains open. The issues above name mechanisms and FAK-owned witnesses, not ports.

## Value frame

- **For:** long-running FAK-owned agent loops that need useful memory without recursive prompt growth or session inconsistency.
- **Problem:** FAK has memory, recall, prompt-placement, policy, and queried-skill primitives, but the durable extraction and session-coherence seams are not yet closed.
- **Today:** turn-end distillation is tracked as a spine, graph recall is tracked separately, and skill hot-swap is global.
- **Better because:** artifact-coupled checkpoints prevent silent loss, capture hygiene prevents self-reinforcing memory, session pins prevent mixed skill instructions, and CJK bigrams recover phrases without a new embedder.
- **Witness:** each filed issue contains one deterministic failure fixture and a typed receipt boundary; no upstream benchmark number is used as FAK evidence.

Centrality: **Enabling**, except session-pinned skill coherence, which is **Core** to #1103. P1 bounds repeated prompt/extraction work; P2 counts checkpoint, retry, token-expansion, and stale-conflict overhead; P3 scopes state per session and retains bounded lexical expansion; P4 makes cursor, sanitization, version-bind, conflict, and retrieval evidence queryable.

## Dated source ledger

All rows were observed at `2026-08-28T21:58:11Z`. Open forge items are direction or negative evidence, not shipped behavior.

| Source | Source event and state | Immutable anchor | Platform/context and effect on the conclusion | Refresh trigger |
|---|---|---|---|---|
| Default branch `feat/server_team` | `2026-08-27T06:49:33Z`, shipped at the checked head | [`5299c00aaf65481703c180fd69df066d11254eb7`](https://github.com/TencentCloud/TencentDB-Agent-Memory/commit/5299c00aaf65481703c180fd69df066d11254eb7) | Node 22.16+ multi-service tree; pins every code anchor in this note. | Default branch moves. |
| Release `v2.0.1` | `2026-08-25T08:37:36Z`, released | [`v2.0.1`](https://github.com/TencentCloud/TencentDB-Agent-Memory/releases/tag/v2.0.1) | Confirms an actively packaged product rather than an unshipped source sketch; release claims remain subordinate to code/tests. | New release or moved tag. |
| First-class Pi adapter | `2026-08-27T06:49:33Z`, merged as the checked head | [PR #1126](https://github.com/TencentCloud/TencentDB-Agent-Memory/pull/1126) | Shows the current cross-harness direction; it does not add named behavior tests for the memory contracts borrowed here. | Adapter follow-up, revert, or test addition. |
| License contradiction | Issue created `2026-08-19T15:56:08Z`, maintainer response `2026-08-20T02:34:55Z`, still open | [root `LICENSE`](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/5299c00aaf65481703c180fd69df066d11254eb7/LICENSE), [`README.docker.md:264-266`](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/5299c00aaf65481703c180fd69df066d11254eb7/README.docker.md#L264-L266), [#1073](https://github.com/TencentCloud/TencentDB-Agent-Memory/issues/1073) | Root source says MIT while the deployment guide says proprietary and GitHub reports `Other`; all transfer decisions therefore stay INSPIRE-ONLY. | #1073 resolution or either licensing file changes. |
| Prompt-cache regression report | Created `2026-05-31T16:35:59Z`, updated `2026-08-20T08:43:43Z`, open | [#120](https://github.com/TencentCloud/TencentDB-Agent-Memory/issues/120) | OpenAI-compatible provider report; makes cache placement a measured tradeoff rather than an upstream performance claim. | Reproduction, fix, closure, or provider-contract change. |
| Benchmark reproducibility requests | Created `2026-05-22T09:17:32Z` and `2026-05-28T10:24:30Z`; updated `2026-08-06T10:52:35Z` and `2026-08-06T11:00:29Z`; open | [#73](https://github.com/TencentCloud/TencentDB-Agent-Memory/issues/73), [#106](https://github.com/TencentCloud/TencentDB-Agent-Memory/issues/106) | Public users could not reproduce or requested a method; headline PersonaMem numbers are EXCLUDE until a pinned evaluator/corpus appears. | Reproducible evaluator, corpus, and rerun land. |
| Local source inventory | Generated `2026-08-28T21:42:35Z`, exhaustive tree walk | [`fak-study-inventory-map/1`](../research/inventory/tencentcloud-tencentdb-agent-memory.json) at the checked revision | 995 files, 11 subsystems, 826 runtime files, two test-named files; supplies the coverage denominator and test-maturity caveat. | Checked revision or inventory classifier changes. |

## What the system is trying to do

The upstream worldview is that a durable agent should not treat its prompt transcript as its memory system. It records a raw L0 boundary, extracts scene/episode L1 records, rolls those into semantic L2 knowledge, and maintains an L3 persona. A separate knowledge engine makes linked pages retrievable on demand. Memory and skills are exposed through proxy bridges so the model sees a smaller, governed tool surface rather than storage credentials or arbitrary mutation authority.

That worldview matches FAK's “context is not memory” direction, but its product boundary differs. FAK owns the agent kernel, policy gate, context placement, receipts, and lifecycle. It should compose the missing semantics into those leaves instead of adopting a Node/Redis/SQLite/vector-service control plane.

## Load-bearing mechanisms

| Mechanism | Pinned upstream anchor | Transfer decision |
|---|---|---|
| Split checkpoint ownership, locked atomic capture, monotonic cursor repair | [`checkpoint.ts:4-24,616-742`](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/5299c00aaf65481703c180fd69df066d11254eb7/MemoryCore/src/utils/checkpoint.ts#L4-L24) | **ABSENT / DEFAULT:** adapt the artifact-coupled invariant in #9865. |
| Restore original user text, sanitize injected tags, strip assistant code before L0 | [`l0-recorder.ts:199-265`](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/5299c00aaf65481703c180fd69df066d11254eb7/MemoryCore/src/core/conversation/l0-recorder.ts#L199-L265) | **ABSENT / DEFAULT:** capture-boundary spine #9866. |
| Session-first skill version pin; reads stay pinned; writes use expected version and advance after success | [`skill-bridge.ts:175-182`](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/5299c00aaf65481703c180fd69df066d11254eb7/MemoryProxy/src/skill/skill-bridge.ts#L175-L182), [`840-856`](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/5299c00aaf65481703c180fd69df066d11254eb7/MemoryProxy/src/skill/skill-bridge.ts#L840-L856), [`1001-1012`](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/5299c00aaf65481703c180fd69df066d11254eb7/MemoryProxy/src/skill/skill-bridge.ts#L1001-L1012) | **ABSENT / DEFAULT:** FAK-native session binding #9867. |
| Mixed Latin/CJK split plus full-run and bigram terms at both index and query time | [`manager.ts:332-373`](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/5299c00aaf65481703c180fd69df066d11254eb7/MemoryKnowledge/src/engines/wiki/manager.ts#L332-L373) | **ABSENT / DEFAULT:** bounded shared lexical tokenizer #9868. |
| Lexical seeds followed by decayed, bounded graph expansion | [`graph-search.ts:45-87`](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/5299c00aaf65481703c180fd69df066d11254eb7/MemoryKnowledge/src/engines/wiki/graph-search.ts#L45-L87) | **PARTIAL / DEFAULT:** already #2621; no duplicate. |
| Stable/dynamic recall placement modes and stable-first deduplication | [`pipeline.ts:284-327`](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/5299c00aaf65481703c180fd69df066d11254eb7/MemoryProxy/src/injection/pipeline.ts#L284-L327) | **PRESENT / DEFAULT:** keep FAK's `syspromptmmu`/`cachemeta` seam. |
| Model-unforgeable read-only memory bridge | [`memory-bridge.ts:259-385`](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/5299c00aaf65481703c180fd69df066d11254eb7/MemoryProxy/src/injection/memory-bridge.ts#L259-L385) | **PRESENT / DEFAULT:** keep FAK policy/adjudicator authority. |
| Source-provenance cascade deletion | [`cascade.ts:69-228`](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/5299c00aaf65481703c180fd69df066d11254eb7/MemoryKnowledge/src/engines/wiki/cascade.ts#L69-L228) | **PARTIAL / WATCH:** #2624 owns structural supersession; wait for a concrete provenance-deletion dogfood need. |

## FAK on-axis witness

The completion audit reran `fak capabilities`, all four `fak-dev index docs|leaves|verbs|claims` surfaces, raw source search, and issue read-back. The issue bodies carry the full command result; the compact record is:

| Axis query | Capability result | Index/raw-code refinement | Verdict |
|---|---|---|---|
| `artifact coupled memory extraction checkpoint cursor durable write monotonic lease loss retry` | `no matching capability` | Generic durable-artifact/checkpoint-score hits only. `internal/trajectory/trajectory.go:26-45` records turns and `internal/memq/memstore.go:58-96` writes memory, but no extraction cursor binds them. | **ABSENT-on-axis** → #9865. |
| `memory capture original user prompt remove injected recall before distillation` | Compression, stable-prompt reuse, session control/restore, and policy cards | These are adjacent. `internal/trajectory/trajectory.go:296-349` copies producer query bytes without original-versus-injected provenance or a capture filter. | **PARTIAL neighborhood, ABSENT-on-axis** → #9866. |
| `session pinned skill version expected version optimistic write hot swap` | Session restore/control and model routing cards | The `skill` verb is present, but `cmd/fak/skill.go:73-88,453-459` persists one global page table. Raw search finds `expected_version` for code-file edits, not skill/session binding. | **PARTIAL global versioning, ABSENT-on-axis** → #9867. |
| `CJK bigram lexical memory recall mixed Chinese Latin tokenizer` | Session restore only | The index returns recall and model-tokenizer leaves. Reading `internal/recall/recall.go:612-615` and `internal/memq/memq.go:539-542` proves memory retrieval keeps a whole CJK run as one token; the model tokenizer is off-axis. | **PARTIAL lexical recall, ABSENT-on-axis** → #9868. |

`docs/integrations/agent-memory.md` explicitly says FAK has no embedder, vector search, or fact-extraction pipeline. That remains a product boundary, not an excuse to add one for these four issues. The memory-loop family #2346/#2349 continues to own lesson proposal/promotion, while #2618/#2621 owns graph recall. Initial wrapper-based index calls stalled, so they were not used as null evidence; the final direct `fak-dev` reads above completed and were confirmed against source.

## Candidate matrix

| Candidate and source state | Axis and upstream user-world reason | FAK on-axis witness/seam | Portfolio/license | Action or disconfirming check |
|---|---|---|---|---|
| Artifact-coupled monotonic extraction checkpoint — shipped [`checkpoint.ts:4-24,616-742`](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/5299c00aaf65481703c180fd69df066d11254eb7/MemoryCore/src/utils/checkpoint.ts#L616-L742) | Replay determinism for long-running, retrying, multi-session memory workers. | **ABSENT** between `trajectory.Turn` and `memq.MemStore.AddPromoted`. | **DEFAULT · INSPIRE-ONLY** | #9865; retire if an existing artifact/cursor coordinator is found or the failure fixture cannot reproduce skipped/duplicate work. |
| Pre-distillation capture hygiene — shipped [`l0-recorder.ts:199-265`](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/5299c00aaf65481703c180fd69df066d11254eb7/MemoryCore/src/core/conversation/l0-recorder.ts#L199-L265) | Capture fidelity for harnesses that prepend recalled memory into the user turn. | **ABSENT** at `trajectory.turnFromEvent`; prompt placement itself is PRESENT. | **DEFAULT · INSPIRE-ONLY** | #9866; reject heuristics if callers cannot supply original bytes or explicit injected spans. |
| Per-session skill-version coherence — shipped [`skill-bridge.ts:175-182,840-856,1001-1012`](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/5299c00aaf65481703c180fd69df066d11254eb7/MemoryProxy/src/skill/skill-bridge.ts#L175-L182) | Instruction consistency for teams updating shared skills while agents remain live. | **PARTIAL**: global page-table/CAS swap exists; session binding does not. | **DEFAULT · INSPIRE-ONLY** | #9867; retire if global remap can already prove a live session never observes mixed versions. |
| Bounded CJK/mixed lexical terms — shipped [`manager.ts:332-373`](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/5299c00aaf65481703c180fd69df066d11254eb7/MemoryKnowledge/src/engines/wiki/manager.ts#L332-L373) | Phrase recall for Chinese and mixed identifiers without an external segmenter. | **PARTIAL** lexical recall; **ABSENT** phrase discrimination in the two memory tokenizers. | **DEFAULT · INSPIRE-ONLY** | #9868; reject if bounded bigrams do not improve the fixed corpus or regress English ordering/cost. |
| Bounded graph-neighbor recall — shipped [`graph-search.ts:45-87`](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/5299c00aaf65481703c180fd69df066d11254eb7/MemoryKnowledge/src/engines/wiki/graph-search.ts#L45-L87) | Relational recall for wiki-like memory assets while keeping expansion bounded. | **PARTIAL**, already owned by #2621 under #2618. | **DEFAULT · INSPIRE-ONLY** | No duplicate; #2621 remains the falsifier and implementation owner. |
| Progressive turn-end semantic distillation — shipped across the L0/L1/L2/L3 pipeline | Compact durable learning for users who expect continuity across harness sessions. | **PARTIAL**, first working proposal/promotion spine already #2349. | **WATCH · INSPIRE-ONLY** | Keep #2349 minimal; do not clone the whole hierarchy unless dogfood proves another independently shippable layer. |
| Stable-prefix/dynamic-recall placement — shipped [`pipeline.ts:284-327`](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/5299c00aaf65481703c180fd69df066d11254eb7/MemoryProxy/src/injection/pipeline.ts#L284-L327) | Provider-cache reuse for agents that recall every turn. | **PRESENT-on-axis** in FAK prompt/cache placement; issue #120 warns the tradeoff is workload-dependent. | **DEFAULT · native FAK** | No issue; reopen only if a matched trace shows FAK placement causes avoidable prefix churn. |
| Model-unforgeable read-only memory bridge — shipped [`memory-bridge.ts:259-385`](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/5299c00aaf65481703c180fd69df066d11254eb7/MemoryProxy/src/injection/memory-bridge.ts#L259-L385) | Least authority for models consuming organizational memory. | **PRESENT-on-axis** through FAK policy/adjudicator, with a broader default-deny boundary. | **DEFAULT · native FAK** | No issue; a policy-bypass witness would reopen the comparison. |
| Source-provenance cascade deletion — shipped [`cascade.ts:69-228`](https://github.com/TencentCloud/TencentDB-Agent-Memory/blob/5299c00aaf65481703c180fd69df066d11254eb7/MemoryKnowledge/src/engines/wiki/cascade.ts#L69-L228) | Correct retirement for knowledge imported from mutable source collections. | **PARTIAL** near #2624 structural supersession, but no current FAK dogfood demands source cascade. | **WATCH · INSPIRE-ONLY** | Review when a concrete imported-source removal leaves derived pages behind; then file the smallest provenance-retirement leaf. |
| PersonaMem/cache-hit headline numbers — README/open-issue claims | Outcome and cache efficiency. Their users want measurable quality, but the public method is missing or disputed. | No FAK implementation axis can be justified from an unreproduced number. | **EXCLUDE** | Reopen only with a pinned evaluator, corpus, method, and matched rerun resolving #73/#106/#120. |
| Direct code or deployment reuse — exact checked tree | Supply-chain clarity for self-hosters and downstream redistribution. | Not a capability gap. | **EXCLUDE / INSPIRE-ONLY** | Reclassify only after #1073 resolves the MIT/proprietary boundary and provenance is independently reviewed. |

## Default and coverage frontiers

**Default frontier.** Keep FAK's kernel-owned policy/adjudication and stable-prefix placement as the common-case defaults. Add the four narrow semantics in #9865-#9868 at their existing leaves: replay-safe extraction, explicit capture hygiene, session-coherent skills, and bounded multilingual lexical recall. None requires the upstream service topology, vector store, or L0-L3 product hierarchy.

**Coverage frontier.** No upstream backend survives as an optional module or recipe today: adopting the Node/Redis/SQLite/vector stack would widen dependencies and authority without a witnessed cohort that FAK's native leaves cannot serve. Source-cascade deletion remains WATCH for imported mutable knowledge. The only absent supported cohort with immediate evidence is Chinese/mixed-text lexical-memory users (#9868); the other three gaps improve correctness for all long-running sessions.

Support remains bounded: #9868 is a small shared-tokenizer/test change; #9866 is a small-to-medium typed capture/provenance boundary with privacy fixtures; #9865 and #9867 are medium state-lifecycle changes requiring crash/retry/reap/conflict tests and queryable receipts. Their parent epics own maintenance and review triggers. If their declared failure fixtures do not reproduce or their incremental cost exceeds the measured saved retries/consistency gain, the default proposal is disconfirmed rather than preserved for hypothetical users.

## Negative knowledge

- Advancing a capture cursor after a failed memory sink can permanently erase the turn from later extraction. The checkpoint and artifact commit must be one outcome boundary.
- A lock acquisition is not enough: losing a renewable lease during the critical section makes the write untrustworthy and should requeue.
- Injected memory copied into the recorded user message creates a feedback loop; sanitizing only after extraction is too late.
- A global skill hot-swap and a session-coherent skill read are different semantics. Global CAS protects publication, not the instructions already observed by a running agent.
- Unicode-aware “letter” splitting is not sufficient for CJK lexical overlap; an entire sentence can remain one token.
- Per-turn memory injection can lower provider prompt-cache reuse. Upstream issue [#120](https://github.com/TencentCloud/TencentDB-Agent-Memory/issues/120) is negative evidence for measuring placement net-true, not a number to transplant.
- The pinned tree's two test-named files are not evidence for the behavior breadth represented by 826 runtime files. Every borrow needs its own FAK failure fixture.

## Source and forge coverage

The generated [inventory map](../research/inventory/tencentcloud-tencentdb-agent-memory.json) walked all 995 tracked files, 233 directories, 11 top-level subsystems, and 250,516 text lines at the pin. Three isolated workers partitioned MemoryCore, Knowledge/Proxy/deployment, and forge/governance history. The coordinator independently reopened the retained source anchors, the FAK seams, and every filed issue body before accepting them.

Non-tree coverage included README/architecture/roadmap/changelog, runtime source, the two test-named files and smoke/config surfaces, 14 GitHub releases, and the public forge denominator of 320 issues and 858 pull requests plus discussions. Public issue/PR comments and review evidence were sampled around license, benchmarks, cache placement, ACLs, checkpoint races, and the pinned Pi adapter rather than asserted from worker summaries alone.

Unavailable or deliberately unclaimed surfaces: private advisories and maintainer review, deleted/private branches, expired Actions logs, Docker image bytes/SBOMs, external repositories named only for provenance, and a complete transitive license audit. The local checkout was shallow, so commit-history conclusions came from public forge reads, not local `git log`. The repository monitor's inventory gate currently also encounters pre-existing schema drift in another row (`paper_source`); this study records the error rather than treating the global gate as green.

Refresh when the checked revision moves, upstream #1073 resolves, a reproducible benchmark/eval lands, the memory pipeline gains substantive behavior tests, or FAK closes any of #9865-#9868.

## Filed trail

- [#9851](https://github.com/anthony-chaudhary/fak/issues/9851): research tracker.
- [#9865](https://github.com/anthony-chaudhary/fak/issues/9865): artifact-coupled extraction checkpoint.
- [#9866](https://github.com/anthony-chaudhary/fak/issues/9866): capture decontamination.
- [#9867](https://github.com/anthony-chaudhary/fak/issues/9867): per-session skill-version pin.
- [#9868](https://github.com/anthony-chaudhary/fak/issues/9868): bounded CJK lexical memory tokenizer.
- Existing owners retained: [#2349](https://github.com/anthony-chaudhary/fak/issues/2349), [#2621](https://github.com/anthony-chaudhary/fak/issues/2621), and [#2624](https://github.com/anthony-chaudhary/fak/issues/2624).

Companions: [`study-repo`](../../.claude/skills/study-repo/SKILL.md) · [`field-borrow`](../../.claude/skills/field-borrow/SKILL.md) · [memory-loop epic #2346](https://github.com/anthony-chaudhary/fak/issues/2346) · [verified-recall epic #2395](https://github.com/anthony-chaudhary/fak/issues/2395) · [queried-skill epic #1103](https://github.com/anthony-chaudhary/fak/issues/1103).
