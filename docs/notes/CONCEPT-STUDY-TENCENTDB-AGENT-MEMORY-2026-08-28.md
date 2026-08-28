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

The study used `fak capabilities`, the docs/leaf/verb/claim index surfaces where responsive, and raw source/issue search before retaining gaps.

- Queries for layered agent memory, memory-asset loadouts, versioned skill assets, wiki link graphs, and proxy memory injection returned no direct capability match. Generic model-routing and session-control hits were rejected as off-axis.
- `docs/integrations/agent-memory.md` explicitly says FAK has no embedder, vector search, or fact-extraction pipeline. That is a product boundary, not a reason to introduce one for these four issues.
- `cmd/fak/skill.go:73-88` proves global version pins, residency reads, and guarded hot-swap are present; raw search found no per-session skill-version binding or `expected_version` contract.
- `internal/recall/recall.go:612-615` and `internal/memq/memq.go:539-542` independently tokenize by non-letter/non-digit separators, leaving uninterrupted CJK text as one long lexical term.
- The memory-loop issue family #2346/#2349 owns lesson proposal/promotion, while #2618/#2621 owns graph recall. Neither contains the checkpoint or capture-hygiene witness filed here.
- Some `fak dev index` queries stalled on this peer-dirty workstation; their absence was not treated as proof. The retained verdicts rest on direct source reads and issue read-back.

## Candidate matrix

| Candidate | FAK status | Disposition | Outcome |
|---|---|---|---|
| Artifact-coupled, monotonic extraction checkpoint | **ABSENT** | **DEFAULT · INSPIRE-ONLY** | #9865 |
| Pre-distillation capture hygiene | **ABSENT** | **DEFAULT · INSPIRE-ONLY** | #9866 |
| Per-session skill-version coherence | **ABSENT** | **DEFAULT · INSPIRE-ONLY** | #9867 |
| Bounded CJK/mixed lexical memory tokens | **ABSENT** | **DEFAULT · INSPIRE-ONLY** | #9868 |
| Graph-neighbor recall | **PARTIAL** | **DEFAULT** | Existing #2621 |
| First witnessed turn-end lesson | **PARTIAL** | **WATCH** | Existing #2349 is the spine; do not clone L0-L3 wholesale. |
| Stable-prefix/dynamic-recall placement | **PRESENT** | **DEFAULT** | Keep FAK's native placement/accounting. |
| Model-unforgeable memory authorization | **PRESENT** | **DEFAULT** | Keep FAK's default-deny policy gate. |
| Source-cascade deletion | **PARTIAL** | **WATCH** | Revisit only with a FAK provenance-retirement witness. |
| PersonaMem/cache-hit headline numbers | **ABSENT** | **EXCLUDE** | No reproducible pinned method; issues #73, #106, and #120 qualify or dispute outcomes. |
| Direct code/deployment reuse | **ABSENT** | **EXCLUDE** | License conflict #1073 is unresolved. |

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

