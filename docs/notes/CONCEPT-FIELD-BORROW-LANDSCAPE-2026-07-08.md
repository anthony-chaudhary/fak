---
title: "field-borrow pass over the 2026-07-07 related-repos landscape"
description: "Witnessed capability-import: 20 borrows from the 2026-07-07 landscape sweep dogfooded against fak's own self-query (fak capabilities / fak index + grep), grounded in a file:line seam, deduped vs the 915 open issues, and filed epic-anchored — 17 real PARTIAL gaps (#3190-#3206), 2 dropped PRESENT, 1 present-with-narrow-extension."
---

# field-borrow: the 2026-07-07 landscape → witnessed, grounded backlog

Applying the [`field-borrow`](../../.claude/skills/field-borrow/SKILL.md) discipline to the
verified [related-repos landscape](RESEARCH-related-repos-landscape-2026-07-07.md): for each
borrow, dogfood fak's self-query surface (`fak capabilities` / `fak index docs|leaves|verbs` +
a raw `Grep`) to witness **PRESENT / PARTIAL / ABSENT**, ground each real gap in a `file:line`
seam, dedup against the 915 open issues, pick the parent epic, and file. **The witness caught
two PRESENT cases** (fak already ships them) — the exact "don't file what you already have"
guarantee this skill exists for.

## Outcome — 17 filed, 2 dropped PRESENT, 1 narrow extension

| Capability (named source) | Verdict | Seam (file:line) | Outcome |
|---|---|---|---|
| Signed git Reference State Log, verify without trusting the forge (gittuf, NDSS'25) | PARTIAL | `internal/safesync/safepush.go:24` · `journal.go:378` | [#3190](https://github.com/anthony-chaudhary/fak/issues/3190) |
| Mandatory artifact-digest binding on the verifier (sigstore-go) | PARTIAL | `internal/agent/receipt.go:169` | [#3191](https://github.com/anthony-chaudhary/fak/issues/3191) |
| Merkle inclusion/consistency proofs over the journal (Trillian, RFC 6962) | PARTIAL | `internal/journal/journal.go:821` | [#3192](https://github.com/anthony-chaudhary/fak/issues/3192) |
| Material/product/command-run attestations across a pipeline (in-toto witness) | PARTIAL | `internal/witness/witness.go:258` | [#3193](https://github.com/anthony-chaudhary/fak/issues/3193) |
| Machine-check the adjudicator fold's order-independence (cedar-go, verification-guided) | PARTIAL | `internal/kernel/kernel.go:188` | [#3194](https://github.com/anthony-chaudhary/fak/issues/3194) |
| Typed non-Turing-complete condition language for IFC deny rules (cel-go) | PARTIAL | `internal/adjudicator/decide.go:155` | [#3195](https://github.com/anthony-chaudhary/fak/issues/3195) |
| In-process Wasm object-capability sandbox (wazero/Extism) | PARTIAL | `internal/toolprocgate/spawnbroker.go:117` | [#3196](https://github.com/anthony-chaudhary/fak/issues/3196) |
| Native EAGLE-3 draft head wiring the shipped verify-accept substrate (EAGLE-3/SpecForge) | PARTIAL | `internal/model/kv.go:742` | [#3197](https://github.com/anthony-chaudhary/fak/issues/3197) |
| Query-agnostic importance-by-reconstruction KV eviction (KVzip) | PARTIAL | `internal/radixkv/radixkv.go:82` | [#3198](https://github.com/anthony-chaudhary/fak/issues/3198) |
| Real RDMA/NVLink/NVMe-oF KV-transport backend (NIXL/Mooncake/InfiniStore) | PARTIAL | `internal/model/kv_transport_registry.go:108` | [#3199](https://github.com/anthony-chaudhary/fak/issues/3199) |
| First-seen MCP server quarantine + tool-definition pinning (mcpproxy-go/mcp-scan) | PARTIAL | `internal/adjudicator/decide.go:510` | [#3200](https://github.com/anthony-chaudhary/fak/issues/3200) |
| Inspect AI + MCP-Bench as scored floor-grading harnesses (AgentDojo family) | PARTIAL | `internal/agentdojo/register.go:36` | [#3201](https://github.com/anthony-chaudhary/fak/issues/3201) |
| Score tree-disjoint arbitration's conflict-rate reduction (AgenticFlict dataset) | PARTIAL | `internal/laneadmit/laneadmit.go:95` | [#3202](https://github.com/anthony-chaudhary/fak/issues/3202) |
| Cite the 2026 agent-OS convergence in prior-art (ActPlane/ACP/AgenticOS/Right-to-History) | PARTIAL (docs) | `docs/industry-scorecard/security.md:12` | [#3203](https://github.com/anthony-chaudhary/fak/issues/3203) |
| LLMLingua-2 token-level compressor plugin (LLMLingua) | PARTIAL | `internal/headroom/native.go:18` | [#3204](https://github.com/anthony-chaudhary/fak/issues/3204) |
| Batch-size-invariance rung + fronted-engine determinism posture (Thinking Machines) | PARTIAL | `internal/model/batch_test.go:161` | [#3205](https://github.com/anthony-chaudhary/fak/issues/3205) |
| Bitemporal validity windows + as-of-time recall (Zep/Graphiti) | PARTIAL | `internal/memq/notesbackend.go:100` | [#3206](https://github.com/anthony-chaudhary/fak/issues/3206) |
| Per-block privacy-labeled cross-tenant KV sharing (SafeKV/KVShare) | **PRESENT** | `internal/gateway/l3share.go:117` · `cachemeta/pool.go:201` | not filed — ships under epic #53 |
| IFC confidentiality+integrity label lattice (CaMeL/FIDES/Progent/MVAR) | **PRESENT** | `internal/ifc/ifc.go:610` (header names CaMeL/FIDES) | not filed — already shipped |
| OS-level Landlock/Firecracker sandbox (go-landlock/firecracker-go-sdk) | PRESENT (partial) | `internal/guard/landlock.go` | comment on #3196 (extend to SpawnBroker; microVM ABSENT) |

## What the witness step earned

- **Two filed-gap errors prevented.** SafeKV and the CaMeL/FIDES IFC lattice look like obvious
  "fak should add" borrows from the landscape — both are already shipped and fail-closed
  (`l3share.go`/`cachemeta/pool.go`; `internal/ifc/ifc.go`, whose header literally names CaMeL/FIDES).
  Filing them would have been the exact anti-pattern field-borrow exists to kill.
- **One reframe from "adopt" to the real gap.** cedar-go is already matched by fak's default-deny
  forbid-over-permit engine; the actual gap is that the fold's *order-independence* is only asserted
  in comments (#3194), not machine-checked — a sharper, cheaper issue than "adopt Cedar."
- **Two PRESENT-narrowed borrows.** Determinism (#3205) and OS-sandbox (Landlock) both turned out
  largely present — fak proves batched==serial bit-exact and already Landlock-jails the agent child —
  so the filed/commented gaps are narrow (cross-batch-size rung; SpawnBroker-level jailing + microVM),
  not the headline capability.

## Method / fences
- Witnessed by a read-only parallel pass (`fak capabilities`/`fak index` + `Grep` + `gh` dedup); no
  code changed. Sources are the adversarially existence-verified 2026-07-07 landscape (zero refuted).
- Parent epics: #2391 (witness) ×3, #2389 (policy) ×3, #2236 (superset) ×3, #2354 (security) ×2,
  plus #2392, #50, #868, #3165, #2393, #2395. Children cross-reference their epic in-body.
- 3 candidates hit a transient schema-retry cap in the batch pass and were re-witnessed by hand
  (determinism, bitemporal, OS-sandbox) — hence the manual #3205/#3206 + the #3196 comment.

## Companions
- Source landscape: [`RESEARCH-related-repos-landscape-2026-07-07.md`](RESEARCH-related-repos-landscape-2026-07-07.md)
- Durable searches (operational): [`tools/idea_scout_topics_fresh.json`](https://github.com/anthony-chaudhary/fak/blob/main/tools/idea_scout_topics_fresh.json)
- Skill: [`field-borrow`](../../.claude/skills/field-borrow/SKILL.md) · prior instance: [`CONCEPT-FIELD-BORROW-QUERY-QUALITY-2026-07-08.md`](CONCEPT-FIELD-BORROW-QUERY-QUALITY-2026-07-08.md)
