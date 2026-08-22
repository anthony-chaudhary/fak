# Concept study: FreeToken edge-native MoE serving (2026-08-22)

## Verdict

FreeToken is a young but technically substantial Apache-2.0 edge-serving engine. Its strongest transferable idea for fak is not the Python/Triton runtime; it is treating **model-state retention and memory geometry as agent-loop controls**. The deep pass found two independently shippable PARTIAL/ABSENT axes and filed them: semantic tool-call state anchors (#8601) and idle expert/KV/recurrent cache reallocation (#8603). FreeToken's expert-residency policy, prefix radix cache, OpenAI/Anthropic surfaces, and daemon are already PRESENT in fak on the relevant axes or weaker than fak's existing mechanisms.

## Scope and provenance

- Repository: <https://github.com/FlashML-org/FreeToken>
- Main revision pinned and read: `FlashML-org/FreeToken@0ab982f10905fa775962a4eddcb44caa50065251` (2026-08-20 commit, observed 2026-08-22).
- Release: `v0.1.2@9db1a39455a3fb107f3db83e381d10ceadfe5d99`, published 2026-08-19.
- Proposed branch read: `feat/decode-token-checkpoint@ea5348b4e316b7a73f27e94af012039f7ec778f8` (unmerged as observed 2026-08-22).
- Paper read: Yang et al., *FreeToken: Efficient Edge-Native MoE Serving with Bandwidth-Adaptive Execution*, arXiv:2608.16157v1 (2026-08-21).
- License: Apache-2.0. Direct adaptation is permitted with attribution and notice preservation; the filed work calls for native Go adaptations, not copied Python.
- Public state observed 2026-08-22: 1,320 stars, 103 forks, 20 issue records (GitHub's repository summary showed 17 open), 23 PR records (15 open), one release, and no Discussions surface. Those counts are observations, not fak-authored performance evidence.

## What was read

This was a deep pass, not `--quick` or `--draft`:

- Product/worldview: `README.md`, install/quickstart/model/CLI docs, daemon docs, cache status/report docs, paper, release notes, and packaging/install scripts.
- Request path: OpenAI, Anthropic, Responses API, generation, reasoning/function-call parsing, tokenizer process, scheduler I/O, and shell/control APIs.
- Scheduling/state: prefill and decode schedulers, request table, cache manager, radix/hybrid/SWA/recurrent cache pools, abort and rebuild paths.
- Model execution: engine graph, attention backends (dense, DSA/DSV4 sparse, linear, SWA), distributed layer, model registry/loaders, and MoE execution.
- Resource policy: startup cache budget, runtime cache rebuild, expert offload cache, CPU executor, pinned host banks, cache report/status accounting, and daemon supervision.
- Native kernels: Triton/CUDA/C++ kernel trees, AOT machinery, FP8/NVFP4/MXFP4/Q4 paths, and the open FP8 `scaled_mm` branch.
- Validation: tests across engine, scheduler, KV cache, MoE/offload, kernels, server/parser, daemon, tokenizer, distributed, and install surfaces.
- Project history: all reachable commits and remote branches, v0.1.2 delta, every public issue body/comment, every public PR body/file list, and the current release.

Completeness critic: I re-ran the subsystem inventory against the tree after the first pass. The initially under-read surfaces were runtime cache rebuild, daemon/accounting, parser branches, and native kernel/open-branch work; those were then read. Nothing material remained unopened at subsystem level. I did not execute CUDA tests or reproduce paper throughput because this study's deliverable is source-grounded borrowing; performance claims remain explicitly external until a fak-controlled same-hardware benchmark exists.

## Architecture and claims

FreeToken is an edge-native MoE serving engine for large open-weight models on heterogeneous consumer hardware. Its execution stack combines:

- GPU-resident hot experts with CPU/host-backed cold experts and capped-fetch LRU replacement.
- Radix/paged KV caches plus model-specific SWA, linear/recurrent, DSA, and DSV4 pools.
- Separate prefill/decode scheduling and model-specific attention/kernel backends.
- Runtime cache geometry rebuild across expert slots, full-attention KV pages, SWA pages, and recurrent/Mamba slots.
- OpenAI-, Anthropic-, and Responses-compatible APIs, a shell/TUI, and a supervising daemon.

The paper's 0.64--0.95 tokens/s and 1.43--2.54x claims are author-reported on selected DeepSeek-family models and hardware. They are useful hypotheses, not fak-controlled facts. The repository is only about one month old, model support is concentrated in recent architectures, and open issues/PRs still cover packaging, multi-GPU, Windows, GGUF, compatibility, and benchmarking. Use FreeToken as a high-signal design peer, not a mature universal baseline.

## Current-fak witness and candidate ledger

Dogfood used varied `fak capabilities` queries plus raw repository and issue searches. Every classification below was refined by reading the actual fak seam; an umbrella cache/MoE hit was not treated as PRESENT on the narrower axis.

| Borrow | Source `path:line@rev` | Axis | Their-worldview reason | fak witness on-axis | Portfolio / license | Filed |
|---|---|---|---|---|---|---|
| Semantic tool-call state anchor | `python/freetoken/server/function_call_parser.py:95-126@ea5348b`; `python/freetoken/scheduler/cache.py:53-123@ea5348b` | Capture exact KV + recurrent state at a parser-recognized tool-call opener, then validate it after transcript normalization | Agent clients rewrite tool-call syntax; exact byte-prefix matching can discard still-valid state | **PARTIAL.** `internal/model/kvcache.go:191-247` clones/reserves state and #444/#2241/#6851 cover hybrid safety/restore, but no parser-emitted semantic anchor binds the snapshot to a normalized next turn | **OPTIONAL-MODULE** until one exact model/parser spine proves equivalence and net saved prefill; inspire from unmerged Apache-2.0 proposal | #8601 |
| Idle native cache-geometry rebuild | `python/freetoken/server/api_server.py:567-613@0ab982f`; `python/freetoken/scheduler/scheduler.py:155-190@0ab982f`; `python/freetoken/engine/engine.py:726-847@0ab982f` | Transactionally trade expert slots against KV/SWA/recurrent capacity without unloading weights; busy rejection + rollback | Consumer GPUs have one scarce VRAM pool and workload mix changes | **PARTIAL.** `internal/model/paging_ring.go:51-66,172-182` fixes expert-ring budget at construction; `internal/model/kvcache.go:26-34,227-247` grows one cache, but no joint idle transaction or rollback exists | **OPTIONAL-MODULE** explicit operator control; integrate native Go mechanism under Apache-2.0 attribution | #8603 |
| Workload-aware expert residency and eviction diagnostics | `python/freetoken/moe/offload_cache.py:764-916@0ab982f` | Capped-fetch LRU plus frequency histogram, oracle-hit, working-set, and entropy diagnostics | PCIe/host bandwidth dominates edge MoE decode | **PRESENT-on-axis, stronger in fak.** `internal/model/paging_ring.go` provides bounded resident weights; #4233 and #5615 added Belady-regret measurement and policy selection; #3174/#5606 landed activated expert offload | **DEFAULT** existing fak path; compare in benchmarks, do not fork | — |
| Radix prefix cache with locked active nodes and LRU leaf eviction | `python/freetoken/kvcache/radix_cache.py:40-303@0ab982f`; `python/freetoken/scheduler/cache.py:50-213@0ab982f` | Reuse longest token prefix and evict only unlocked leaves | Multi-turn serving repeatedly shares long prefixes | **PRESENT/DIVERGENT.** fak owns exact native per-session clone/reserve and can delegate/observe SGLang/vLLM radix caches; importing a second Python radix owner would violate fak's native/external ownership boundary | **EXCLUDE** as duplicate runtime; retain existing native/external seams | — |
| Compute/transfer-derived expert-cache sizing | Paper §3.1--3.2; `python/freetoken/engine/cache_budget.py:31-118@0ab982f` | Derive residency from measured CPU and CPU↔GPU bandwidth rather than a fixed count | Edge device ratios vary sharply | **PARTIAL but deduped.** fak already has bandwidth probes, placement, spill fit, and issue-backed expert-cache planning; no new standalone mechanism survived ablation beyond #8603's joint geometry transaction | **WATCH/RECIPE** as an oracle in a same-hardware benchmark, not a new policy leaf | — |
| Full-layer pipelined prefill and architecture-specific sparse attention | `python/freetoken/attention/dsa.py`, `dsv4_sparse.py`, `m3_sparse.py`; engine graph at `0ab982f` | Overlap expert transfer/compute and use native sparse attention per architecture | Frontier edge models are bandwidth-bound and architecturally heterogeneous | **PRESENT/PARTIAL by model.** fak already has model-specific GLM DSA, MoE batching/offload, CUDA/Metal paths, and an active benchmark backlog; FreeToken code is not portable to Go/native backends | **WATCH**; only borrow after a model-specific SOTA/benchmark check | — |
| Provider-compatible server, daemon, shell, and desktop runtime | `python/freetoken/server/*`, `daemon/*`, `shell/*@0ab982f` | Make local inference installable and operable by non-kernel users | Consumer deployment needs lifecycle and familiar APIs | **PRESENT or divergent.** fak already has OpenAI-compatible serving, process/session control, metrics, and guarded agent operation; a Python supervisor/desktop app would displace the one-Go-binary boundary | **EXCLUDE** | — |

## Problem/value frames for filed effects

### #8601 — semantic tool-call anchor

- **Centrality:** Enabling for managed context and net-true native inference efficiency.
- **For:** local agent workloads that normalize model-emitted tool calls.
- **Problem / Today:** normalization invalidates token-prefix reuse after the opener; fak can copy exact state but cannot select/attest this semantic boundary.
- **Better because:** exact model/layout/scope/token binding preserves only the still-valid prefix and falls back safely.
- **Witness:** a fail-before/pass-after rewritten tool-call trace emits output-equivalent continuation with less prefill and bounded retained bytes.
- **P1-P4:** advances managed context and measured efficiency; bounds adaptation to supported completed calls; integrates parser, cache, counters, and refusal reasons.

### #8603 — runtime cache geometry

- **Centrality:** Enabling for managed-context efficiency on native model serving.
- **For:** operators whose MoE workload shifts between context-heavy and expert-bandwidth-heavy phases.
- **Problem / Today:** fixed startup geometry wastes VRAM or forces a model reload.
- **Better because:** an idle-only validated transaction changes the split and rolls back after failure.
- **Witness:** one loaded model survives resize, reports exact old/new bytes, preserves output, avoids reload, and restores the prior geometry under injected failure.
- **P1-P4:** manages physical context capacity; requires net restart/rebuild accounting; admits one bounded idle transaction; exposes status and stable diagnostics.

## Issue/release/history findings

- v0.1.2 changed package versioning; main since the tag only adds community assets. Runtime behavior studied above is therefore effectively the release behavior unless explicitly labeled as an open branch.
- The semantic checkpoint is **not on main**. It comes from `feat/decode-token-checkpoint@ea5348b`; the study and #8601 preserve that proposal status.
- Open branches/PRs show active work on packaging, FP8 `scaled_mm`, model support, parser reasoning effort, and launcher hardening. These are useful maturity signals but not shipped defaults.
- Public issue themes include memory fit, unsupported models/architectures, packaging/platform friction, and performance evidence. They reinforce the bounded-superset stance: fak should borrow modular controls, not adopt FreeToken wholesale.

## Reproduction packet

```powershell
fak tree-doctor --scratch-dir study-freetoken
git clone https://github.com/FlashML-org/FreeToken.git _scratch/study-freetoken/source
git -C _scratch/study-freetoken/source checkout 0ab982f10905fa775962a4eddcb44caa50065251
gh issue list -R FlashML-org/FreeToken --state all --limit 100
gh pr list -R FlashML-org/FreeToken --state all --limit 100
gh release list -R FlashML-org/FreeToken
fak capabilities "idle-only in-place resize of MoE expert slots KV pages and recurrent state with rollback" --json
fak capabilities "preserve KV recurrent state at semantic tool-call boundary before prompt edit" --json
```

## Companions

- Workflow: [`study-repo`](../../.claude/skills/study-repo/SKILL.md)
- Witness/dedupe workflow: [`field-borrow`](../../.claude/skills/field-borrow/SKILL.md)
- Parent studies epic: [#4207](https://github.com/anthony-chaudhary/fak/issues/4207)
- Cache capacity parent: [#985](https://github.com/anthony-chaudhary/fak/issues/985)
- Filed effects: [#8601](https://github.com/anthony-chaudhary/fak/issues/8601), [#8603](https://github.com/anthony-chaudhary/fak/issues/8603)
- Exact hybrid snapshot witness: [#6851](https://github.com/anthony-chaudhary/fak/issues/6851)

## Bottom line

Borrow FreeToken's two bounded controls, not its runtime: make semantic boundaries eligible for exact native state reuse, and let an idle native engine rebalance expert versus context capacity transactionally. Keep expert policy, radix ownership, lifecycle, and API operation on fak's existing stronger seams. No other unfixed borrow survived on-axis ablation.
