# Fresh study-repo pass: inference-serving top 10 (2026-08-11)

**Status:** witnessed source-delta study; actionable net-new borrows filed as #6419 and #6422  
**Method:** current `.claude/skills/study-repo/SKILL.md` at `305d13b260`; public source refs fetched 2026-08-11  
**Prior cohort:** the coordinated 2026-07-18 ten-repository study recorded by `c9553d3548`  
**Scope:** vLLM, SGLang, TensorRT-LLM, Dynamo, Mooncake, KTransformers, LMDeploy, TGI, LMCache, and llama.cpp

## Verdict first

A fresh pass over the same top-ten inference-serving cohort found **two net-new, source-witnessed borrows** after dedupe against fak's live tree and issue queue:

1. **Dynamo: classify turn input once and expose a typed trigger to route policy** — filed as [#6419](https://github.com/anthony-chaudhary/fak/issues/6419).
2. **LMCache: attribute cache requests, hits, bytes, and latency to explicit tiers** — filed as [#6422](https://github.com/anthony-chaudhary/fak/issues/6422).

The other eight passes were still useful but yielded no honest new ticket: five refreshed or strengthened already-filed families, one had no upstream delta, and two had implementation advances that do not fit fak's current boundary better than existing work. This is not a README rescan: every row below includes a current source pin and implementation/test evidence.

## Cohort and delta census

The “top 10 existing studied repos” is not a newly invented ranking. It is the exact ten-repo serving cohort landed together on 2026-07-18 (`c9553d3548`), with all ten prior study notes indexed together. The fresh pass compares each old witness pin to the fetched default-branch tip.

| Repo | Prior pin | Fresh pin | Commits / changed files | Fresh outcome |
|---|---:|---:|---:|---|
| vLLM | `b6ff8a2f509c` | `d8f840071efd` | 925 / 2,125 | existing families strengthened |
| SGLang | `b8ec544946f1` | `546965fc72a7` | 1,007 / 3,987 | existing families strengthened |
| TensorRT-LLM | `f4c5c935aa89` | `afe103e88795` | 574 / 3,173 | no net-new fak-shaped borrow |
| Dynamo | `ea89e8bdfcc8` | `7a4e47ead90e` | 454 / 2,675 | **filed #6419** |
| Mooncake | `5d87a43d2474` | `8acb1e79aad6` | 199 / 674 | existing families strengthened |
| KTransformers | `def0f9313d6e` | `eb9b70c4115c` | 8 / 45 | existing MoE work strengthened |
| LMDeploy | `67c18b900cad` | `0f035871f8e7` | 44 / 540 | existing cache-lifecycle work strengthened |
| TGI | `b4adbf2f6e2e` | `b4adbf2f6e2e` | 0 / 0 | no source delta |
| LMCache | `e38ee4157a11` | `a976ce09dd98` (`origin/dev`) | 126 / 603 | **filed #6422** |
| llama.cpp | `571d0d540df0` | `704485942ab5` | 299 / 1,060 | existing portability/security families strengthened |

Pins are public commit IDs. Local clone paths, host facts, credentials, and private source do not appear in this trail.

## Per-repository evidence and ablated decisions

### 1. vLLM — resource ownership and transitive revision pins

**Source facts.** The CPU offload worker now centralizes shared mmap cleanup only after both transfer directions drain, with failure-path tests (`vllm/v1/kv_offload/cpu/gpu_worker.py@d8f840071efd`; `tests/v1/kv_offload/cpu/test_gpu_worker.py@d8f840071efd`, commit `0fec3d652b`). Secondary artifact loaders now preserve the requested revision across quantization, tensorizer, renderer, and architecture-config paths, with tests (`vllm/config/model.py:538-619@d8f840071efd`, commit `90fd4a333f`).

**Inferred principles.** Resource cleanup belongs to the owner after all asynchronous users drain. A root reproducibility pin is not real if secondary loaders silently float.

**fak opportunity and decision.** Both principles map to existing work rather than a new issue: fak already has durable cleanup/lifecycle programs and “version everything”/artifact witness contracts. The fresh evidence should sharpen acceptance tests when those leaves move, but filing another broad cleanup or pinning ticket would duplicate active families.

**Disconfirming check.** Reopen only if a concrete fak loader is found that accepts a root revision but fetches a secondary artifact without carrying it.

### 2. SGLang — plan-then-I/O and one-crossing synchronization

**Source facts.** HiSparse now separates index sharing/planning from swap-in I/O and overlaps prefetch, with a dedicated correctness test and benchmark (`python/sglang/srt/managers/hisparse_coordinator.py@546965fc72a7`; `test/manual/kernels/test_hisparse_prefetch.py@546965fc72a7`, commit `5469faec45`). DP-attention scheduler synchronization collapses multiple device-to-host reads into one (`python/sglang/srt/managers/scheduler_components/dp_attn.py@546965fc72a7`, commit `b3c02cbce7`).

**Inferred principles.** Compute a shared transfer plan before touching the slow tier; cross an expensive synchronization boundary once with a compact decision record.

**fak opportunity and decision.** These strengthen the previously filed SGLang prefix-cache and scheduling borrows (#5286-#5288) and the active activated-expert residency epic (#5606). They do not justify parallel issues until those contracts expose a missing plan/execute or synchronization seam.

**Disconfirming check.** File a child only if #5606's implementation performs per-expert slow-tier decisions inline rather than consuming one precomputed residency plan.

### 3. TensorRT-LLM — failure-domain kill and zero-copy token handoff

**Source facts.** The executor hard-stops all ranks when one rank's loop crashes (`cpp/tensorrt_llm/executor@afe103e88795`, commit `27e5b703`). KVCacheManager V2 gains zero-copy token passing (`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2@afe103e88795`, commit `0c98e929`) and DSA support (`bfc4966e`).

**Inferred principles.** A distributed request should have an explicit failure domain; avoid copying handoff metadata when ownership and lifetime are already precise.

**fak opportunity and decision.** The failure-domain lesson is already represented by fleet worker lifecycle and supervisor issues; the zero-copy implementation is C++/GPU-runtime-specific and not a direct fit for fak's Go control plane. No new issue.

**Disconfirming check.** Revisit if a measured fak profile identifies token-envelope copying as material after tuned serialization baselines, or if rank-like child failures currently leave a request falsely live.

### 4. Dynamo — typed ingress trigger reused by routing policy

**Source facts.** Dynamo classifies incoming agent requests into a typed `input_trigger` at protocol ingress, with conservative precedence and table tests (`lib/llm/src/protocols/common/input_trigger.rs:42-204@7a4e47ead90e`, commit `3367bc6`). Worker-selection policy receives structured `agent_context` (`lib/kv-router/src/scheduling/selector/policy.rs:70-122@7a4e47ead90e`, commit `b0d1472`). Its router policy pipeline was then unified around typed policy inputs (`lib/runtime/src/routing_policy@7a4e47ead90e`, commit `7a4e47e`).

**Inferred principle.** Derive request shape once at ingress, carry it as typed context, and prevent each routing policy from reparsing raw prompts differently.

**fak opportunity and decision.** Adapt this to model/provider/cache routing while keeping authorization independent. Filed [#6419](https://github.com/anthony-chaudhary/fak/issues/6419) with enum, precedence, integration, captured route witness, and “hint not authorization” criteria.

**Their-worldview difference.** Dynamo uses the fact to place work among GPU workers; fak should use it to select a model/provider/cache route under an unchanged policy floor.

**Disconfirming check.** #6419 must close as duplicate if a current route request already carries an equivalent typed, single-source field end to end.

### 5. Mooncake — bounded per-peer lanes and durable producer views

**Source facts.** TCP transfer work is scheduled through bounded per-peer connection lanes with explicit queueing and lifecycle tests (`mooncake-transfer-engine/src/transport/tcp_transport/tcp_transport_lane_impl.h@8acb1e79aad6`; `mooncake-transfer-engine/tests/tcp_write_visibility_test.cpp@8acb1e79aad6`, commit `a293384`). Durable oplog prefixes now carry a producer view and batch snapshot metadata (`mooncake-store/src/ha/oplog@8acb1e79aad6`, commit `9187a2c`; `mooncake-store/src/ha/snapshot/batch_oplog/metadata.cpp@8acb1e79aad6`, commit `38924b2`).

**Inferred principles.** Bound concurrency per remote peer, not merely globally. Durable replay metadata must identify the producer view that made an ordering claim.

**fak opportunity and decision.** Both map to existing fleet lane leases, bounded worker coordination, and witnessed checkpoint/session-image work. No new ticket without a demonstrated unbounded per-peer queue or producer-ambiguous replay record.

**Disconfirming check.** Revisit if current fleet transport has only a global concurrency cap and one slow peer can consume every slot.

### 6. KTransformers — narrow delta, stronger expert-residency evidence

**Source facts.** The eight-commit delta adds end-to-end full-parameter/LoRA SFT and RAWINT4 expert loading/dispatch work (`kt-kernel/python/sft@eb9b70c4115c`, commit `924754a`; `kt-kernel/operators/avx2@eb9b70c4115c`, commits `d1a3ed8` and `937f61c`).

**Inferred principle.** Expert-weight representation and dispatch should remain explicit across CPU instruction sets and training/inference modes.

**fak opportunity and decision.** This is useful comparative evidence for activated-expert offloading epic #5606, especially its bounded residency and representation contracts. It does not expose a separate fak control-plane technique, so no new issue.

**Disconfirming check.** Revisit if #5606 conflates expert representation with placement and cannot select a backend-specific loader independently.

### 7. LMDeploy — checkpoint lifecycle made a first-class prefix-cache module

**Source facts.** LMDeploy split a monolithic block trie into node, trie, checkpoint, checkpoint-lifecycle, and KV-lifecycle modules with extensive tests (`lmdeploy/pytorch/paging/block_trie/README.md@0f035871f8e7`; `lmdeploy/pytorch/paging/block_trie/checkpoint_lifecycle.py@0f035871f8e7`; `tests/pytorch/paging/test_block_trie@0f035871f8e7`, commit `d777afa`). Cache usage is now reported directly rather than reconstructed through redundant fields (`lmdeploy/pytorch/paging/scheduler.py@0f035871f8e7`, commit `4bff45f`).

**Inferred principles.** Prefix-cache checkpoints need their own lifecycle abstraction and tests. Cache use should be emitted at the authority that owns it.

**fak opportunity and decision.** These reinforce existing session image/checkpoint and cache-observation families, including #4107 and #1193. A new generic checkpoint ticket would be duplicate; direct per-tier cache attribution is handled more specifically by #6422.

**Disconfirming check.** Revisit if fak's cache/session checkpoint format has no explicit create/commit/retire lifecycle and relies on callers to infer state from file presence.

### 8. TGI — frozen source, no synthetic novelty

**Source fact.** The fetched default branch remains exactly `b4adbf2f6e2e`, the prior studied pin: zero commits and zero changed files.

**Decision.** The prior admission-control borrows (#5266, #5268, #5270) remain the authoritative output. The updated skill does not license inventing a new ticket when source evidence has not changed.

**Disconfirming check.** Run again if the canonical repository resumes development or its active successor is explicitly added to the cohort.

### 9. LMCache — per-tier cache truth

**Source facts.** LMCache added a distinct L2 usage schema and collector covering requests, hit/miss, bytes, latency, and coarse backend dimensions (`docs/design/usage_telemetry/l2_metrics.md:20-109@a976ce09dd98`; `lmcache/usage_telemetry/l2_usage.py@a976ce09dd98`; `tests/test_usage_telemetry.py@a976ce09dd98`, commit `a5a06220`). It also made the key directory authoritative for fleet blend lookup (`docs/design/v1/mp_coordinator/blend_index.md@a976ce09dd98`, commit `b12b078a`).

**Inferred principle.** Each cache tier must report its own truth; aggregate hit rates cannot honestly attribute value or regressions among local, managed, provider, and remote tiers.

**fak opportunity and decision.** Adapt the dimensions into fak's existing cache observation record and reconcile tier totals to aggregate totals, with explicit leak tests. Filed [#6422](https://github.com/anthony-chaudhary/fak/issues/6422).

**Their-worldview difference.** LMCache optimizes a multi-tier KV fleet and optionally reports usage; fak's telemetry must serve an honesty ledger and therefore must be local-first, scrubbed, and reproducible.

**Disconfirming check.** #6422 closes as duplicate only if current outputs already expose all required dimensions per tier with reconciliation and privacy tests.

### 10. llama.cpp — capability-driven loading and stronger tool isolation

**Source facts.** `--load-mode auto` now avoids mmap on devices that do not support it, while legacy flags converge on one explicit mode (`common/arg.cpp:818-826,2582-2621@704485942ab5`, commit `153d324`). Server tool isolation gained SSH-remote and rootless-Podman support (`examples/server@704485942ab5`, commit `4ae84de`).

**Inferred principles.** Select resource strategy from capability evidence, not platform folklore. Isolation policy should support multiple execution substrates behind one contract.

**fak opportunity and decision.** Both principles are already represented by hardware/fleet routing and the default-deny capability floor. No new issue absent a concrete fak path that selects mmap/direct-I/O solely from OS name or treats one sandbox runtime as the policy.

**Disconfirming check.** Revisit if a current model loader uses a hard-coded platform branch where a probed capability is available.

## Dedupe and terminal-action audit

- Live-tree searches covered model routing/request shape, cache-tier telemetry, bounded peer lanes, checkpoint lifecycle, revision propagation, and cleanup ownership.
- GitHub searches covered open and closed issues for the same concepts before filing.
- #6419 is scoped to a typed ingress classification reused by route policy; it does not claim authorization semantics.
- #6422 is scoped to tier-attributed cache truth and reconciliation; it does not duplicate a generic metrics dashboard.
- No issue was filed for TGI's zero-delta source or for implementation-specific GPU/C++ techniques without a fak-shaped seam.
- No follow-up remains only in prose: the two actionable net-new borrows are filed; every other candidate has an explicit disconfirming condition rather than a silently deferred task.

## Reproduction

```text
for each repo:
  git fetch origin
  git rev-parse <default-remote-ref>
  git rev-list --count <prior-pin>..<fresh-ref>
  git diff --name-only <prior-pin>..<fresh-ref>
  git log --no-merges <prior-pin>..<fresh-ref>
  inspect changed implementation + tests + rationale at the fresh pin

against fak:
  search live code/docs for the inferred principle
  search open and closed GitHub issues for semantic duplicates
  file only source-grounded, non-duplicate, fak-shaped borrows
```

## Honest limits

This is a source and contract study, not a runtime benchmark of ten serving systems. It proves what changed, what implementation/tests were inspected, and why two adaptations merit fak tickets. It does not claim their performance results transfer to fak, and it does not claim any borrow is shipped until its issue's captured witness exists.
