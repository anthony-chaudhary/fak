# Concept study — sglang (RadixAttention / structured-gen serving engine) → witnessed borrows for fak

- **Repo:** https://github.com/sgl-project/sglang.git
- **Pinned:** `@b8ec544946f1c5b6e17a919a691b05c5b3e7af84` (HEAD; `[DSA] Integrate Q8KV8 FP8 Sparse MLA Prefill …`)
- **Date:** 2026-07-18 · **License:** Apache-2.0 (integrate-eligible; every borrow below stayed **inspire** / clean-room)
- **Method:** a real `/study C:\work\sglang --deep` pass — 6 parallel subsystem deep-readers + a completeness-critic + a fak-side on-axis witness subagent. Every source cited as `path:line@b8ec5449`.

## Fan-out coverage

Deep-read (load-bearing serving-systems subsystems), each verified against the real tree:

1. **mem_cache** — RadixAttention token-level radix tree (mid-node edge split, refcount pinning of in-flight paths, LRU-leaf evict), HiCache GPU→CPU→disk tiering (hit-count-gated promotion), async L3 prefetch with *revocation* + occupancy rate-limiting, `extra_key` namespacing, crash-safe reserve→commit disk store.
2. **managers/scheduler** — longest-prefix-match queue ordering, in-batch cold-prefix dedup, retraction/preemption under KV pressure with a keep-priority order, new-token-ratio backpressure, overlap scheduling (FutureMap), routing-key affinity.
3. **disaggregation** — prefill/decode split, `decode_prefix_len` held-prefix handshake, KV-event stream, mooncake/nixl/mori transports, decode-side offload.
4. **constrained + speculative** — xgrammar/outlines/llguidance backends, jump-forward, grammar-compile dedup, **reasoner-grammar think-span suspension + `max_think_tokens` budget**, EAGLE3 draft/verify, ngram/prompt-lookup, adaptive spec depth keyed on batch size.
5. **eplb + DP/EP** — online per-expert load recording, balancedness-gated rebalance, live expert-location remap, offline placement simulator, DeepEP dispatch.
6. **worldview + operability** — server_args defaults, Prometheus metric set, soft+hard watchdog, `/health_generate`, abort_all.

**Completeness-critic residue (skipped, justified):** raw GPU-kernel backends (flashinfer/triton/fa3 attention, `sgl-kernel/`, quantization kernels) — compute kernels fak routes through `sota-check`, not gateway-transferable; and lora / multimodal / dllm (diffusion LM) / function_call parser / elastic_ep / checkpoint_engine — niche or off fak's axis. Named, not silently dropped.

## Worldview (reconstructed from defaults/metrics/benchmarks)

sglang serves **high-QPS token generation for many concurrent independent requests on GPU clusters**. RadixAttention is ON by default and **cache-hit-rate is the headline gauge**; the whole scheduler is organized to *maximize cross-request prefix reuse and GPU utilization* (LPM queue ordering, cold-prefix dedup, overlap scheduling). Structured-generation throughput is a first-class concern (jump-forward, reasoner-grammar). Priority/fairness/preemption are all **opt-in** — throughput-first, SLA-second. **fak's world is orthogonal:** fak is an **audited agent-fleet gateway + GLM-offload research kernel that routes turns across backend engines** (it even ships an `internal/engine/sglang.go` adapter). fak optimizes replay determinism, provenance/trust isolation, cross-agent cache *value*, and cost governance — not raw cluster tok/s. So sglang's *mechanisms* transfer only where they land on a fak axis; its pure-throughput loop mechanisms are DIVERGENT by design.

## Candidate table

| # | Borrow (one line) | Source `path:line@b8ec5449` | The one axis | Witness vs fak | Verdict → action |
|---|---|---|---|---|---|
| 1 | Concurrent cold-prefix fill suppression (serialize one fill among N identical cold prefixes) | `python/sglang/srt/managers/schedule_policy.py` (twin-defer) | admission-gate the first cold fill so peers hit warm | ABSENT in code but **already filed #1914**; fak has reusable single-flight at `internal/microagent/hibernate.go:180` | COVERED → cross-link #1914 (no new issue) |
| 2 | Trust/principal-scoped prefix-cache namespacing (`extra_key`) | `python/sglang/srt/mem_cache/radix_cache.py:208` | identical tokens under different trust domains never share a node | **PRESENT-on-axis**: `internal/radixkv/namespace.go:47` (`rootFor(ns)`) is exactly this | DROP (proud read) |
| 3 | Offline expert-placement simulator (record→replay→score) | `python/sglang/srt/eplb/eplb_simulator/` | score a candidate placement offline vs recorded traces | **PRESENT-on-axis**: `internal/model/expert_residency_lfu.go:89` + `expert_placement_drift.go:42` replay-and-score | DROP |
| 4 | Grammar/schema compile-cache with copy-on-hit + off-hot-path compile | `python/sglang/srt/constrained/base_grammar_backend.py:198` | reuse a compiled matcher clone on hit; compile off the decode thread | **PARTIAL**: fak dedups by digest (`internal/grammar/grammar.go:73`) but *deliberately* recomputes the byte-FSM per token (`compile.go:30-31`) | PARTIAL, deliberate-design + low value → NOTE only |
| 5 | Per-block-hash KV event stream (tier/priority diffs) | `python/sglang/srt/disaggregation/kv_events.py:167` | ordered, replay-able cache-event stream a KV-router subscribes to | **PARTIAL**: `internal/engine/cacheevents.go:354` has tiered fan-out but no monotonic-seq + replay-from-N; **content covered by #5260/#5256** | PARTIAL, near-dupe of #5260 → NOTE (seq+replay refinement, comment-worthy on #5260) |
| 6 | **Reasoner think-token budget** — count think tokens, force `</think>` at a cap | `reasoner_grammar_backend.py:44-46,137@b8ec5449`; `grammar_manager.py:117-129`; `sampling/custom_logit_processor.py:75-80` | a witnessed cap on reasoning-token count with forced exit | **ABSENT**: fak only forwards provider `reasoning_effort` string (`internal/agent/chat.go:1223`), no count/cap/exit | ABSENT → **FILED #5286** (inspire) |
| 6a | Gate constrained-decode mask on reasoning-end (don't mask `<think>`) | `reasoner_grammar_backend.py:34-46` | reasoning→answer boundary correctness under constrained decode | ABSENT but **already filed #5262** (from vllm study) | COVERED → #5286 reuses it as prerequisite |
| 7 | **Soft watchdog** — diagnostic state dump before kill | `managers/scheduler.py:1080-1089@b8ec5449` (`init_soft_watchdog`); `utils/cudacore_pyspy_dump_utils.py:70@b8ec5449` (`pyspy_dump_schedulers`) | capture a reversible diagnostic dump of an alive-but-stalled worker before the destructive step | **ABSENT**: `internal/resume/trajectory_watchdog.go:32-36` decides NUDGE/REVIVE with zero state capture | ABSENT → **FILED #5287** (inspire) |
| 8 | **Prefix-delta KV handoff** — destination advertises held prefix, sender skips it | `disaggregation/prefill.py:297-300@b8ec5449`; `disaggregation/decode.py:974-980@b8ec5449` | transfer only the uncached suffix on a node-to-node KV handoff | **PARTIAL**: transfer span exists (`internal/model/paged_kv_transfer.go:60`) + delta math exists (`internal/cachemeta/prefix_stability.go:224`), but no held-prefix negotiation wires them | PARTIAL → **FILED #5288** (inspire) |

### Already-mined territory (dedupe drops, not re-filed)

RadixAttention core → `internal/radixkv/radixkv.go` (explicit clean-room rebuild, adds `EvictNode` policy-eviction). Prefix-aware routing → #4303 gossip directory, #3893 coverage pricing. Cross-agent expert coalescing → #5243/#5248. EPLB balanced-packing + hot-expert replication → #3886; deferred-expert pipeline → #5239. Tool-call/turn speculation → #4102/#4105/#4106/#4234/#4236/#809. KV teleport/migration → #4301/#4302/#4307. KV-event surface → #3320/#5260/#3145/#40. Prior study notes cover EPLB (`CONCEPT-STUDY-EPLB-2026-07-10.md`) and dedicated-MoE-caching.

## Filed this pass

- **#5286** — feat(agent): witnessed think-token budget — count reasoning tokens and force `</think>` exit at a cap.
- **#5287** — feat(resume): soft watchdog — capture a diagnostic state dump on an alive-but-stalled session before nudge/revive.
- **#5288** — feat(serving): prefix-delta KV handoff — destination advertises its held prefix, sender ships only the suffix.

## Companions

- **field-borrow:** all three are `inspire` / clean-room Go against Apache-2.0 sglang; provenance = the `@b8ec5449` source lines cited per issue. No bytes vendored.
- **epic:** superset **#2236** (fak > best of vLLM + SGLang + Dynamo + TRT-LLM); siblings = the per-engine study epics (#5256 tensorrt-llm, #3983 kvcache-factory, #3900 ktransformers, #4352 colibri, #4207 inference-radar). No `sglang-study` epic exists yet — these three land as standalone leaves referencing #2236.
- **cross-links:** #5286 ⟂ #5262 (reasoning-end boundary, reused); #5288 ⟂ #4296/#4301/#4302/#4303 (KV mobility); #5 note ⟂ #5260 (seq+replay refinement); #1 ⟂ #1914 (single-flight primitive to reuse).
