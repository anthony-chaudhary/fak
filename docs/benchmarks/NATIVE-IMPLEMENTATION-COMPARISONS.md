# Native implementation benchmark contracts

Status: **INCOMPLETE**. This is the machine-readable starting spine for requiring every fak-native implementation to be compared with the strongest practical alternatives. It deliberately does not turn missing measurements into claims.

Run:

```text
fak native-benchmarks
fak native-benchmarks --json
fak native-benchmarks --check   # non-zero until every declared contract has a witness
```

## Comparison rule

Each native capability must declare one shared-workload contract with:

1. the fak-native arm;
2. the tuned no-feature or incumbent baseline (never a deliberately weak baseline);
3. the next-best external implementation known for that capability;
4. an explicit inventory of every fak first-class integration that supplies an equivalent capability, with validation that each inventory entry has a measured `fak + integration` arm beside `fak + native` on the same workload;
5. quality/correctness, latency, token/resource use, and total-cost metrics; and
6. a reproducible witness before any performance claim is promoted.

An integration arm is additive to the next-best arm, not a substitute for it. If the best external implementation is also a first-class integration, one arm may carry both facts in its provenance, but the report must still name the integration explicitly.

The command discovers every production Go leaf directly beneath `internal/` and requires an explicit disposition for each one: `capability`, `multi_capability`, or `infrastructure`. Unclassified leaves remain missing; discovery never promotes a package to covered by inference. Capability and multi-capability leaves name every benchmark contract they contain, while infrastructure leaves carry a reason and need no performance comparison. This makes classification debt machine-visible without pretending that one package equals one capability. The registry starts with the two examples called out by the operator:

| Native capability | Native location | tuned baseline | next-best comparison | required outcomes |
|---|---|---|---|---|
| tool filtering | `internal/gateway/mcp_defer.go` | all schemas, provider cache enabled | retrieval-based selection (ToolRAG class) | task success, tool recall, input tokens, TTFT, total cost |
| context compression | `internal/headroom/native.go` | full history, provider cache enabled | LongLLMLingua | task success, retained-fact recall, input tokens, latency, total cost |
| prefix KV reuse | `internal/radixkv/radixkv.go` | prefix caching disabled | SGLang RadixAttention, plus `fak + llm-d` | output equivalence, prefix hit rate, TTFT, throughput, KV bytes, total cost |
| timeout-phase attribution | `internal/timeoutphase/timeoutphase.go` | one timeout bucket | OpenTelemetry spans, Datadog APM, AWS X-Ray | phase precision/recall, drops, latency, CPU/RSS/network/storage, total cost |
| KV-memory budget modeling | `internal/kvbudget/kvbudget.go` | full-MHA closed form | vLLM memory profiler, SGLang memory pool, NVIDIA GenAI-Perf | bytes/token and peak-allocation error, fit/concurrency, latency/throughput, GPU/host memory, total cost |
| failure-signature resume backoff | `internal/resumebackoff/resumebackoff.go` | immediate resume | Kubernetes CrashLoopBackOff, systemd RestartSec, AWS Step Functions retry | schedule equivalence, storm prevention/recovery, restarts, CPU/RSS/network, total cost |
| retry-attempt budgeting | `internal/attemptbudget/attemptbudget.go` | unlimited retries | Envoy retry budget, gRPC retry policy, AWS SDK adaptive retry | retry/stop equivalence, recovery/amplification, latency, requests, CPU/RSS/network, total cost |
| cache observability | `internal/cacheobs/cacheobs.go` | no telemetry | Prometheus, OpenTelemetry metrics, Datadog DogStatsD | counter/ratio equivalence, drops/cardinality, latency, CPU/RSS/network/storage, total cost |
| cache-cost accounting | `internal/cacheprice/cacheprice.go` | charge full prompt | AWS, Google Cloud, and Azure pricing calculators | admission-token equivalence, billed-unit error, latency, bytes, RSS, total cost |
| tool-call rate limiting | `internal/ratelimit/ratelimit.go` | no limiter | Envoy local rate limit, Kong rate limiting, Redis-cell | decision equivalence, overshoot, latency/throughput, state/network bytes, RSS, total cost |
| engine-cache invalidation | `internal/enginecache/enginecache.go` | no invalidation | standalone vLLM, standalone SGLang, LMCache, plus separate `fak + vLLM` and `fak + SGLang` arms | poisoned-reuse prevention, invalidated objects, latency, control requests/bytes, RSS, total cost |
| tool-result caching | `internal/vdso/vdso.go` | uncached optimized upstream | Redis client-side/server-assisted cache and Momento Cache | output equivalence, hit rate, latency, upstream calls, RSS, total cost |
| context-memory management | `internal/ctxmmu/mmu.go` | retain full history | Letta, plus separate `fak + mem0`, `fak + Letta`, `fak + Zep/Graphiti`, and `fak + LangMem` arms | task success, write precision, retained-fact recall, tokens, latency, RSS, total cost |
| policy adjudication | `internal/adjudicator/decide.go` | direct allow/deny lookup | OPA/Rego and Cedar | verdict equivalence, policy coverage, latency, throughput, RSS, total cost |
| model routing | `internal/modelroute/modelroute.go` | fixed strongest model | RouteLLM, plus separate `fak + LiteLLM`, `fak + OpenRouter`, and `fak + Portkey` arms | task success, route quality, latency, tokens, RSS, total cost |
| tokenization | `internal/tokenizer/tokenizer.go` | exhaustive adjacent-pair BPE | llama.cpp tokenizer, plus `fak + Hugging Face tokenizers` | exact token IDs, decode round-trip, throughput, latency, RSS, initialization, total cost |

All currently have missing witnesses, which is why `--check` fails. The registry is in `internal/nativebench`; new native capabilities must be added there with their alternatives before their benchmark obligation can be considered covered.

## Scope still to enumerate

The benchmark-governance packages (`internal/bench*` and `internal/nativebench`) are explicitly classified as infrastructure with per-leaf reasons. This is not a blanket naming rule: each entry is enumerated and validation still leaves every newly discovered package unclassified by default.

This spine does **not** yet prove repository-wide coverage. Leaf discovery is exhaustive at the package boundary and the disposition schema is now enforced, but most leaves are still explicitly unclassified. The authoritative completion work is to classify every discovered leaf, split every multi-capability leaf into contracts, map equivalent first-class integrations, and attach benchmark witnesses. Until the unclassified count reaches zero and every contract passes, the broad claim “all native implementations are benchmarked against next best alternatives” remains **not yet**.
| Deadline-aware EDF admission + predicted-miss shedding | `internal/deadlineadmit` | native; FIFO/no shedding; Mooncake; vLLM; SGLang; `fak + vLLM`; `fak + SGLang` | [local contract](DEADLINE-ADMISSION-ALTERNATIVES-2026-08-10.md), external witnesses [#6135](https://github.com/anthony-chaudhary/fak/issues/6135) | INCOMPLETE |
| GitHub API mutation reserve + hourly call estimator | `internal/mutationbudget` | native; direct/no reserve; Octokit; `gh api`; Envoy global rate limit | [local contract](MUTATION-BUDGET-ALTERNATIVES-2026-08-10.md), external witnesses [#6136](https://github.com/anthony-chaudhary/fak/issues/6136) | INCOMPLETE |
| Worker dispatch-to-heartbeat histogram + p50/p95 | `internal/launchlatency` | native; raw/no summary; Prometheus; OpenTelemetry; Datadog | [local contract](LAUNCH-LATENCY-ALTERNATIVES-2026-08-10.md), external witnesses [#6137](https://github.com/anthony-chaudhary/fak/issues/6137) | INCOMPLETE |
| Compute-region taxonomy + live-lease collision admission | `internal/computeadmit` | native; no admission; Kubernetes; Slurm; Ray; AWS Batch | [local contract](COMPUTE-REGION-ADMISSION-ALTERNATIVES-2026-08-10.md), external witnesses [#6139](https://github.com/anthony-chaudhary/fak/issues/6139) | INCOMPLETE |
| Shared lane/tree collision admission | `internal/laneadmit` | native; geometry-only; `fak + DOS arbitrate`; GitHub Actions; Kubernetes Lease; etcd | [local contract](LANE-TREE-ADMISSION-ALTERNATIVES-2026-08-10.md), external/integration witnesses [#6144](https://github.com/anthony-chaudhary/fak/issues/6144) | INCOMPLETE |
| Shared execution-surface region admission | `internal/regionadmit` | native; geometry-only; `fak + DOS arbitrate`; `fak + Git-ref leases`; Kubernetes Lease; etcd; GitHub Actions | [local contract](SHARED-REGION-ADMISSION-ALTERNATIVES-2026-08-10.md), external/integration witnesses [#6149](https://github.com/anthony-chaudhary/fak/issues/6149) | INCOMPLETE |
| Prefix-cache budget/reuse sweep + ROI knee | `internal/cachesweep` | native; no cache; libCacheSim; Caffeine; Redis/Valkey | [local contract](PREFIX-CACHE-SWEEP-ALTERNATIVES-2026-08-10.md), external witnesses [#6146](https://github.com/anthony-chaudhary/fak/issues/6146) | INCOMPLETE |
| Warm-cache budget planning + value-ranked scheduling | `internal/vcachegov` | native; demand-only; `fak + LMCache`; `fak + Mooncake`; `fak + NIXL`; vLLM APC; SGLang HiCache | [local contract](WARM-CACHE-BUDGET-SCHEDULING-ALTERNATIVES-2026-08-10.md), external/integration witnesses [#6152](https://github.com/anthony-chaudhary/fak/issues/6152) | INCOMPLETE |
| Cross-provenance cache reuse-divergence detection | `internal/cachewitness` | native; raw counters; `fak + Prometheus`; `fak + OpenTelemetry`; Prometheus rules; Datadog | [local contract](CACHE-REUSE-DIVERGENCE-ALTERNATIVES-2026-08-10.md), external/integration witnesses [#6155](https://github.com/anthony-chaudhary/fak/issues/6155) | INCOMPLETE |
| Trailing-window cache reuse regression gate | `internal/cachevalueledger` | native; raw JSONL; `fak + Prometheus`; `fak + OpenTelemetry`; Prometheus rules; Datadog | [local contract](CACHE-REUSE-TREND-GATE-ALTERNATIVES-2026-08-10.md), external/integration witnesses [#6157](https://github.com/anthony-chaudhary/fak/issues/6157) | INCOMPLETE |
| Concentration-weighted shared cache-budget allocation | `internal/vcachecal` | native; equal-share; volume-proportional; `fak + LMCache`; `fak + Mooncake`; vLLM; SGLang | [local contract](CACHE-BUDGET-CONCENTRATION-ALLOCATION-ALTERNATIVES-2026-08-10.md), external/integration witnesses [#6159](https://github.com/anthony-chaudhary/fak/issues/6159) | INCOMPLETE |
| Dedicated prefix-cache warming and readback accounting | `internal/vcachewarm` | native; demand-only; `fak + Anthropic`; `fak + Gemini`; `fak + OpenAI`; `fak + LMCache`; `fak + Mooncake`; vLLM; SGLang | [local contract](DEDICATED-CACHE-WARMING-ALTERNATIVES-2026-08-10.md), external/integration witnesses [#6162](https://github.com/anthony-chaudhary/fak/issues/6162) | INCOMPLETE |
| Provider-cache economics and realized-value fold | `internal/vcacheobserve` | native; raw usage; `fak + Prometheus`; `fak + OpenTelemetry`; Anthropic; OpenAI; Datadog; LangSmith | [local contract](PROVIDER-CACHE-ECONOMICS-ALTERNATIVES-2026-08-10.md), external/integration witnesses [#6164](https://github.com/anthony-chaudhary/fak/issues/6164) | INCOMPLETE |
| Bounded fsynced provider-cache snapshots and tolerant replay | `internal/vcachesnapshot` | native; append-only JSONL; `fak + Prometheus`; `fak + OpenTelemetry`; SQLite WAL; Prometheus TSDB; ClickHouse | [local contract](BOUNDED-CACHE-SNAPSHOT-ALTERNATIVES-2026-08-10.md), external/integration witnesses [#6165](https://github.com/anthony-chaudhary/fak/issues/6165) | INCOMPLETE |
| Codex session token-counter extraction and content sanitization | `internal/vcacheextract` | native; raw JSONL pass-through; `fak + OpenTelemetry`; `fak + Prometheus`; jq; Vector VRL; Fluent Bit | [local contract](CODEX-TOKEN-SANITIZATION-ALTERNATIVES-2026-08-10.md), external/integration witnesses [#6169](https://github.com/anthony-chaudhary/fak/issues/6169) | INCOMPLETE |
| Cache-warmth context-elision honesty lint | `internal/vcacheqa` | native; tuned non-test text scan; go/analysis; Semgrep; CodeQL; golangci-lint custom analyzer | [local contract](CACHE-HONESTY-LINT-ALTERNATIVES-2026-08-10.md), external witnesses [#6171](https://github.com/anthony-chaudhary/fak/issues/6171) | INCOMPLETE |
| Provider-cache telemetry reconciliation and confirmed-only cost booking | `internal/vcachestar` | native; trust manifest; `fak + Anthropic`; `fak + OpenAI`; `fak + Gemini`; `fak + Prometheus`; `fak + OpenTelemetry`; Prometheus rules; Datadog; LangSmith | [local contract](CACHE-TELEMETRY-RECONCILIATION-ALTERNATIVES-2026-08-10.md), integration/external witnesses [#6173](https://github.com/anthony-chaudhary/fak/issues/6173) | INCOMPLETE |
| Default-on cache readiness across cold-path, usefulness, and provenance evidence | `internal/vcachescore` | native; usefulness threshold; `fak + Prometheus`; `fak + OpenTelemetry`; OPA/Rego; Prometheus rules; Datadog; LangSmith | [local contract](DEFAULT-CACHE-READINESS-ALTERNATIVES-2026-08-10.md), integration/external witnesses [#6178](https://github.com/anthony-chaudhary/fak/issues/6178) | INCOMPLETE |
| Net-true, strawman, and not-yet claim grading | `internal/claimcheck` | native; witness-presence heuristic; `fak + Prometheus`; `fak + OpenTelemetry`; OPA/Rego; OpenAI Evals; LangSmith; Braintrust; DeepEval | [local contract](NET-TRUE-CLAIM-GRADING-ALTERNATIVES-2026-08-10.md), integration/external witnesses [#6180](https://github.com/anthony-chaudhary/fak/issues/6180) | INCOMPLETE |
| Go structural expression search with metavariable bindings | `internal/astquery` | native; literal search; Semgrep; ast-grep; Comby; gogrep | [local contract](GO-AST-QUERY-ALTERNATIVES-2026-08-10.md), external witnesses [#6182](https://github.com/anthony-chaudhary/fak/issues/6182) | INCOMPLETE |
