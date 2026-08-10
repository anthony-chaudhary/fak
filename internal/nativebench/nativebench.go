// Package nativebench owns the benchmark obligations for fak-native capabilities.
// It records comparison arms, not results: a result becomes claimable only when a
// reproducible witness is attached to every required arm.
package nativebench

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

type AlternativeClass string

const (
	TunedBaseline         AlternativeClass = "tuned_baseline"
	NextBest              AlternativeClass = "next_best"
	FirstClassIntegration AlternativeClass = "first_class_integration"
)

type Alternative struct {
	Name        string           `json:"name"`
	Class       AlternativeClass `json:"class"`
	Integration string           `json:"integration,omitempty"`
	Source      string           `json:"source"`
}

type Contract struct {
	Capability   string        `json:"capability"`
	NativePath   string        `json:"native_path"`
	Workload     string        `json:"workload"`
	Metrics      []string      `json:"metrics"`
	Alternatives []Alternative `json:"alternatives"`
	Witness      string        `json:"witness,omitempty"`
	Integrations []string      `json:"integrations,omitempty"`
}

type Finding struct {
	Capability string `json:"capability"`
	Reason     string `json:"reason"`
}

type LeafDisposition string

const (
	DispositionCapability      LeafDisposition = "capability"
	DispositionInfrastructure  LeafDisposition = "infrastructure"
	DispositionMultiCapability LeafDisposition = "multi_capability"
)

type LeafClassification struct {
	Leaf         string          `json:"leaf"`
	Disposition  LeafDisposition `json:"disposition"`
	Capabilities []string        `json:"capabilities,omitempty"`
	Reason       string          `json:"reason"`
}

type Coverage struct {
	NativeLeaves          int                     `json:"native_leaves"`
	CoveredLeaves         int                     `json:"covered_leaves"`
	ClassifiedLeaves      int                     `json:"classified_leaves"`
	UnclassifiedLeaves    int                     `json:"unclassified_leaves"`
	DispositionCounts     map[LeafDisposition]int `json:"disposition_counts"`
	Classifications       []LeafClassification    `json:"classifications,omitempty"`
	MissingLeaves         []string                `json:"missing_leaves,omitempty"`
	UnclassifiedLeafNames []string                `json:"unclassified_leaf_names,omitempty"`
	OrphanContracts       []string                `json:"orphan_contracts,omitempty"`
	DiscoveryComplete     bool                    `json:"discovery_complete"`
}

type Report struct {
	Contracts []Contract `json:"contracts"`
	Coverage  Coverage   `json:"coverage"`
	Findings  []Finding  `json:"findings"`
	Complete  bool       `json:"complete"`
}

var leafClassifications = []LeafClassification{{
	Leaf: "internal/cachesweep", Disposition: DispositionMultiCapability,
	Capabilities: []string{"prefix_cache_budget_sweep"},
	Reason:       "native prefix-access trace replay across cache budgets with infinite-cache ceiling and ROI knee",
},
	{
		Leaf: "internal/laneadmit", Disposition: DispositionMultiCapability,
		Capabilities: []string{"lane_tree_collision_admission"},
		Reason:       "native shared lane/tree collision, exclusivity, ancestry, read-only, and self-renewal admission",
	},
	{
		Leaf: "internal/regionadmit", Disposition: DispositionMultiCapability,
		Capabilities: []string{"shared_region_admission"},
		Reason:       "native execution-surface region collision, narrowed-tree, hierarchy, exclusivity, read-only, and self-renewal admission",
	},
	{
		Leaf: "internal/vcachegov", Disposition: DispositionMultiCapability,
		Capabilities: []string{"warm_cache_budget_scheduling"},
		Reason:       "native rate-limit-aware warm-cache budget planning and value-ranked candidate scheduling; governor, proof, and secret-policy functions remain separate capability debt",
	},
	{
		Leaf: "internal/cachewitness", Disposition: DispositionMultiCapability,
		Capabilities: []string{"cache_reuse_divergence_detection"},
		Reason:       "native cross-provenance cache-reuse divergence fold is covered; metrics parsing and managed-cache self-check remain separate capability debt",
	},
	{
		Leaf: "internal/cachevalueledger", Disposition: DispositionMultiCapability,
		Capabilities: []string{"cache_reuse_trend_gate"},
		Reason:       "native trailing-window cache-reuse regression gate is covered; ledger I/O and aggregate cache-value scoring remain separate capability debt",
	},
	{
		Leaf: "internal/vcachecal", Disposition: DispositionMultiCapability,
		Capabilities: []string{"cache_budget_concentration_allocation"},
		Reason:       "native concentration-weighted shared cache-budget allocation is covered; warmth estimation, probing, fitting, prediction error, and resume calibration remain separate capability debt",
	},
	{
		Leaf: "internal/vcachewarm", Disposition: DispositionMultiCapability,
		Capabilities: []string{"dedicated_cache_warming"},
		Reason:       "native dedicated cache-warm planning and provider-readback accounting are covered; fanout concurrency and formula helpers remain separate capability debt",
	},
	{
		Leaf: "internal/vcacheobserve", Disposition: DispositionMultiCapability,
		Capabilities: []string{"provider_cache_economics_fold"},
		Reason:       "native provider-cache economics/value fold is covered; provider action planning/application and context joins remain separate capability debt",
	},
	{
		Leaf: "internal/vcachesnapshot", Disposition: DispositionCapability,
		Capabilities: []string{"bounded_cache_snapshot"},
		Reason:       "native bounded fsynced JSONL provider-cache snapshot persistence and tolerant replay",
	},
	{
		Leaf: "internal/vcacheextract", Disposition: DispositionMultiCapability,
		Capabilities: []string{"codex_token_telemetry_sanitization"},
		Reason:       "native Codex JSONL token-counter extraction, allowlist sanitization, and observed-turn conversion are covered; session discovery remains separate capability debt",
	},
	{
		Leaf: "internal/vcacheqa", Disposition: DispositionMultiCapability,
		Capabilities: []string{"cache_honesty_ast_lint"},
		Reason:       "native AST lint for cache-warmth context-elision reasoning is covered; forced-miss, witness-chain, provenance, determinism, and aggregate gate capabilities remain separate debt",
	},
	{
		Leaf: "internal/vcachestar", Disposition: DispositionMultiCapability,
		Capabilities: []string{"provider_cache_telemetry_reconciliation"},
		Reason:       "native provider cache-read reconciliation, false-warm demotion, divergence location, and confirmed-only cost booking are covered; preflight and star planning remain separate capability debt",
	},
	{
		Leaf: "internal/vcachescore", Disposition: DispositionMultiCapability,
		Capabilities: []string{"default_cache_readiness_gate"},
		Reason:       "native default-cache readiness gate over cold-path correctness, versioned usefulness, separated evidence provenance, and unsupported active paths is covered; aggregate scoring, economics, activation, and index planning remain separate debt",
	},
	{
		Leaf: "internal/claimcheck", Disposition: DispositionMultiCapability,
		Capabilities: []string{"net_true_claim_grading"},
		Reason:       "native six-question net-true/strawman/not-yet claim grading is covered; witness planning, fixture fingerprinting, and reuse remain separate capability debt",
	},
	{
		Leaf: "internal/astquery", Disposition: DispositionCapability,
		Capabilities: []string{"go_structural_expression_search"},
		Reason:       "native Go expression AST matching with metavariable back-references, wildcards, bindings, and source positions",
	},
	{
		Leaf: "internal/trigram", Disposition: DispositionMultiCapability,
		Capabilities: []string{"indexed_literal_code_search"},
		Reason:       "native trigram-postings literal search with exact line verification and short-query fallback is covered; regex candidate extraction and lexical similarity remain separate capability debt",
	},
	{
		Leaf: "internal/codelint", Disposition: DispositionMultiCapability,
		Capabilities: []string{"go_syntax_validation"},
		Reason:       "native standalone Go syntax validation with all-errors locations and no semantic false positives is covered; JSON, external language packs, registry routing, summaries, and LSP framing remain separate capability debt",
	},
	{
		Leaf: "internal/computeadmit", Disposition: DispositionMultiCapability,
		Capabilities: []string{"compute_region_admission"},
		Reason:       "native compute-region taxonomy and live-lease collision admission",
	},
	{
		Leaf: "internal/launchlatency", Disposition: DispositionMultiCapability,
		Capabilities: []string{"worker_launch_latency_summary"},
		Reason:       "native dispatch-to-heartbeat histogram, percentile, and negative-clock-skew fold",
	},
	{
		Leaf: "internal/mutationbudget", Disposition: DispositionMultiCapability,
		Capabilities: []string{"github_mutation_budgeting"},
		Reason:       "native GitHub API mutation reserve guard and mixed hourly call estimator",
	},
	{
		Leaf: "internal/deadlineadmit", Disposition: DispositionMultiCapability,
		Capabilities: []string{"deadline_aware_admission"},
		Reason:       "native EDF ordering plus predicted-miss shedding while retaining non-degradable requests",
	},
	{
		Leaf: "internal/timeoutphase", Disposition: DispositionMultiCapability,
		Capabilities: []string{"timeout_phase_attribution"},
		Reason:       "native closed-vocabulary attribution of worker timeouts to startup, edit, test, commit, push, or unknown phases",
	},
	{
		Leaf: "internal/kvbudget", Disposition: DispositionMultiCapability,
		Capabilities: []string{"kv_memory_budget_modeling"},
		Reason:       "native MHA/MLA/DSA KV bytes-per-token, stream-fit, and context-budget closed forms",
	},
	{
		Leaf: "internal/resumebackoff", Disposition: DispositionMultiCapability,
		Capabilities: []string{"failure_signature_resume_backoff"},
		Reason:       "native same-signature exponential resume backoff and cross-session signature parking",
	},
	{
		Leaf: "internal/attemptbudget", Disposition: DispositionMultiCapability,
		Capabilities: []string{"retry_attempt_budgeting"},
		Reason:       "native failure-class-aware attempt ceilings, cooldowns, transient headroom, and typed block routing",
	},
	{
		Leaf: "internal/cacheobs", Disposition: DispositionMultiCapability,
		Capabilities: []string{"cache_observability"},
		Reason:       "native prompt-prefix reuse, cacheability, eligibility, miss-attribution, and reuse-distribution aggregation",
	},
	{
		Leaf: "internal/cacheprice", Disposition: DispositionMultiCapability,
		Capabilities: []string{"cache_cost_accounting"},
		Reason:       "native resident-prefix admission-token and cache-shedding value arithmetic",
	},
	{
		Leaf: "internal/ratelimit", Disposition: DispositionMultiCapability,
		Capabilities: []string{"tool_call_rate_limiting"},
		Reason:       "native per-trace/per-tool/global call and cost limiter plus typed retry-after denial",
	},
	{
		Leaf: "internal/enginecache", Disposition: DispositionMultiCapability,
		Capabilities: []string{"engine_cache_invalidation"},
		Reason:       "native governance-bound invalidation planner and first-class vLLM/SGLang cache-reset adapters",
	},
	{
		Leaf: "internal/vdso", Disposition: DispositionMultiCapability,
		Capabilities: []string{"tool_result_caching"},
		Reason:       "vdso implements pure, content-cached, and static local tool-result fast paths plus sound invalidation; tool-result caching is contracted separately",
	},
	{
		Leaf: "internal/ctxmmu", Disposition: DispositionMultiCapability,
		Capabilities: []string{"context_memory_management"},
		Reason:       "ctxmmu implements result quarantine, context paging, tool-schema residency, and durable-memory write admission; context-memory management is contracted separately",
	},
	{
		Leaf: "internal/adjudicator", Disposition: DispositionMultiCapability,
		Capabilities: []string{"policy_adjudication"},
		Reason:       "adjudicator contains the native capability-floor decision engine plus several specialized security rungs; policy adjudication is contracted separately",
	},
	{
		Leaf: "internal/modelroute", Disposition: DispositionMultiCapability,
		Capabilities: []string{"model_routing"},
		Reason:       "modelroute implements per-aspect selection and ensembles plus manifest validation and live reload; routing is contracted separately",
	},
	{
		Leaf: "internal/gateway", Disposition: DispositionMultiCapability,
		Capabilities: []string{"tool_filtering"},
		Reason:       "gateway hosts several wire/runtime capabilities; tool filtering is one separately contracted capability",
	},
	{
		Leaf: "internal/headroom", Disposition: DispositionMultiCapability,
		Capabilities: []string{"context_compression"},
		Reason:       "headroom contains native structural compression plus external compressor adapters and admission plumbing",
	},
	{
		Leaf: "internal/bench", Disposition: DispositionInfrastructure,
		Reason: "benchmark corpus/report orchestration and claim bookkeeping; not a runtime optimization",
	},
	{
		Leaf: "internal/benchauthority", Disposition: DispositionInfrastructure,
		Reason: "benchmark authority document validation; not a runtime optimization",
	},
	{
		Leaf: "internal/benchcatalog", Disposition: DispositionInfrastructure,
		Reason: "benchmark catalog indexing and validation; not a runtime optimization",
	},
	{
		Leaf: "internal/benchckpt", Disposition: DispositionInfrastructure,
		Reason: "benchmark checkpoint metadata handling; not a runtime optimization",
	},
	{
		Leaf: "internal/benchcli", Disposition: DispositionInfrastructure,
		Reason: "benchmark command parsing/rendering; not a runtime optimization",
	},
	{
		Leaf: "internal/benchids", Disposition: DispositionInfrastructure,
		Reason: "benchmark identifier validation; not a runtime optimization",
	},
	{
		Leaf: "internal/benchlineagegate", Disposition: DispositionInfrastructure,
		Reason: "benchmark provenance gate; not a runtime optimization",
	},
	{
		Leaf: "internal/benchloop", Disposition: DispositionInfrastructure,
		Reason: "benchmark run-loop orchestration; not a runtime optimization",
	},
	{
		Leaf: "internal/benchpost", Disposition: DispositionInfrastructure,
		Reason: "benchmark result publication helper; not a runtime optimization",
	},
	{
		Leaf: "internal/benchruns", Disposition: DispositionInfrastructure,
		Reason: "benchmark run ledger and rendering; not a runtime optimization",
	},
	{
		Leaf: "internal/benchscore", Disposition: DispositionInfrastructure,
		Reason: "benchmark score normalization/reporting; not a runtime optimization",
	},
	{
		Leaf: "internal/nativebench", Disposition: DispositionInfrastructure,
		Reason: "benchmark governance and coverage auditing; it does not implement a runtime optimization",
	},
	{
		Leaf: "internal/radixkv", Disposition: DispositionMultiCapability,
		Capabilities: []string{"prefix_kv_reuse"},
		Reason:       "radixkv implements prefix-indexed KV reuse plus retention and eviction policy; prefix reuse is contracted separately",
	},
	{
		Leaf: "internal/tokenizer", Disposition: DispositionMultiCapability,
		Capabilities: []string{"tokenization"},
		Reason:       "tokenizer implements model-compatible BPE encoding, decoding, pretokenization, and incremental decode; tokenization is contracted separately",
	},
}

var contracts = []Contract{{
	Capability: "prefix_cache_budget_sweep",
	NativePath: "internal/cachesweep/cachesweep.go",
	Workload:   "same timestamped exact-prefix and divergent-child trace, finite token budgets, unbounded ceiling, write-delay overlay, token-cost assumptions, eviction semantics, and independent reuse oracle across every arm",
	Metrics:    []string{"hit_correctness", "reuse_ratio", "reused_tokens", "evictions", "roi_knee_error", "simulation_latency_ms", "throughput_accesses_per_second", "cpu_seconds", "peak_rss_bytes", "storage_bytes", "network_bytes", "total_cost"},
	Alternatives: []Alternative{
		{Name: "no prefix cache", Class: TunedBaseline, Source: "internal/cachesweep/compare.go"},
		{Name: "libCacheSim", Class: NextBest, Source: "https://github.com/1a1a11a/libCacheSim"},
		{Name: "Caffeine simulator", Class: NextBest, Source: "https://github.com/ben-manes/caffeine/wiki/Simulator"},
		{Name: "Redis or Valkey maxmemory policies", Class: NextBest, Source: "https://redis.io/docs/latest/develop/reference/eviction/"},
	},
	Witness: "../../docs/benchmarks/PREFIX-CACHE-SWEEP-ALTERNATIVES-2026-08-10.md",
},
	{
		Capability: "lane_tree_collision_admission",
		NativePath: "internal/laneadmit/laneadmit.go",
		Workload:   "same same-lane-disjoint, cross-lane-overlap, exclusive-lane, read-only, and self-renewal requests, live leases, taxonomy trees, and independent admission oracle across every arm",
		Metrics:    []string{"admission_precision", "admission_recall", "false_denies", "false_allows", "decision_latency_ms", "acquisition_latency_ms", "throughput_decisions_per_second", "cpu_seconds", "peak_rss_bytes", "network_bytes", "storage_bytes", "total_cost"},
		Alternatives: []Alternative{
			{Name: "geometry-only tree overlap", Class: TunedBaseline, Source: "internal/laneadmit/compare.go"},
			{Name: "DOS arbitrate", Class: FirstClassIntegration, Integration: "dos", Source: "dos arbitrate"},
			{Name: "GitHub Actions concurrency groups", Class: NextBest, Source: "https://docs.github.com/actions/using-jobs/using-concurrency"},
			{Name: "Kubernetes Lease coordination", Class: NextBest, Source: "https://kubernetes.io/docs/concepts/architecture/leases/"},
			{Name: "etcd concurrency mutex", Class: NextBest, Source: "https://pkg.go.dev/go.etcd.io/etcd/client/v3/concurrency"},
		},
		Witness:      "../../docs/benchmarks/LANE-TREE-ADMISSION-ALTERNATIVES-2026-08-10.md",
		Integrations: []string{"dos"},
	},
	{
		Capability: "shared_region_admission",
		NativePath: "internal/regionadmit/regionadmit.go",
		Workload:   "same narrowed-disjoint, cross-lane-overlap, exclusive-lane, read-only, self-renewal, and hierarchical sub-lane requests, live leases, taxonomy, and independent admission oracle across every arm",
		Metrics:    []string{"admission_precision", "admission_recall", "false_denies", "false_allows", "decision_latency_ms", "acquisition_latency_ms", "throughput_decisions_per_second", "cpu_seconds", "peak_rss_bytes", "network_bytes", "storage_bytes", "total_cost"},
		Alternatives: []Alternative{
			{Name: "geometry-only region overlap", Class: TunedBaseline, Source: "internal/regionadmit/compare.go"},
			{Name: "fak + DOS arbitrate", Class: FirstClassIntegration, Integration: "dos", Source: "dos arbitrate"},
			{Name: "fak + Git-ref leases", Class: FirstClassIntegration, Integration: "leaseref", Source: "internal/leaseref"},
			{Name: "Kubernetes Lease coordination", Class: NextBest, Source: "https://kubernetes.io/docs/concepts/architecture/leases/"},
			{Name: "etcd concurrency mutex", Class: NextBest, Source: "https://pkg.go.dev/go.etcd.io/etcd/client/v3/concurrency"},
			{Name: "GitHub Actions concurrency groups", Class: NextBest, Source: "https://docs.github.com/actions/using-jobs/using-concurrency"},
		},
		Witness:      "../../docs/benchmarks/SHARED-REGION-ADMISSION-ALTERNATIVES-2026-08-10.md",
		Integrations: []string{"dos", "leaseref"},
	},
	{
		Capability: "compute_region_admission",
		NativePath: "internal/computeadmit/computeadmit.go",
		Workload:   "same overlapping, disjoint, out-of-taxonomy, and different-class compute claims, live lease, class address space, exclusivity, and independent admission oracle across every arm",
		Metrics:    []string{"admission_precision", "admission_recall", "constraint_violations", "scheduling_latency_ms", "throughput_decisions_per_second", "cpu_seconds", "peak_rss_bytes", "control_plane_bytes", "accelerator_idle_seconds", "total_cost"},
		Alternatives: []Alternative{
			{Name: "dispatch without region admission", Class: TunedBaseline, Source: "internal/computeadmit/compare.go"},
			{Name: "Kubernetes scheduler", Class: NextBest, Source: "https://kubernetes.io/docs/concepts/scheduling-eviction/kube-scheduler/"},
			{Name: "Slurm scheduler", Class: NextBest, Source: "https://slurm.schedmd.com/"},
			{Name: "Ray scheduler", Class: NextBest, Source: "https://docs.ray.io/en/latest/ray-core/scheduling/"},
			{Name: "AWS Batch", Class: NextBest, Source: "https://docs.aws.amazon.com/batch/"},
		},
		Witness: "../../docs/benchmarks/COMPUTE-REGION-ADMISSION-ALTERNATIVES-2026-08-10.md",
	},
	{
		Capability: "warm_cache_budget_scheduling",
		NativePath: "internal/vcachegov/warmbudget.go",
		Workload:   "same provider rate-limit snapshot, TTL, anchor size, six mixed-value/secret candidates, request trace, and independent warm-set quality oracle across every arm",
		Metrics:    []string{"useful_warm_hits", "value_captured", "wasted_writes", "quota_violations", "planning_latency_ms", "request_latency_ms", "throughput_requests_per_second", "input_tokens", "cache_write_tokens", "cache_read_tokens", "cpu_seconds", "peak_rss_bytes", "network_bytes", "storage_bytes", "total_cost"},
		Alternatives: []Alternative{
			{Name: "demand-only fills without proactive warming", Class: TunedBaseline, Source: "internal/vcachegov/compare.go"},
			{Name: "fak + LMCache", Class: FirstClassIntegration, Integration: "lmcache", Source: "internal/cachemeta/lmcache_transfer.go"},
			{Name: "fak + Mooncake", Class: FirstClassIntegration, Integration: "mooncake", Source: "internal/cachemeta/mooncake_transfer.go"},
			{Name: "fak + NIXL", Class: FirstClassIntegration, Integration: "nixl", Source: "internal/cachemeta/nixl_lease.go"},
			{Name: "vLLM automatic prefix caching", Class: NextBest, Source: "https://docs.vllm.ai/en/latest/design/prefix_caching/"},
			{Name: "SGLang HiCache and cache-aware scheduling", Class: NextBest, Source: "https://docs.sglang.ai/advanced_features/hicache.html"},
		},
		Witness:      "../../docs/benchmarks/WARM-CACHE-BUDGET-SCHEDULING-ALTERNATIVES-2026-08-10.md",
		Integrations: []string{"lmcache", "mooncake", "nixl"},
	},
	{
		Capability: "cache_reuse_divergence_detection",
		NativePath: "internal/cachewitness/divergence.go",
		Workload:   "same three stable, divergent, and single-provenance cache records, tolerance, and independent alert oracle across every arm",
		Metrics:    []string{"true_alerts", "false_alerts", "missed_divergences", "alert_latency_ms", "ingest_latency_ms", "throughput_records_per_second", "cpu_seconds", "peak_rss_bytes", "network_bytes", "storage_bytes", "total_cost"},
		Alternatives: []Alternative{
			{Name: "raw reuse counters without divergence detection", Class: TunedBaseline, Source: "internal/cachewitness/compare.go"},
			{Name: "fak + Prometheus", Class: FirstClassIntegration, Integration: "prometheus", Source: "gateway /metrics"},
			{Name: "fak + OpenTelemetry", Class: FirstClassIntegration, Integration: "opentelemetry", Source: "internal/otel"},
			{Name: "Prometheus recording and alerting rules", Class: NextBest, Source: "https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/"},
			{Name: "Datadog anomaly monitor", Class: NextBest, Source: "https://docs.datadoghq.com/monitors/types/anomaly/"},
		},
		Witness:      "../../docs/benchmarks/CACHE-REUSE-DIVERGENCE-ALTERNATIVES-2026-08-10.md",
		Integrations: []string{"prometheus", "opentelemetry"},
	},
	{
		Capability: "cache_reuse_trend_gate",
		NativePath: "internal/cachevalueledger/ledger.go",
		Workload:   "same five chronological stable/degraded/single-turn cache ledger rows, minimum-window threshold, tolerance, and independent regression oracle across every arm",
		Metrics:    []string{"true_alerts", "false_alerts", "missed_regressions", "alert_latency_ms", "query_latency_ms", "throughput_rows_per_second", "prompt_tokens_represented", "cpu_seconds", "peak_rss_bytes", "network_bytes", "storage_bytes", "total_cost"},
		Alternatives: []Alternative{
			{Name: "raw JSONL ledger without trend gate", Class: TunedBaseline, Source: "internal/cachevalueledger/compare.go"},
			{Name: "fak + Prometheus", Class: FirstClassIntegration, Integration: "prometheus", Source: "gateway /metrics"},
			{Name: "fak + OpenTelemetry", Class: FirstClassIntegration, Integration: "opentelemetry", Source: "internal/otel"},
			{Name: "Prometheus recording and alerting rules", Class: NextBest, Source: "https://prometheus.io/docs/prometheus/latest/configuration/recording_rules/"},
			{Name: "Datadog change and anomaly monitor", Class: NextBest, Source: "https://docs.datadoghq.com/monitors/types/change-alert/"},
		},
		Witness:      "../../docs/benchmarks/CACHE-REUSE-TREND-GATE-ALTERNATIVES-2026-08-10.md",
		Integrations: []string{"prometheus", "opentelemetry"},
	},
	{
		Capability: "cache_budget_concentration_allocation",
		NativePath: "internal/vcachecal/allocation.go",
		Workload:   "same 1200-byte budget, top-K one, concentrated/flat/unmeasured buckets, demand weights, deterministic conservation, and independent allocation oracle across every arm",
		Metrics:    []string{"allocation_quality", "captured_value", "budget_conservation", "starved_buckets", "allocation_latency_ms", "throughput_allocations_per_second", "cache_bytes", "cpu_seconds", "peak_rss_bytes", "network_bytes", "storage_bytes", "total_cost"},
		Alternatives: []Alternative{
			{Name: "equal-share cache allocation", Class: TunedBaseline, Source: "internal/vcachecal/compare.go"},
			{Name: "request-volume proportional allocation", Class: TunedBaseline, Source: "internal/vcachecal/compare.go"},
			{Name: "fak + LMCache", Class: FirstClassIntegration, Integration: "lmcache", Source: "internal/cachemeta/lmcache_transfer.go"},
			{Name: "fak + Mooncake", Class: FirstClassIntegration, Integration: "mooncake", Source: "internal/cachemeta/mooncake_transfer.go"},
			{Name: "vLLM cache-aware routing", Class: NextBest, Source: "https://docs.vllm.ai/en/latest/serving/parallelism_scaling/"},
			{Name: "SGLang HiCache and cache-aware scheduling", Class: NextBest, Source: "https://docs.sglang.ai/advanced_features/hicache.html"},
		},
		Witness:      "../../docs/benchmarks/CACHE-BUDGET-CONCENTRATION-ALLOCATION-ALTERNATIVES-2026-08-10.md",
		Integrations: []string{"lmcache", "mooncake"},
	},
	{
		Capability: "dedicated_cache_warming",
		NativePath: "internal/vcachewarm/warm.go",
		Workload:   "same five profitable/refused/automatic warm cases, prefix fingerprints, capability and tool guards, two-read fanout, and independent decision/readback oracle across every arm",
		Metrics:    []string{"decision_accuracy", "useful_warms", "wasted_warms", "cache_read_tokens", "cache_write_tokens", "planning_latency_ms", "ttft_ms", "throughput_requests_per_second", "cpu_seconds", "peak_rss_bytes", "network_bytes", "storage_bytes", "total_cost"},
		Alternatives: []Alternative{
			{Name: "demand-only fills without dedicated warming", Class: TunedBaseline, Source: "internal/vcachewarm/compare.go"},
			{Name: "fak + Anthropic prompt caching", Class: FirstClassIntegration, Integration: "anthropic", Source: "gateway Anthropic adapter"},
			{Name: "fak + Gemini CachedContent", Class: FirstClassIntegration, Integration: "gemini", Source: "internal/gemini"},
			{Name: "fak + OpenAI automatic prefix caching", Class: FirstClassIntegration, Integration: "openai", Source: "gateway OpenAI adapter"},
			{Name: "fak + LMCache", Class: FirstClassIntegration, Integration: "lmcache", Source: "internal/cachemeta/lmcache_transfer.go"},
			{Name: "fak + Mooncake", Class: FirstClassIntegration, Integration: "mooncake", Source: "internal/cachemeta/mooncake_transfer.go"},
			{Name: "vLLM automatic prefix caching", Class: NextBest, Source: "https://docs.vllm.ai/en/latest/design/prefix_caching/"},
			{Name: "SGLang HiCache", Class: NextBest, Source: "https://docs.sglang.ai/advanced_features/hicache.html"},
		},
		Witness:      "../../docs/benchmarks/DEDICATED-CACHE-WARMING-ALTERNATIVES-2026-08-10.md",
		Integrations: []string{"anthropic", "gemini", "openai", "lmcache", "mooncake"},
	},
	{
		Capability: "provider_cache_economics_fold",
		NativePath: "internal/vcacheobserve/vcacheobserve.go",
		Workload:   "same four cold/write/read/context provider-cache turns, two families, multipliers, and independent counter/value oracle across every arm",
		Metrics:    []string{"counter_accuracy", "saved_token_equiv", "hit_rate", "fold_latency_ms", "query_latency_ms", "input_tokens", "cache_write_tokens", "cache_read_tokens", "context_shed_tokens", "cpu_seconds", "peak_rss_bytes", "network_bytes", "storage_bytes", "total_cost"},
		Alternatives: []Alternative{
			{Name: "raw provider usage without economics fold", Class: TunedBaseline, Source: "internal/vcacheobserve/compare.go"},
			{Name: "fak + Prometheus", Class: FirstClassIntegration, Integration: "prometheus", Source: "gateway /metrics"},
			{Name: "fak + OpenTelemetry", Class: FirstClassIntegration, Integration: "opentelemetry", Source: "internal/otel"},
			{Name: "Anthropic usage and cost reporting", Class: NextBest, Source: "https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching"},
			{Name: "OpenAI usage and cost reporting", Class: NextBest, Source: "https://platform.openai.com/docs/guides/prompt-caching"},
			{Name: "Datadog LLM Observability", Class: NextBest, Source: "https://docs.datadoghq.com/llm_observability/"},
			{Name: "LangSmith", Class: NextBest, Source: "https://docs.smith.langchain.com/observability/how_to_guides/monitor_costs"},
		},
		Witness:      "../../docs/benchmarks/PROVIDER-CACHE-ECONOMICS-ALTERNATIVES-2026-08-10.md",
		Integrations: []string{"prometheus", "opentelemetry"},
	},
	{
		Capability: "bounded_cache_snapshot",
		NativePath: "internal/vcachesnapshot/vcachesnapshot.go",
		Workload:   "same five ordered provider-cache turns, three-row retention window, replay-fidelity oracle, fsync requirement, and represented-token count across every arm",
		Metrics:    []string{"retained_rows", "dropped_rows", "corrupt_rows", "lost_rows", "write_latency_ms", "read_latency_ms", "throughput_rows_per_second", "input_tokens_represented", "cpu_seconds", "peak_rss_bytes", "network_bytes", "storage_bytes", "total_cost"},
		Alternatives: []Alternative{
			{Name: "unbounded append-only JSONL", Class: TunedBaseline, Source: "internal/vcachesnapshot/compare.go"},
			{Name: "fak + Prometheus", Class: FirstClassIntegration, Integration: "prometheus", Source: "gateway /metrics"},
			{Name: "fak + OpenTelemetry", Class: FirstClassIntegration, Integration: "opentelemetry", Source: "internal/otel"},
			{Name: "SQLite WAL", Class: NextBest, Source: "https://sqlite.org/wal.html"},
			{Name: "Prometheus TSDB", Class: NextBest, Source: "https://prometheus.io/docs/prometheus/latest/storage/"},
			{Name: "ClickHouse", Class: NextBest, Source: "https://clickhouse.com/docs/engines/table-engines/mergetree-family/mergetree"},
		},
		Witness:      "../../docs/benchmarks/BOUNDED-CACHE-SNAPSHOT-ALTERNATIVES-2026-08-10.md",
		Integrations: []string{"prometheus", "opentelemetry"},
	},
	{
		Capability: "codex_token_telemetry_sanitization",
		NativePath: "internal/vcacheextract/codex.go",
		Workload:   "same four ordered mixed Codex JSONL records, exact two-row input/cached-token oracle, and forbidden prompt/tool/response field and byte-leak oracle across every arm",
		Metrics:    []string{"eligible_rows", "missed_rows", "extra_rows", "counter_mismatches", "forbidden_fields", "forbidden_bytes", "parse_failures", "malformed_tail_rows", "latency_ms", "throughput_bytes_per_second", "cpu_seconds", "peak_rss_bytes", "input_bytes", "output_bytes", "network_bytes", "operator_seconds", "total_cost"},
		Alternatives: []Alternative{
			{Name: "raw JSONL pass-through", Class: TunedBaseline, Source: "internal/vcacheextract/compare.go"},
			{Name: "fak + OpenTelemetry", Class: FirstClassIntegration, Integration: "opentelemetry", Source: "internal/otel"},
			{Name: "fak + Prometheus", Class: FirstClassIntegration, Integration: "prometheus", Source: "gateway /metrics"},
			{Name: "jq streaming projection", Class: NextBest, Source: "https://jqlang.org/manual/"},
			{Name: "Vector VRL remap", Class: NextBest, Source: "https://vector.dev/docs/reference/vrl/"},
			{Name: "Fluent Bit filter pipeline", Class: NextBest, Source: "https://docs.fluentbit.io/manual/pipeline/filters"},
		},
		Witness:      "../../docs/benchmarks/CODEX-TOKEN-SANITIZATION-ALTERNATIVES-2026-08-10.md",
		Integrations: []string{"opentelemetry", "prometheus"},
	},
	{
		Capability: "cache_honesty_ast_lint",
		NativePath: "internal/vcacheqa/vcacheqa.go",
		Workload:   "same fixed Go package with two true cache-context elision findings, clean live code, excluded test content, and non-Go decoys; exact ordered file/line oracle across every arm",
		Metrics:    []string{"true_positives", "false_positives", "false_negatives", "location_errors", "parse_failures", "latency_ms", "throughput_bytes_per_second", "cpu_seconds", "peak_rss_bytes", "input_bytes", "operator_seconds", "total_cost"},
		Alternatives: []Alternative{
			{Name: "tuned non-test text scan", Class: TunedBaseline, Source: "internal/vcacheqa/compare.go"},
			{Name: "go/analysis analyzer", Class: NextBest, Source: "https://pkg.go.dev/golang.org/x/tools/go/analysis"},
			{Name: "Semgrep", Class: NextBest, Source: "https://semgrep.dev/docs/"},
			{Name: "CodeQL", Class: NextBest, Source: "https://codeql.github.com/docs/"},
			{Name: "golangci-lint custom analyzer", Class: NextBest, Source: "https://golangci-lint.run/docs/plugins/go-plugins/"},
		},
		Witness: "../../docs/benchmarks/CACHE-HONESTY-LINT-ALTERNATIVES-2026-08-10.md",
	},
	{
		Capability: "provider_cache_telemetry_reconciliation",
		NativePath: "internal/vcachestar/star.go",
		Workload:   "same believed-warm old/new prefix segments and bytes, zero provider cache-read telemetry, uncached-token count, exact divergence oracle, demotion requirement, and confirmed-only cost booking across every arm",
		Metrics:    []string{"demotion_accuracy", "alarm_accuracy", "divergence_segment_error", "divergence_token_error", "divergence_byte_error", "booked_uncached_tokens", "rebate_tokens", "latency_ns", "cpu_seconds", "peak_rss_bytes", "telemetry_bytes", "network_bytes", "storage_bytes", "operator_seconds", "total_cost"},
		Alternatives: []Alternative{
			{Name: "trust warm manifest and modeled savings", Class: TunedBaseline, Source: "internal/vcachestar/compare.go"},
			{Name: "fak + Anthropic prompt caching", Class: FirstClassIntegration, Integration: "anthropic", Source: "internal/provider"},
			{Name: "fak + OpenAI prompt caching", Class: FirstClassIntegration, Integration: "openai", Source: "internal/provider"},
			{Name: "fak + Gemini context caching", Class: FirstClassIntegration, Integration: "gemini", Source: "internal/provider"},
			{Name: "fak + Prometheus", Class: FirstClassIntegration, Integration: "prometheus", Source: "gateway /metrics"},
			{Name: "fak + OpenTelemetry", Class: FirstClassIntegration, Integration: "opentelemetry", Source: "internal/otel"},
			{Name: "Prometheus recording and alert rules", Class: NextBest, Source: "https://prometheus.io/docs/prometheus/latest/configuration/recording_rules/"},
			{Name: "Datadog monitors", Class: NextBest, Source: "https://docs.datadoghq.com/monitors/"},
			{Name: "LangSmith traces", Class: NextBest, Source: "https://docs.smith.langchain.com/observability"},
		},
		Witness:      "../../docs/benchmarks/CACHE-TELEMETRY-RECONCILIATION-ALTERNATIVES-2026-08-10.md",
		Integrations: []string{"anthropic", "openai", "gemini", "prometheus", "opentelemetry"},
	},
	{
		Capability: "default_cache_readiness_gate",
		NativePath: "internal/vcachescore/readiness.go",
		Workload:   "same five ordered readiness reports covering fully witnessed readiness, provider-only evidence, wrong plane provenance, unsupported active-cache capability, and cold-path failure with exact verdict and reason-class oracle",
		Metrics:    []string{"true_ready", "true_blocked", "false_ready", "false_blocked", "reason_mismatches", "latency_ns", "throughput_cases_per_second", "cpu_seconds", "peak_rss_bytes", "input_bytes", "network_bytes", "storage_bytes", "operator_seconds", "total_cost"},
		Alternatives: []Alternative{
			{Name: "usefulness-score threshold only", Class: TunedBaseline, Source: "internal/vcachescore/compare.go"},
			{Name: "fak + Prometheus", Class: FirstClassIntegration, Integration: "prometheus", Source: "gateway /metrics"},
			{Name: "fak + OpenTelemetry", Class: FirstClassIntegration, Integration: "opentelemetry", Source: "internal/otel"},
			{Name: "OPA/Rego", Class: NextBest, Source: "https://www.openpolicyagent.org/docs/latest/policy-language/"},
			{Name: "Prometheus rules", Class: NextBest, Source: "https://prometheus.io/docs/prometheus/latest/configuration/recording_rules/"},
			{Name: "Datadog monitors", Class: NextBest, Source: "https://docs.datadoghq.com/monitors/"},
			{Name: "LangSmith evaluations", Class: NextBest, Source: "https://docs.smith.langchain.com/evaluation"},
		},
		Witness:      "../../docs/benchmarks/DEFAULT-CACHE-READINESS-ALTERNATIVES-2026-08-10.md",
		Integrations: []string{"prometheus", "opentelemetry"},
	},
	{
		Capability: "net_true_claim_grading",
		NativePath: "internal/claimcheck/claimcheck.go",
		Workload:   "same nine labeled claims spanning honest net-true, gated realized, strawman baseline, missing baseline/witness/provenance/net/scope, and unrealized seam with exact verdict and failing-question oracle",
		Metrics:    []string{"exact_verdicts", "wrong_net_true", "wrong_strawman", "wrong_not_yet", "reason_mismatches", "latency_ns", "throughput_claims_per_second", "cpu_seconds", "peak_rss_bytes", "input_bytes", "model_tokens", "network_bytes", "storage_bytes", "operator_seconds", "total_cost"},
		Alternatives: []Alternative{
			{Name: "accept claim when any witness exists", Class: TunedBaseline, Source: "internal/claimcheck/compare.go"},
			{Name: "fak + Prometheus", Class: FirstClassIntegration, Integration: "prometheus", Source: "gateway /metrics"},
			{Name: "fak + OpenTelemetry", Class: FirstClassIntegration, Integration: "opentelemetry", Source: "internal/otel"},
			{Name: "OPA/Rego", Class: NextBest, Source: "https://www.openpolicyagent.org/docs/latest/policy-language/"},
			{Name: "OpenAI Evals graders", Class: NextBest, Source: "https://platform.openai.com/docs/guides/evals"},
			{Name: "LangSmith evaluators", Class: NextBest, Source: "https://docs.smith.langchain.com/evaluation"},
			{Name: "Braintrust scorers", Class: NextBest, Source: "https://www.braintrust.dev/docs/guides/scores"},
			{Name: "DeepEval metrics", Class: NextBest, Source: "https://deepeval.com/docs/metrics-introduction"},
		},
		Witness:      "../../docs/benchmarks/NET-TRUE-CLAIM-GRADING-ALTERNATIVES-2026-08-10.md",
		Integrations: []string{"prometheus", "opentelemetry"},
	},
	{
		Capability: "go_structural_expression_search",
		NativePath: "internal/astquery/astquery.go",
		Workload:   "same Go source and repeated-metavariable call pattern with two true calls, inconsistent-argument call, comment/string decoys, and unrelated call; exact ordered location and binding oracle",
		Metrics:    []string{"true_positives", "false_positives", "false_negatives", "binding_errors", "location_errors", "parse_failures", "latency_ns", "throughput_bytes_per_second", "cpu_seconds", "peak_rss_bytes", "input_bytes", "operator_seconds", "total_cost"},
		Alternatives: []Alternative{
			{Name: "literal text search", Class: TunedBaseline, Source: "internal/astquery/compare.go"},
			{Name: "Semgrep", Class: NextBest, Source: "https://semgrep.dev/docs/"},
			{Name: "ast-grep", Class: NextBest, Source: "https://ast-grep.github.io/"},
			{Name: "Comby", Class: NextBest, Source: "https://comby.dev/docs/"},
			{Name: "gogrep", Class: NextBest, Source: "https://github.com/mvdan/gogrep"},
		},
		Witness: "../../docs/benchmarks/GO-AST-QUERY-ALTERNATIVES-2026-08-10.md",
	},
	{
		Capability: "indexed_literal_code_search",
		NativePath: "internal/trigram/trigram.go",
		Workload:   "same six-file corpus and four literal queries spanning multiple matches, no match, short-pattern fallback, and Unicode with exact ordered path/line oracle",
		Metrics:    []string{"exact_queries", "false_positives", "false_negatives", "location_errors", "build_latency_ms", "query_latency_ms", "throughput_bytes_per_second", "cpu_seconds", "peak_rss_bytes", "corpus_bytes", "index_bytes", "network_bytes", "storage_bytes", "operator_seconds", "total_cost"},
		Alternatives: []Alternative{
			{Name: "optimized in-memory linear scan", Class: TunedBaseline, Source: "internal/trigram/compare.go"},
			{Name: "ripgrep", Class: NextBest, Source: "https://github.com/BurntSushi/ripgrep"},
			{Name: "git grep", Class: NextBest, Source: "https://git-scm.com/docs/git-grep"},
			{Name: "Zoekt", Class: NextBest, Source: "https://github.com/sourcegraph/zoekt"},
			{Name: "livegrep", Class: NextBest, Source: "https://github.com/livegrep/livegrep"},
			{Name: "Sourcegraph Search", Class: NextBest, Source: "https://sourcegraph.com/docs/code-search"},
		},
		Witness: "../../docs/benchmarks/INDEXED-CODE-SEARCH-ALTERNATIVES-2026-08-10.md",
	},
	{
		Capability: "go_syntax_validation",
		NativePath: "internal/codelint/packs.go",
		Workload:   "same four standalone Go files covering valid source, single syntax error, multiple recoverable syntax errors, and syntax-valid unresolved identifier with exact syntax-only classification and location oracle",
		Metrics:    []string{"correct_files", "false_syntax_errors", "missed_syntax_errors", "reported_errors", "location_errors", "latency_ms", "throughput_bytes_per_second", "cpu_seconds", "peak_rss_bytes", "input_bytes", "operator_seconds", "total_cost"},
		Alternatives: []Alternative{
			{Name: "go/parser first-error-only", Class: TunedBaseline, Source: "internal/codelint/compare.go"},
			{Name: "go test compile", Class: NextBest, Source: "https://pkg.go.dev/cmd/go#hdr-Test_packages"},
			{Name: "gofmt", Class: NextBest, Source: "https://pkg.go.dev/cmd/gofmt"},
			{Name: "go vet", Class: NextBest, Source: "https://pkg.go.dev/cmd/vet"},
			{Name: "staticcheck", Class: NextBest, Source: "https://staticcheck.dev/docs/"},
			{Name: "golangci-lint", Class: NextBest, Source: "https://golangci-lint.run/"},
			{Name: "gopls diagnostics", Class: NextBest, Source: "https://go.dev/gopls/"},
		},
		Witness: "../../docs/benchmarks/GO-SYNTAX-VALIDATION-ALTERNATIVES-2026-08-10.md",
	},
	{
		Capability: "worker_launch_latency_summary",
		NativePath: "internal/launchlatency/launchlatency.go",
		Workload:   "same six dispatch-to-heartbeat observations, bucket edges, percentile convention, negative-clock-skew sample, and independent summary oracle across every arm",
		Metrics:    []string{"bucket_accuracy", "quantile_error", "dropped_observations", "ingestion_latency_ms", "query_latency_ms", "cpu_seconds", "peak_rss_bytes", "network_bytes", "storage_bytes", "total_cost"},
		Alternatives: []Alternative{
			{Name: "raw launch events without summary", Class: TunedBaseline, Source: "internal/launchlatency/compare.go"},
			{Name: "Prometheus histogram", Class: NextBest, Source: "https://prometheus.io/docs/practices/histograms/"},
			{Name: "OpenTelemetry metrics", Class: NextBest, Source: "https://opentelemetry.io/docs/concepts/signals/metrics/"},
			{Name: "Datadog distribution metric", Class: NextBest, Source: "https://docs.datadoghq.com/metrics/distributions/"},
		},
		Witness: "../../docs/benchmarks/LAUNCH-LATENCY-ALTERNATIVES-2026-08-10.md",
	},
	{
		Capability: "github_mutation_budgeting",
		NativePath: "internal/mutationbudget/mutationbudget.go",
		Workload:   "same eight-close, five-comment, two-fetch plan, observed twelve-call remainder, five-call reserve, reset time, and independent hold oracle across every arm",
		Metrics:    []string{"hold_correctness", "calls_attempted", "calls_avoided", "decision_latency_ms", "cpu_seconds", "peak_rss_bytes", "network_bytes", "total_cost"},
		Alternatives: []Alternative{
			{Name: "direct API calls without reserve", Class: TunedBaseline, Source: "internal/mutationbudget/compare.go"},
			{Name: "GitHub Octokit rate-limit handling", Class: NextBest, Source: "https://github.com/octokit"},
			{Name: "gh api rate-limit handling", Class: NextBest, Source: "https://cli.github.com/manual/gh_api"},
			{Name: "Envoy global rate limit", Class: NextBest, Source: "https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/rate_limit_filter"},
		},
		Witness: "../../docs/benchmarks/MUTATION-BUDGET-ALTERNATIVES-2026-08-10.md",
	},
	{
		Capability: "deadline_aware_admission",
		NativePath: "internal/deadlineadmit/deadlineadmit.go",
		Workload:   "same four-request queue with tied deadlines, one degradable predicted miss, one non-degradable miss, fixed now and threshold, and independent admission oracle across every arm",
		Metrics:    []string{"admission_precision", "admission_recall", "deadline_miss_rate", "queue_latency_ms", "throughput_requests_per_second", "cpu_seconds", "peak_rss_bytes", "accelerator_seconds", "total_cost"},
		Alternatives: []Alternative{
			{Name: "FIFO without predicted-miss shedding", Class: TunedBaseline, Source: "internal/deadlineadmit/compare.go"},
			{Name: "Mooncake deadline-aware admission", Class: NextBest, Source: "https://github.com/kvcache-ai/Mooncake"},
			{Name: "vLLM priority scheduling", Class: NextBest, Source: "https://docs.vllm.ai/"},
			{Name: "SGLang priority scheduling", Class: NextBest, Source: "https://docs.sglang.ai/"},
			{Name: "fak + vLLM priority scheduling", Class: FirstClassIntegration, Integration: "vllm", Source: "internal/engine/vllm.go"},
			{Name: "fak + SGLang priority scheduling", Class: FirstClassIntegration, Integration: "sglang", Source: "internal/engine/sglang.go"},
		},
		Witness:      "../../docs/benchmarks/DEADLINE-ADMISSION-ALTERNATIVES-2026-08-10.md",
		Integrations: []string{"vllm", "sglang"},
	},
	{
		Capability: "timeout_phase_attribution",
		NativePath: "internal/timeoutphase/timeoutphase.go",
		Workload:   "same timeout-at-stage trace, instrumentation points, process lifecycle, sampling policy, and independent phase oracle across every arm",
		Metrics:    []string{"phase_precision", "phase_recall", "dropped_traces", "ingestion_latency_ms", "query_latency_ms", "cpu_seconds", "peak_rss_bytes", "network_bytes", "storage_bytes", "total_cost"},
		Alternatives: []Alternative{
			{Name: "one undifferentiated timeout bucket", Class: TunedBaseline, Source: "internal/timeoutphase/compare.go"},
			{Name: "OpenTelemetry spans", Class: NextBest, Source: "https://opentelemetry.io/docs/concepts/signals/traces/"},
			{Name: "Datadog APM", Class: NextBest, Source: "https://docs.datadoghq.com/tracing/"},
			{Name: "AWS X-Ray", Class: NextBest, Source: "https://docs.aws.amazon.com/xray/latest/devguide/aws-xray.html"},
		},
	},
	{
		Capability: "kv_memory_budget_modeling",
		NativePath: "internal/kvbudget/kvbudget.go",
		Workload:   "same model shape, precision, context lengths, batch and concurrency, serving lifecycle, and independent GPU allocation oracle across every arm",
		Metrics:    []string{"kv_bytes_per_token_error", "peak_allocation_error", "fit_concurrency_equivalence", "latency_ms", "throughput_tokens_per_second", "gpu_memory_bytes", "host_memory_bytes", "total_cost"},
		Alternatives: []Alternative{
			{Name: "full-MHA closed form", Class: TunedBaseline, Source: "internal/kvbudget/compare.go"},
			{Name: "vLLM memory profiler", Class: NextBest, Source: "https://docs.vllm.ai/"},
			{Name: "SGLang memory pool", Class: NextBest, Source: "https://docs.sglang.ai/"},
			{Name: "NVIDIA GenAI-Perf", Class: NextBest, Source: "https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html"},
		},
	},
	{
		Capability: "failure_signature_resume_backoff",
		NativePath: "internal/resumebackoff/resumebackoff.go",
		Workload:   "same repeated-failure restart trace, delay and ceiling policy, process lifecycle, concurrency, and independent schedule oracle across every arm",
		Metrics:    []string{"schedule_equivalence", "restart_storm_prevention", "recovery_time_ms", "restart_count", "cpu_seconds", "peak_rss_bytes", "network_bytes", "total_cost"},
		Alternatives: []Alternative{
			{Name: "immediate resume", Class: TunedBaseline, Source: "internal/resumebackoff/compare.go"},
			{Name: "Kubernetes CrashLoopBackOff", Class: NextBest, Source: "https://kubernetes.io/docs/concepts/workloads/pods/pod-lifecycle/"},
			{Name: "systemd RestartSec", Class: NextBest, Source: "https://www.freedesktop.org/software/systemd/man/latest/systemd.service.html"},
			{Name: "AWS Step Functions retry", Class: NextBest, Source: "https://docs.aws.amazon.com/step-functions/latest/dg/concepts-error-handling.html"},
		},
	},
	{
		Capability: "retry_attempt_budgeting",
		NativePath: "internal/attemptbudget/attemptbudget.go",
		Workload:   "same upstream failure trace, retryable-status classes, attempt and backoff policy, process lifetime, concurrency, and independent retry/stop oracle across every arm",
		Metrics:    []string{"retry_stop_equivalence", "successful_recovery", "request_amplification", "latency_ms", "upstream_requests", "cpu_seconds", "peak_rss_bytes", "network_bytes", "total_cost"},
		Alternatives: []Alternative{
			{Name: "unlimited retries", Class: TunedBaseline, Source: "internal/attemptbudget/compare.go"},
			{Name: "Envoy retry budget", Class: NextBest, Source: "https://www.envoyproxy.io/docs/envoy/latest/intro/arch_overview/http/http_connection_management#arch-overview-http-routing-retry"},
			{Name: "gRPC retry policy", Class: NextBest, Source: "https://grpc.io/docs/guides/retry/"},
			{Name: "AWS SDK adaptive retry", Class: NextBest, Source: "https://docs.aws.amazon.com/sdkref/latest/guide/feature-retry-behavior.html"},
		},
	},
	{
		Capability: "cache_observability",
		NativePath: "internal/cacheobs/cacheobs.go",
		Workload:   "same cache observation trace, label/cardinality policy, process lifetime, aggregation interval, and independent counter oracle across every arm",
		Metrics:    []string{"counter_ratio_equivalence", "dropped_events", "cardinality", "ingestion_latency_ms", "query_latency_ms", "cpu_seconds", "peak_rss_bytes", "network_bytes", "storage_bytes", "total_cost"},
		Alternatives: []Alternative{
			{Name: "no telemetry", Class: TunedBaseline, Source: "internal/cacheobs/compare.go"},
			{Name: "Prometheus client", Class: NextBest, Source: "https://prometheus.io/docs/instrumenting/clientlibs/"},
			{Name: "OpenTelemetry metrics", Class: NextBest, Source: "https://opentelemetry.io/docs/specs/otel/metrics/"},
			{Name: "Datadog DogStatsD", Class: NextBest, Source: "https://docs.datadoghq.com/developers/dogstatsd/"},
		},
	},
	{
		Capability: "cache_cost_accounting",
		NativePath: "internal/cacheprice/cacheprice.go",
		Workload:   "same provider-observed prompt and resident-prefix trace, service SKU, rates, billing period, and independent bill reconciliation across every arm",
		Metrics:    []string{"admission_token_equivalence", "billed_unit_error", "latency_ms", "bytes_processed", "peak_rss_bytes", "total_cost"},
		Alternatives: []Alternative{
			{Name: "charge full prompt", Class: TunedBaseline, Source: "internal/cacheprice/compare.go"},
			{Name: "AWS Pricing Calculator", Class: NextBest, Source: "https://calculator.aws/"},
			{Name: "Google Cloud Pricing Calculator", Class: NextBest, Source: "https://cloud.google.com/products/calculator"},
			{Name: "Azure Pricing Calculator", Class: NextBest, Source: "https://azure.microsoft.com/pricing/calculator/"},
		},
	},
	{
		Capability: "tool_call_rate_limiting",
		NativePath: "internal/ratelimit/ratelimit.go",
		Workload:   "same request arrival trace, call and cost caps, key dimension, window semantics, warmup, concurrency, and independent admission oracle across every arm",
		Metrics:    []string{"decision_equivalence", "overshoot_calls", "latency_ms", "throughput_calls_per_second", "state_bytes", "network_bytes", "peak_rss_bytes", "total_cost"},
		Alternatives: []Alternative{
			{Name: "no limiter", Class: TunedBaseline, Source: "internal/ratelimit/compare.go"},
			{Name: "Envoy local rate limit", Class: NextBest, Source: "https://www.envoyproxy.io/docs/envoy/latest/configuration/http/http_filters/local_rate_limit_filter"},
			{Name: "Kong rate limiting", Class: NextBest, Source: "https://docs.konghq.com/hub/kong-inc/rate-limiting/"},
			{Name: "Redis-cell", Class: NextBest, Source: "https://github.com/brandur/redis-cell"},
		},
	},
	{
		Capability: "engine_cache_invalidation",
		NativePath: "internal/enginecache/enginecache.go",
		Workload:   "same quarantined KV span plus dependent attention-index invalidation, engine state, process lifetime, warmup, and post-invalidation reuse oracle across every arm",
		Metrics:    []string{"poisoned_reuse_prevented", "invalidated_objects", "latency_ms", "control_requests", "bytes_transferred", "peak_rss_bytes", "total_cost"},
		Alternatives: []Alternative{
			{Name: "no invalidation", Class: TunedBaseline, Source: "internal/enginecache/compare.go"},
			{Name: "vLLM", Class: NextBest, Source: "https://docs.vllm.ai/"},
			{Name: "SGLang", Class: NextBest, Source: "https://docs.sglang.ai/"},
			{Name: "LMCache", Class: NextBest, Source: "https://docs.lmcache.ai/"},
			{Name: "fak + vLLM", Class: FirstClassIntegration, Integration: "vllm", Source: "cmd/fak/serve.go"},
			{Name: "fak + SGLang", Class: FirstClassIntegration, Integration: "sglang", Source: "cmd/fak/serve.go"},
		},
		Integrations: []string{"vllm", "sglang"},
	},
	{
		Capability: "tool_result_caching",
		NativePath: "internal/vdso/vdso.go",
		Workload:   "same deterministic tool calls, result bytes, upstream service, cache budget, concurrency, warmup, invalidation trace, and correctness oracle across every arm",
		Metrics:    []string{"output_equivalence", "hit_rate", "latency_ms", "upstream_calls", "peak_rss_bytes", "total_cost"},
		Alternatives: []Alternative{
			{Name: "uncached optimized upstream", Class: TunedBaseline, Source: "internal/vdso/compare.go"},
			{Name: "Redis client-side/server-assisted cache", Class: NextBest, Source: "https://redis.io/docs/latest/develop/clients/client-side-caching/"},
			{Name: "Momento Cache", Class: NextBest, Source: "https://www.gomomento.com/"},
		},
	},
	{
		Capability: "context_memory_management",
		NativePath: "internal/ctxmmu/mmu.go",
		Workload:   "same long-horizon tasks, candidate memory writes, read-back queries, model, token budget, process lifetime, and independent grader across every arm",
		Metrics:    []string{"task_success", "write_precision", "retained_fact_recall", "input_tokens", "latency_ms", "peak_rss_bytes", "total_cost"},
		Alternatives: []Alternative{
			{Name: "retain full history without memory management", Class: TunedBaseline, Source: "internal/ctxmmu/compare.go"},
			{Name: "Letta", Class: NextBest, Source: "https://docs.letta.com/"},
			{Name: "fak + mem0", Class: FirstClassIntegration, Integration: "mem0", Source: "docs/integrations/agent-memory.md"},
			{Name: "fak + Letta", Class: FirstClassIntegration, Integration: "letta", Source: "docs/integrations/agent-memory.md"},
			{Name: "fak + Zep/Graphiti", Class: FirstClassIntegration, Integration: "zep/graphiti", Source: "docs/integrations/agent-memory.md"},
			{Name: "fak + LangMem/LangGraph memory", Class: FirstClassIntegration, Integration: "langmem", Source: "docs/integrations/agent-memory.md"},
		},
		Integrations: []string{"mem0", "letta", "zep/graphiti", "langmem"},
	},
	{
		Capability: "policy_adjudication",
		NativePath: "internal/adjudicator/decide.go",
		Workload:   "same structural policy semantics, tool-call corpus, process lifetime, warmup, concurrency, and correctness oracle across every arm",
		Metrics:    []string{"verdict_equivalence", "policy_coverage", "latency_ms", "throughput_calls_per_second", "peak_rss_bytes", "total_cost"},
		Alternatives: []Alternative{
			{Name: "direct allow/deny lookup (tuned no-engine baseline)", Class: TunedBaseline, Source: "internal/adjudicator/compare.go"},
			{Name: "OPA/Rego", Class: NextBest, Source: "https://www.openpolicyagent.org/"},
			{Name: "Cedar", Class: NextBest, Source: "https://www.cedarpolicy.com/"},
		},
	},
	{
		Capability: "model_routing",
		NativePath: "internal/modelroute/modelroute.go",
		Workload:   "same prompts, candidate models, task-quality grader, concurrency, warmup, and provider conditions across every arm",
		Metrics:    []string{"task_success", "route_quality", "latency_ms", "input_tokens", "output_tokens", "peak_rss_bytes", "total_cost"},
		Alternatives: []Alternative{
			{Name: "fixed strongest model (tuned no-routing baseline)", Class: TunedBaseline, Source: "internal/modelroute/compare.go"},
			{Name: "RouteLLM", Class: NextBest, Source: "https://github.com/lm-sys/RouteLLM"},
			{Name: "fak + LiteLLM Router", Class: FirstClassIntegration, Integration: "litellm", Source: "docs/integrations/litellm.md"},
			{Name: "fak + OpenRouter routing", Class: FirstClassIntegration, Integration: "openrouter", Source: "docs/integrations/openrouter.md"},
			{Name: "fak + Portkey router", Class: FirstClassIntegration, Integration: "portkey", Source: "docs/integrations/portkey.md"},
		},
		Integrations: []string{"litellm", "openrouter", "portkey"},
	},
	{
		Capability: "tokenization",
		NativePath: "internal/tokenizer/tokenizer.go",
		Workload:   "same model tokenizer artifact, text corpus, special-token policy, warmup, and correctness oracle across every arm",
		Metrics:    []string{"exact_token_ids", "decode_roundtrip", "tokens_per_second", "latency_ms", "peak_rss_bytes", "initialization_ms", "total_cost"},
		Alternatives: []Alternative{
			{Name: "exhaustive adjacent-pair BPE scan (tuned incumbent)", Class: TunedBaseline, Source: "internal/tokenizer/bpe_merge_test.go"},
			{Name: "llama.cpp tokenizer", Class: NextBest, Source: "internal/tokenizer/oracle_qwen_test.go"},
			{Name: "fak + Hugging Face tokenizers", Class: FirstClassIntegration, Integration: "huggingface/tokenizers", Source: "internal/tokenizer/tokenizer_test.go"},
		},
		Integrations: []string{"huggingface/tokenizers"},
	},
	{
		Capability: "tool_filtering",
		NativePath: "internal/gateway/mcp_defer.go",
		Workload:   "same tool catalog, requests, model, provider cache state, and correctness grader across every arm",
		Metrics:    []string{"task_success", "tool_recall", "input_tokens", "ttft_ms", "total_cost"},
		Alternatives: []Alternative{
			{Name: "all tool schemas (tuned no-filter baseline)", Class: TunedBaseline, Source: "in-repository A/B arm"},
			{Name: "retrieval-based tool selection (ToolRAG class)", Class: NextBest, Source: "https://arxiv.org/abs/2403.06011"},
		},
	},
	{
		Capability: "prefix_kv_reuse",
		NativePath: "internal/radixkv/radixkv.go",
		Workload:   "same model, shared-prefix request traces, cache budget, concurrency, warmup, and correctness checks across every arm",
		Metrics:    []string{"output_equivalence", "prefix_hit_rate", "ttft_ms", "throughput_tokens_per_second", "kv_bytes", "total_cost"},
		Alternatives: []Alternative{
			{Name: "provider/vLLM prefix caching disabled (tuned no-reuse baseline)", Class: TunedBaseline, Source: "docs/benchmarks/RADIXATTENTION-RESULTS.md"},
			{Name: "SGLang RadixAttention", Class: NextBest, Source: "https://arxiv.org/abs/2312.07104"},
			{Name: "fak + llm-d prefix-cache-aware routing", Class: FirstClassIntegration, Integration: "llm-d", Source: "docs/integrations/llm-d.md"},
		},
		Integrations: []string{"llm-d"},
	},
	{
		Capability: "context_compression",
		NativePath: "internal/headroom/native.go",
		Workload:   "same long-horizon transcripts, model, context budget, and end-task grader across every arm",
		Metrics:    []string{"task_success", "retained_fact_recall", "input_tokens", "latency_ms", "total_cost"},
		Alternatives: []Alternative{
			{Name: "full history with provider caching (tuned no-compression baseline)", Class: TunedBaseline, Source: "in-repository A/B arm"},
			{Name: "LongLLMLingua", Class: NextBest, Source: "https://arxiv.org/abs/2310.06839"},
			{Name: "fak + LLMLingua-2 compressor", Class: FirstClassIntegration, Integration: "headroom/lingua", Source: "https://github.com/anthony-chaudhary/fak/issues/3204"},
		},
		Integrations: []string{"headroom/lingua"},
	},
}

func All() []Contract {
	out := append([]Contract(nil), contracts...)
	sort.Slice(out, func(i, j int) bool { return out[i].Capability < out[j].Capability })
	return out
}

func validateClassifications(classifications []LeafClassification, cs []Contract) []Finding {
	var findings []Finding
	knownCapabilities := make(map[string]struct{}, len(cs))
	for _, contract := range cs {
		knownCapabilities[contract.Capability] = struct{}{}
	}
	seenLeaves := make(map[string]struct{}, len(classifications))
	for _, classification := range classifications {
		if classification.Leaf == "" || !strings.HasPrefix(classification.Leaf, "internal/") {
			findings = append(findings, Finding{Reason: fmt.Sprintf("classification leaf %q must name an internal/ package", classification.Leaf)})
		}
		if _, duplicate := seenLeaves[classification.Leaf]; duplicate {
			findings = append(findings, Finding{Reason: fmt.Sprintf("classification leaf %q appears more than once", classification.Leaf)})
		}
		seenLeaves[classification.Leaf] = struct{}{}
		if strings.TrimSpace(classification.Reason) == "" {
			findings = append(findings, Finding{Reason: fmt.Sprintf("classification leaf %q has no reason", classification.Leaf)})
		}
		switch classification.Disposition {
		case DispositionInfrastructure:
			if len(classification.Capabilities) != 0 {
				findings = append(findings, Finding{Reason: fmt.Sprintf("infrastructure leaf %q must not declare capabilities", classification.Leaf)})
			}
		case DispositionCapability, DispositionMultiCapability:
			if len(classification.Capabilities) == 0 {
				findings = append(findings, Finding{Reason: fmt.Sprintf("%s leaf %q must declare at least one capability", classification.Disposition, classification.Leaf)})
			}
			seenCapabilities := make(map[string]struct{}, len(classification.Capabilities))
			for _, capability := range classification.Capabilities {
				if _, duplicate := seenCapabilities[capability]; duplicate {
					findings = append(findings, Finding{Capability: capability, Reason: fmt.Sprintf("leaf %q declares the capability more than once", classification.Leaf)})
				}
				seenCapabilities[capability] = struct{}{}
				if _, exists := knownCapabilities[capability]; !exists {
					findings = append(findings, Finding{Capability: capability, Reason: fmt.Sprintf("leaf %q references an unknown benchmark contract", classification.Leaf)})
				}
			}
		default:
			findings = append(findings, Finding{Reason: fmt.Sprintf("classification leaf %q has invalid disposition %q", classification.Leaf, classification.Disposition)})
		}
	}
	return findings
}

func Validate(cs []Contract) []Finding {
	var findings []Finding
	seen := map[string]bool{}
	for _, c := range cs {
		if c.Capability == "" || seen[c.Capability] {
			findings = append(findings, Finding{c.Capability, "capability must be non-empty and unique"})
		}
		seen[c.Capability] = true
		if c.NativePath == "" {
			findings = append(findings, Finding{c.Capability, "native implementation path is required"})
		}
		if c.Workload == "" {
			findings = append(findings, Finding{c.Capability, "shared workload contract is required"})
		}
		if len(c.Metrics) == 0 {
			findings = append(findings, Finding{c.Capability, "quality, cost, and latency metrics are required"})
		}
		classes := map[AlternativeClass]int{}
		integrations := map[string]int{}
		for _, a := range c.Alternatives {
			classes[a.Class]++
			if a.Name == "" || a.Source == "" {
				findings = append(findings, Finding{c.Capability, "every alternative needs a name and source"})
			}
			if a.Class == FirstClassIntegration {
				if a.Integration == "" {
					findings = append(findings, Finding{c.Capability, "first-class integration alternative needs its integration id"})
				}
				integrations[a.Integration]++
			}
		}
		if classes[TunedBaseline] == 0 {
			findings = append(findings, Finding{c.Capability, "tuned baseline arm is required"})
		}
		if classes[NextBest]+classes[FirstClassIntegration] == 0 {
			findings = append(findings, Finding{c.Capability, "next-best alternative arm is required"})
		}
		for id, n := range integrations {
			if n > 1 {
				findings = append(findings, Finding{c.Capability, fmt.Sprintf("integration %q appears more than once", id)})
			}
		}
		declaredIntegrations := make(map[string]bool, len(c.Integrations))
		for _, id := range c.Integrations {
			if id == "" || declaredIntegrations[id] {
				findings = append(findings, Finding{c.Capability, "integration inventory entries must be non-empty and unique"})
			}
			declaredIntegrations[id] = true
			if integrations[id] == 0 {
				findings = append(findings, Finding{c.Capability, fmt.Sprintf("equivalent first-class integration %q has no comparison arm", id)})
			}
		}
		for id := range integrations {
			if !declaredIntegrations[id] {
				findings = append(findings, Finding{c.Capability, fmt.Sprintf("integration arm %q is absent from the equivalent-integration inventory", id)})
			}
		}
		if c.Witness == "" {
			findings = append(findings, Finding{c.Capability, "benchmark witness is missing"})
		} else if _, err := os.Stat(filepath.Clean(c.Witness)); err != nil {
			findings = append(findings, Finding{c.Capability, fmt.Sprintf("benchmark witness %q is not readable: %v", c.Witness, err)})
		}
	}
	return findings
}

func DiscoverNativeLeaves(root string) ([]string, error) {
	entries, err := os.ReadDir(filepath.Join(root, "internal"))
	if err != nil {
		return nil, fmt.Errorf("discover native leaves: %w", err)
	}
	var leaves []string
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		files, err := os.ReadDir(filepath.Join(root, "internal", entry.Name()))
		if err != nil {
			return nil, fmt.Errorf("discover native leaf %s: %w", entry.Name(), err)
		}
		hasCode := false
		for _, file := range files {
			name := file.Name()
			if !file.IsDir() && filepath.Ext(name) == ".go" && !strings.HasSuffix(name, "_test.go") {
				hasCode = true
				break
			}
		}
		if hasCode {
			leaves = append(leaves, entry.Name())
		}
	}
	sort.Strings(leaves)
	return leaves, nil
}

func AuditRoot(root string) Report {
	cs := All()
	fs := Validate(cs)
	fs = append(fs, validateClassifications(leafClassifications, cs)...)
	leaves, err := DiscoverNativeLeaves(root)
	coverage := Coverage{DispositionCounts: make(map[LeafDisposition]int)}
	if err != nil {
		fs = append(fs, Finding{Reason: err.Error()})
	} else {
		coverage.DiscoveryComplete = true
		coverage.NativeLeaves = len(leaves)
		contractByCapability := make(map[string]struct{}, len(cs))
		contractLeafByCapability := make(map[string]string, len(cs))
		for _, contract := range cs {
			contractByCapability[contract.Capability] = struct{}{}
			nativePath := filepath.Clean(contract.NativePath)
			if filepath.Ext(nativePath) == ".go" {
				nativePath = filepath.Dir(nativePath)
			}
			contractLeafByCapability[contract.Capability] = filepath.ToSlash(nativePath)
		}
		classificationByLeaf := make(map[string]LeafClassification, len(leafClassifications))
		for _, classification := range leafClassifications {
			classificationByLeaf[classification.Leaf] = classification
		}
		leafSet := make(map[string]struct{}, len(leaves))
		for _, leafName := range leaves {
			leaf := "internal/" + leafName
			leafSet[leaf] = struct{}{}
			classification, classified := classificationByLeaf[leaf]
			if !classified {
				coverage.UnclassifiedLeafNames = append(coverage.UnclassifiedLeafNames, leaf)
				continue
			}
			coverage.Classifications = append(coverage.Classifications, classification)
			coverage.DispositionCounts[classification.Disposition]++
			covered := classification.Disposition == DispositionInfrastructure
			if classification.Disposition != DispositionInfrastructure {
				covered = len(classification.Capabilities) > 0
				for _, capability := range classification.Capabilities {
					if _, ok := contractByCapability[capability]; !ok || contractLeafByCapability[capability] != leaf {
						covered = false
					}
				}
			}
			if covered {
				coverage.CoveredLeaves++
			} else {
				coverage.MissingLeaves = append(coverage.MissingLeaves, leaf)
			}
		}
		coverage.ClassifiedLeaves = len(coverage.Classifications)
		coverage.UnclassifiedLeaves = len(coverage.UnclassifiedLeafNames)
		coverage.MissingLeaves = append(coverage.MissingLeaves, coverage.UnclassifiedLeafNames...)
		for _, contract := range cs {
			leaf := contractLeafByCapability[contract.Capability]
			if _, ok := leafSet[leaf]; !ok {
				coverage.OrphanContracts = append(coverage.OrphanContracts, leaf)
			}
		}
		sort.Strings(coverage.MissingLeaves)
		sort.Strings(coverage.UnclassifiedLeafNames)
		sort.Strings(coverage.OrphanContracts)
		sort.Slice(coverage.Classifications, func(i, j int) bool { return coverage.Classifications[i].Leaf < coverage.Classifications[j].Leaf })
		if coverage.UnclassifiedLeaves > 0 {
			fs = append(fs, Finding{Reason: fmt.Sprintf("%d native leaves are unclassified (capability, multi_capability, or infrastructure)", coverage.UnclassifiedLeaves)})
		}
		classifiedMissing := len(coverage.MissingLeaves) - coverage.UnclassifiedLeaves
		if classifiedMissing > 0 {
			fs = append(fs, Finding{Reason: fmt.Sprintf("%d classified native leaves have no matching comparison contract", classifiedMissing)})
		}
		if len(coverage.OrphanContracts) > 0 {
			fs = append(fs, Finding{Reason: fmt.Sprintf("%d contracts do not map to a discovered native leaf", len(coverage.OrphanContracts))})
		}
	}
	return Report{Contracts: cs, Coverage: coverage, Findings: fs, Complete: len(fs) == 0}
}

func Audit() Report {
	root, err := os.Getwd()
	if err != nil {
		return Report{Contracts: All(), Findings: []Finding{{Reason: err.Error()}}}
	}
	return AuditRoot(root)
}
