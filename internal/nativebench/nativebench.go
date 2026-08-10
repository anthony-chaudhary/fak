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
	Leaf: "internal/laneadmit", Disposition: DispositionMultiCapability,
	Capabilities: []string{"lane_tree_collision_admission"},
	Reason:       "native shared lane/tree collision, exclusivity, ancestry, read-only, and self-renewal admission",
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
