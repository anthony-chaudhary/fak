package gateway

import (
	"context"
	"encoding/json"
	"net/http"
	"runtime"
	"sort"
	"strings"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/guardvars"
	"github.com/anthony-chaudhary/fak/internal/resumemetrics"
)

type debugVarsResponse struct {
	Gateway          debugGatewayVars           `json:"gateway"`
	Runtime          debugRuntimeVars           `json:"runtime"`
	Kernel           debugKernelVars            `json:"kernel"`
	Inference        debugInferenceVars         `json:"inference"`
	Upstream         debugUpstreamVars          `json:"upstream"`
	VCache           *debugVCacheVars           `json:"vcache,omitempty"`
	CacheAttribution *debugCacheAttributionVars `json:"cache_attribution,omitempty"`
	ManagedCache     *debugManagedCacheVars     `json:"managed_cache,omitempty"`
	TokenSavings     debugTokenSavingsVars      `json:"token_savings"`
	// ShrinkLevers is the #5493 prompt-shrink-lever posture: which of the three levers are
	// configured ON, and which of those the wire this gateway actually built can run. It sits
	// beside ManagedCache because it answers the same class of question for a different lever
	// family — configured-vs-live — and because the three levers it covers are the ones whose
	// silent inertness on a self-hosted wire would otherwise be read as a verdict on the
	// kernel. Omitted when no lever is configured on. See shrink_lever_live.go.
	ShrinkLevers     *debugShrinkLeverVars          `json:"shrink_levers,omitempty"`
	VCacheFamilies   *debugVCacheFamiliesVars       `json:"vcache_families,omitempty"`
	VCacheGovernor   []vcacheGovernorDecisionRecord `json:"vcache_governor_journal,omitempty"`
	VCacheGovQuality *vcacheGovernorQualityVars     `json:"vcache_governor_quality,omitempty"`
	VCacheWarmth     []vcacheWarmthDemotionRecord   `json:"vcache_warmth_demotions,omitempty"`
	ModelLoad        *debugModelLoadVars            `json:"model_load,omitempty"`
	// Startup is the named, structured home for one-shot boot state. It keeps the
	// timeline, model-load detail, and startup messages queryable after readiness;
	// legacy model_load/startup_report fields remain during migration.
	Startup       debugStartupVars        `json:"startup"`
	KVMemory      *debugKVMemoryVars      `json:"kv_memory,omitempty"`
	RequestMemory *debugRequestMemoryVars `json:"request_memory,omitempty"`
	// MoEResidency is what a serve that declared an expert budget paid to keep the top-k
	// activated experts resident (R6, #5617). Omitted unless some request actually engaged a
	// routed-expert ring, so its absence means "not engaged" rather than "engaged, cost zero".
	MoEResidency   *debugMoEResidencyVars    `json:"moe_residency,omitempty"`
	Sessions       []debugSessionVars        `json:"sessions,omitempty"`
	Assumptions    []SessionAssumption       `json:"assumptions,omitempty"`
	ContextQueries []ContextQueryAuditRecord `json:"context_queries,omitempty"`
	// Endpoints is the live accounts+nodes block — which Claude seats and which serving
	// nodes THIS session is using (fak guard's status area). Nil/omitted unless the host
	// set a provider (SetSessionEndpointsProvider) that has something to report.
	Endpoints *SessionEndpoints `json:"endpoints,omitempty"`
	// Adjudication is the verdict roll-up (the same AdjudicationSummary the guard exit
	// summary folds): tally + by-reason + compaction/tool-prune/deny-all — promoted here
	// so the live pane shows what the kernel blocked/repaired/quarantined instead of the
	// vacuous kernel.Counters (which the Decide proxy never increments). Omitted on a
	// cold gateway that has decided nothing.
	Adjudication *AdjudicationSummary `json:"adjudication,omitempty"`
	// Harness is the live harness-resource block (kernel/agent CPU/RSS/IO/net) — the
	// /debug/vars twin of the /metrics-only fak_harness_* family, so the pane can show
	// live what the exit summary prints. Nil/omitted until the host samples a session.
	Harness *SessionHarness `json:"harness,omitempty"`
	// Fleet is the live cross-MACHINE fleet aggregate — how many operator boxes published
	// snapshots, the stale/action split, and rolled-up totals — so the pane shows the
	// fleet-of-machines health beside the session-local blocks. Nil/omitted unless the
	// host set a provider (SetSessionFleetProvider) that reports a non-empty fleet.
	Fleet *SessionFleet `json:"fleet,omitempty"`
	// StartupReport is the full human-readable startup report the host recorded at boot
	// (fak guard's banner + hook/auth notes) — what `fak info --startup` prints when an
	// attended launch kept the terminal banner compact. Omitted when the host set none.
	StartupReport string `json:"startup_report,omitempty"`
	// Watchdog is the process-global resume/heal watchdog counter surface (#3803): tick
	// count, per-verdict action mix, autoheal-result mix, witnessed post-resume progress,
	// and the last folded monitor/rollup health. Because these are expvars incremented at
	// the point of decision, this block reflects whatever watchdog/autoheal work ran IN THIS
	// process (the in-guard autoheal path fills it directly). Omitted on a cold process that
	// has recorded no watchdog signal at all.
	Watchdog *resumemetrics.Snapshot `json:"watchdog,omitempty"`
	Metrics  debugMetricsVars        `json:"metrics"`
}

// debugInferenceVars surfaces the model-generation throughput the kernel/vDSO counters
// structurally cannot show on a pure chat/proxy workload (they stay 0 — no syscall, no
// fast-path lookup), so an operator watching the panel sees real decode work instead of
// a dead-looking "submits 0". The two rates separate the cold prefill that dominates a
// slow FIRST request (PrefillTokensPerSecond) from steady-state generation
// (DecodeTokensPerSecond); both are measured only over the streaming turns that could
// observe a first-token boundary (TTFTTurns), so they never blend a measured turn with a
// buffered one. InflightMaxAgeSeconds is the oldest in-flight request's age — the
// hung/slow-request detector the completion histograms cannot show until a request ends.
type debugInferenceVars struct {
	Turns                  uint64  `json:"turns"`
	PromptTokens           uint64  `json:"prompt_tokens"`
	CompletionTokens       uint64  `json:"completion_tokens"`
	DurationSeconds        float64 `json:"duration_seconds"`
	OutputTokensPerSecond  float64 `json:"output_tokens_per_second"`
	TTFTTurns              uint64  `json:"ttft_turns"`
	PrefillSeconds         float64 `json:"prefill_seconds"`
	MeanTTFTSeconds        float64 `json:"mean_ttft_seconds"`
	PrefillTokensPerSecond float64 `json:"prefill_tokens_per_second"`
	DecodeTokensPerSecond  float64 `json:"decode_tokens_per_second"`
	InflightMaxAgeSeconds  float64 `json:"inflight_max_age_seconds"`
}

// debugUpstreamVars mirrors the upstream-error /metrics families into /debug/vars so live
// operator panes can show provider/API incidents without scraping Prometheus text.
type debugUpstreamVars struct {
	ErrorsByKind map[string]uint64 `json:"errors_by_kind"`
	Retries      uint64            `json:"retries"`
	// RetryWaitSeconds is the accumulated wall-clock the backoff loop slept between
	// those retries — the time twin of Retries, so a live pane can show how much of
	// the session's slowness was provider pushback fak absorbed.
	RetryWaitSeconds     float64           `json:"retry_wait_seconds"`
	AuthRefreshByOutcome map[string]uint64 `json:"auth_refresh_by_outcome"`
	// ForbiddenRetryByOutcome mirrors the 403 transient-recovery family: recovered (a
	// bounded retry cleared a transient abuse/capacity gate) vs exhausted (a permanent
	// entitlement 403 surfaced). The permission-flap twin of AuthRefreshByOutcome.
	ForbiddenRetryByOutcome map[string]uint64 `json:"forbidden_retry_by_outcome"`
	// AccountFailoverByOutcome mirrors the account-scoped failover family: recovered (a
	// 403/402 named this credential's org/region/billing as walled and a permitted sibling
	// account was adopted so the walled turn completed in place — a heal re-login cannot do)
	// vs exhausted (no permitted sibling existed and the account-scoped denial surfaced). The
	// org-OAuth-disabled twin of ForbiddenRetryByOutcome — its "recovered" count is the number
	// of sessions auto-switched off a walled account instead of dying on a futile /login.
	AccountFailoverByOutcome map[string]uint64 `json:"account_failover_by_outcome"`
	// LastForbiddenDetail is a SCRUBBED, bounded snapshot of the most recent PERSISTENT 403's
	// upstream body — the one operator-side signal that tells org-disabled apart from
	// model-not-permitted apart from an abuse gate, which the 2026-07-03 gem8 storm proved was
	// lost (the raw body never crosses to the client per #82/#346, and nothing recorded it for
	// the operator). This surface is loopback-only, so the body stays operator-side; it is run
	// through the same secret scrubber the logs use before it is stored, and empty until a
	// persistent 403 lands.
	LastForbiddenDetail string `json:"last_forbidden_detail,omitempty"`
	// ProviderExtraBodySet reports whether the live HTTP upstream planner carries a
	// provider-specific extra request body. ProviderExtraBodyKeys carries only the
	// top-level keys, never values, so an operator can verify a Qwen/vLLM/SGLang tuning
	// posture without leaking raw request config.
	ProviderExtraBodySet  bool     `json:"provider_extra_body_set"`
	ProviderExtraBodyKeys []string `json:"provider_extra_body_keys,omitempty"`
}

type debugGatewayVars struct {
	Up               bool    `json:"up"`
	Version          string  `json:"version"`
	Engine           string  `json:"engine"`
	Model            string  `json:"model"`
	VDSO             bool    `json:"vdso"`
	AuthRequired     bool    `json:"auth_required"`
	StartTimeUnix    int64   `json:"start_time_unix"`
	UptimeSeconds    float64 `json:"uptime_seconds"`
	InflightRequests int64   `json:"inflight_requests"`
}

type debugRuntimeVars struct {
	GoVersion    string          `json:"go_version"`
	GOOS         string          `json:"goos"`
	GOARCH       string          `json:"goarch"`
	NumCPU       int             `json:"num_cpu"`
	GOMAXPROCS   int             `json:"gomaxprocs"`
	NumGoroutine int             `json:"num_goroutine"`
	Memory       debugMemoryVars `json:"memory"`
}

type debugMemoryVars struct {
	AllocBytes      uint64 `json:"alloc_bytes"`
	TotalAllocBytes uint64 `json:"total_alloc_bytes"`
	SysBytes        uint64 `json:"sys_bytes"`
	HeapAllocBytes  uint64 `json:"heap_alloc_bytes"`
	HeapSysBytes    uint64 `json:"heap_sys_bytes"`
	HeapObjects     uint64 `json:"heap_objects"`
	StackInuseBytes uint64 `json:"stack_inuse_bytes"`
	NextGCBytes     uint64 `json:"next_gc_bytes"`
	LastGCUnixNano  uint64 `json:"last_gc_unix_nano"`
	NumGC           uint32 `json:"num_gc"`

	// The three below are DERIVED, and they are the ones that catch an allocation
	// blowup. Raw MemStats were already published here and still did not surface
	// the session-ledger O(n^2): Alloc looks unremarkable between collections and
	// heap_objects stays LOW precisely when the problem is a few enormous buffers.
	//
	// AllocBytesPerSec is the cumulative allocation RATE since boot. A healthy
	// proxy turn allocates a few MB; the ledger bug ran at ~484 MB/s sustained
	// (1.19 TB in 41 min on one guard). Rate is the tell that a per-turn cost has
	// gone superlinear.
	AllocBytesPerSec uint64 `json:"alloc_bytes_per_sec"`
	// PeakHeapSysBytes is the high-water mark of heap memory taken from the OS.
	// Sawtooth RSS means an instantaneous sample almost never catches the peak,
	// which is exactly the number that decides whether this process is why the
	// machine is swapping.
	PeakHeapSysBytes uint64 `json:"peak_heap_sys_bytes"`
	// PeakAllocBytes is the high-water mark of live heap.
	PeakAllocBytes uint64 `json:"peak_alloc_bytes"`
}

// newDebugMemoryVars renders one MemStats sample plus the derived rate/peak fields.
func newDebugMemoryVars(mem *runtime.MemStats) debugMemoryVars {
	perSec, peakSys, peakAlloc := procMemory.observe(mem)
	return debugMemoryVars{
		AllocBytes:       mem.Alloc,
		TotalAllocBytes:  mem.TotalAlloc,
		SysBytes:         mem.Sys,
		HeapAllocBytes:   mem.HeapAlloc,
		HeapSysBytes:     mem.HeapSys,
		HeapObjects:      mem.HeapObjects,
		StackInuseBytes:  mem.StackInuse,
		NextGCBytes:      mem.NextGC,
		LastGCUnixNano:   mem.LastGC,
		NumGC:            mem.NumGC,
		AllocBytesPerSec: perSec,
		PeakHeapSysBytes: peakSys,
		PeakAllocBytes:   peakAlloc,
	}
}

// memoryWatermarks tracks the high-water marks MemStats itself does not keep.
// Sampled on each /debug/vars read (the info pane polls every 2s), which is
// frequent enough to catch a sawtooth's crest and costs nothing when nobody looks.
type memoryWatermarks struct {
	bootUnixNano int64
	peakHeapSys  atomic.Uint64
	peakAlloc    atomic.Uint64
}

var procMemory = memoryWatermarks{bootUnixNano: time.Now().UnixNano()}

// observe folds one MemStats sample into the watermarks and returns the derived
// fields. Monotonic maxima via CAS: concurrent readers race benignly.
func (m *memoryWatermarks) observe(ms *runtime.MemStats) (perSec, peakSys, peakAlloc uint64) {
	for {
		old := m.peakHeapSys.Load()
		if ms.HeapSys <= old || m.peakHeapSys.CompareAndSwap(old, ms.HeapSys) {
			break
		}
	}
	for {
		old := m.peakAlloc.Load()
		if ms.Alloc <= old || m.peakAlloc.CompareAndSwap(old, ms.Alloc) {
			break
		}
	}
	if elapsed := time.Now().UnixNano() - m.bootUnixNano; elapsed > 0 {
		perSec = uint64(float64(ms.TotalAlloc) / (float64(elapsed) / 1e9))
	}
	return perSec, m.peakHeapSys.Load(), m.peakAlloc.Load()
}

type debugKernelVars struct {
	Submits      int64   `json:"submits"`
	VDSOHits     int64   `json:"vdso_hits"`
	EngineCalls  int64   `json:"engine_calls"`
	Denies       int64   `json:"denies"`
	Transforms   int64   `json:"transforms"`
	Quarantines  int64   `json:"quarantines"`
	ResultDenies int64   `json:"result_denies"`
	Admitted     int64   `json:"admitted"`
	VDSOHitRatio float64 `json:"vdso_hit_ratio"`
}

type debugMetricsVars struct {
	HTTP                 []debugHTTPMetricVars           `json:"http"`
	Operations           []debugOperationMetricVars      `json:"operations"`
	Compaction           debugCompactionVars             `json:"compaction"`
	RequestMemory        []debugRequestMemoryMetricVars  `json:"request_memory,omitempty"`
	RequestMemoryFit     []debugRequestMemoryFitVars     `json:"request_memory_fit,omitempty"`
	RequestMemoryTokens  []debugRequestMemoryTokenVars   `json:"request_memory_tokens,omitempty"`
	InKernelOOM          []debugInKernelOOMVars          `json:"in_kernel_oom"`
	InKernelOOMRetries   []debugInKernelOOMRetryVars     `json:"in_kernel_oom_retries,omitempty"`
	InKernelPressureTrim []debugInKernelPressureTrimVars `json:"in_kernel_pressure_trims,omitempty"`
}

type debugModelLoadVars struct {
	Source              string                         `json:"source"`
	Mode                string                         `json:"mode"`
	TotalSeconds        float64                        `json:"total_seconds"`
	Bytes               int64                          `json:"bytes"`
	Tensors             int                            `json:"tensors"`
	Bottleneck          string                         `json:"bottleneck"`
	Phases              []debugModelLoadPhaseVars      `json:"phases"`
	MemoryPlan          []debugModelLoadMemoryPlanVars `json:"memory_plan,omitempty"`
	MemoryCapacities    []debugModelLoadCapacityVars   `json:"memory_capacities,omitempty"`
	MemoryFit           []debugMemoryFitVars           `json:"memory_fit,omitempty"`
	MemoryHeadroomRatio float64                        `json:"memory_headroom_ratio,omitempty"`
	LoadPaths           []debugModelLoadPathVars       `json:"load_paths,omitempty"`
	Messages            []StartupMessage               `json:"messages,omitempty"`
}

type debugModelLoadPathVars struct {
	QuantType       string `json:"quant_type"`
	Class           string `json:"class"`
	ResidentTensors int    `json:"resident_tensors"`
	ResidentBytes   int64  `json:"resident_bytes"`
	DequantTensors  int    `json:"dequant_tensors"`
	DequantBytes    int64  `json:"dequant_bytes"`
}

type debugStartupVars struct {
	Status             string                  `json:"status"`
	StartedAt          string                  `json:"started_at,omitempty"`
	ReadyAt            string                  `json:"ready_at,omitempty"`
	TimeToReadySeconds float64                 `json:"time_to_ready_seconds"`
	UnaccountedSeconds float64                 `json:"unaccounted_seconds"`
	Phases             []debugStartupPhaseVars `json:"phases,omitempty"`
	Messages           []StartupMessage        `json:"messages,omitempty"`
	ModelLoad          *debugModelLoadVars     `json:"model_load,omitempty"`
}

type debugStartupPhaseVars struct {
	Name       string  `json:"name"`
	Seconds    float64 `json:"seconds"`
	Provenance string  `json:"provenance"`
	Stage      string  `json:"stage"`
}

type debugModelLoadPhaseVars struct {
	Phase   string  `json:"phase"`
	Seconds float64 `json:"seconds"`
	Bytes   int64   `json:"bytes"`
	Tensors int     `json:"tensors"`
}

type debugModelLoadMemoryPlanVars struct {
	Class  string `json:"class"`
	Scope  string `json:"scope"`
	Bytes  int64  `json:"bytes"`
	Detail string `json:"detail,omitempty"`
	DType  string `json:"dtype,omitempty"`
}

type debugModelLoadCapacityVars struct {
	Scope      string `json:"scope"`
	TotalBytes int64  `json:"total_bytes"`
	FreeBytes  int64  `json:"free_bytes,omitempty"`
	Known      bool   `json:"known"`
	FreeKnown  bool   `json:"free_known"`
}

type debugMemoryFitVars struct {
	Scope         string `json:"scope"`
	WantBytes     int64  `json:"want_bytes"`
	BudgetBytes   int64  `json:"budget_bytes,omitempty"`
	MarginBytes   int64  `json:"margin_bytes,omitempty"`
	CapacityKnown bool   `json:"capacity_known"`
	FreeKnown     bool   `json:"free_known"`
}

type debugKVMemoryVars struct {
	Enabled               bool    `json:"enabled"`
	Backend               string  `json:"backend"`
	MemoryClass           string  `json:"memory_class"`
	Scope                 string  `json:"scope"`
	DType                 string  `json:"dtype,omitempty"`
	BytesPerToken         int64   `json:"bytes_per_token"`
	ResidentTokens        int     `json:"resident_tokens,omitempty"`
	ResidentBytes         int64   `json:"resident_bytes,omitempty"`
	CapacityKnown         bool    `json:"capacity_known"`
	CapacityFreeKnown     bool    `json:"capacity_free_known"`
	CapacityTotalBytes    int64   `json:"capacity_total_bytes,omitempty"`
	CapacityFreeBytes     int64   `json:"capacity_free_bytes,omitempty"`
	HeadroomRatio         float64 `json:"headroom_ratio,omitempty"`
	FitBudgetBytes        int64   `json:"fit_budget_bytes,omitempty"`
	FitMarginBytes        int64   `json:"fit_margin_bytes,omitempty"`
	BudgetTokens          int     `json:"budget_tokens,omitempty"`
	LRUTokens             int     `json:"lru_tokens,omitempty"`
	MaxDepthTokens        int     `json:"max_depth_tokens,omitempty"`
	Nodes                 int     `json:"nodes,omitempty"`
	Leaves                int     `json:"leaves,omitempty"`
	Evictions             int     `json:"evictions,omitempty"`
	PolicyEvictions       int     `json:"policy_evictions,omitempty"`
	Splits                int     `json:"splits,omitempty"`
	L1DeviceResidentBytes int64   `json:"l1_device_resident_bytes,omitempty"`
	L1HostResidentBytes   int64   `json:"l1_host_resident_bytes,omitempty"`
	L2HostResidentBytes   int64   `json:"l2_host_resident_bytes,omitempty"`
	L2HostCapacityBytes   int64   `json:"l2_host_capacity_bytes,omitempty"`
	L1Hits                int     `json:"l1_hits,omitempty"`
	L1Misses              int     `json:"l1_misses,omitempty"`
	L1Faults              int     `json:"l1_faults,omitempty"`
	L1HitTokens           int     `json:"l1_hit_tokens,omitempty"`
	L2Hits                int     `json:"l2_hits,omitempty"`
	L2Misses              int     `json:"l2_misses,omitempty"`
	L2Faults              int     `json:"l2_faults,omitempty"`
	L2HitTokens           int     `json:"l2_hit_tokens,omitempty"`
	L2StageBytes          int64   `json:"l2_stage_bytes,omitempty"`
	L2RestoreBytes        int64   `json:"l2_restore_bytes,omitempty"`
	L2Evictions           int     `json:"l2_evictions,omitempty"`
	L3Enabled             bool    `json:"l3_enabled"`
	L3ReferencedBytes     int64   `json:"l3_referenced_bytes,omitempty"`
	L3Hits                int     `json:"l3_hits,omitempty"`
	L3Misses              int     `json:"l3_misses,omitempty"`
	L3Faults              int     `json:"l3_faults,omitempty"`
	L3HitTokens           int     `json:"l3_hit_tokens,omitempty"`
	L3StageBytes          int64   `json:"l3_stage_bytes,omitempty"`
	L3RestoreBytes        int64   `json:"l3_restore_bytes,omitempty"`
	L3StageNanos          int64   `json:"l3_stage_nanos,omitempty"`
	L3RestoreNanos        int64   `json:"l3_restore_nanos,omitempty"`
	L3StageFaults         int     `json:"l3_stage_faults,omitempty"`
	L3RestoreFaults       int     `json:"l3_restore_faults,omitempty"`
}

// debugMoEResidencyVars is the activated-expert residency block. Unlike the Prometheus family,
// which leaves ratios to PromQL, this one carries the derived rates precomputed: /debug/vars is
// read by a human answering "is the budget I declared the right size", and making them divide
// page-in bytes by forwarded tokens in their head is how that question goes unanswered.
type debugMoEResidencyVars struct {
	Requests int64 `json:"requests"`
	Tokens   int64 `json:"tokens"`

	Lookups     int64 `json:"lookups"`
	Hits        int64 `json:"hits"`
	PageIns     int64 `json:"page_ins"`
	Evictions   int64 `json:"evictions"`
	Refusals    int64 `json:"refusals"`
	PageInBytes int64 `json:"page_in_bytes"`

	BudgetBytes int64 `json:"budget_bytes,omitempty"`
	PeakBytes   int64 `json:"peak_bytes,omitempty"`

	HitRate             float64 `json:"hit_rate"`
	RefusalRate         float64 `json:"refusal_rate"`
	ExpertBytesPerToken float64 `json:"expert_bytes_per_token"`
	PeakBudgetUsed      float64 `json:"peak_budget_used,omitempty"`

	// The most recent request's framing. Shape makes the byte rates readable (3% of experts
	// activated is why an offload budget can be small); placement is the pin-set gauge, which
	// describes one request and does not sum.
	Experts           int     `json:"experts,omitempty"`
	ExpertsPerToken   int     `json:"experts_per_token,omitempty"`
	ActivatedFraction float64 `json:"activated_fraction,omitempty"`
	PlacementBasis    string  `json:"placement_basis,omitempty"`
	PlacementDrift    float64 `json:"placement_drift,omitempty"`
	// PlacementServedShare is the complement of drift: the share of the request's expert touches
	// the resident plan actually served. Named for what it measures rather than as "coverage",
	// which in this repo already means how much of a surface a test exercises.
	PlacementServedShare float64 `json:"placement_served_share,omitempty"`
	SharedRingAgents     int     `json:"shared_ring_agents,omitempty"`
	AgentsPerPageIn      float64 `json:"agents_per_page_in,omitempty"`

	// ReconciliationFailures is the alarm; FailedChecks names what disagreed on the most recent
	// unreconciled request, so an operator who sees a non-zero count is not left guessing which
	// of the numbers above stopped being trustworthy.
	ReconciliationFailures int64    `json:"reconciliation_failures"`
	FailedChecks           []string `json:"failed_checks,omitempty"`
}

type debugRequestMemoryVars struct {
	Backend       string                         `json:"backend"`
	PromptTokens  int                            `json:"prompt_tokens"`
	MaxNewTokens  int                            `json:"max_new_tokens"`
	PlannedTokens int                            `json:"planned_tokens"`
	HeadroomRatio float64                        `json:"headroom_ratio,omitempty"`
	MemoryPlan    []debugModelLoadMemoryPlanVars `json:"memory_plan,omitempty"`
	Capacities    []debugModelLoadCapacityVars   `json:"capacities,omitempty"`
	Fit           []debugMemoryFitVars           `json:"fit,omitempty"`
}

type debugCompactionVars struct {
	Attempts                    map[string]uint64 `json:"attempts"`
	BailReasons                 map[string]uint64 `json:"bail_reasons"`
	DroppedTurns                uint64            `json:"dropped_turns"`
	ShedTokens                  uint64            `json:"shed_tokens"`
	UncachedTrimResults         uint64            `json:"uncached_trim_results"`
	UncachedTrimShedTokens      uint64            `json:"uncached_trim_shed_tokens"`
	CacheReadTokens             uint64            `json:"cache_read_tokens"`
	LastPostFireCacheReadTokens float64           `json:"last_post_fire_cache_read_tokens"`
	// AnchorStarved was previously rendered ONLY on the Prometheus surface
	// (fak_gateway_compaction_anchor_starved_total), so the JSON front door every operator tool
	// actually polls could not tell the #1407 pathology from a benign short session. Same counter,
	// same meaning — a subset of BailReasons["under_budget"].
	AnchorStarved uint64 `json:"anchor_starved"`
	// SolvencyForced is the fired-at-a-loss subset (Config.CompactSolvencyFloorTokens overrode the
	// burst economics). Surfaced here so a fire count can be read net of survival buys.
	SolvencyForced uint64 `json:"solvency_forced"`
	// Budget is the CONFIGURED compaction line (Config.CompactHistoryBudget) this gateway resolved.
	// `fak guard` overrides the flag default for every launch (resolveGuardCompactBudget), so the
	// number in --help is NOT reliably the number in force; this is the one actually being compared.
	Budget int `json:"budget"`
	// LastSuffixTokens / PeakSuffixTokens are the compactible messages[] span the most recent and
	// the largest-so-far bail measured against Budget. Headroom = Budget - LastSuffixTokens. They
	// answer the question a bail-reason tally cannot: whether "under_budget" means this session is
	// short, or means the line sits above the span this traffic ever reaches.
	LastSuffixTokens uint64 `json:"last_suffix_tokens"`
	PeakSuffixTokens uint64 `json:"peak_suffix_tokens"`
}

type debugHTTPMetricVars struct {
	Route   string           `json:"route"`
	Method  string           `json:"method"`
	Status  string           `json:"status"`
	Latency debugLatencyVars `json:"latency"`
}

type debugOperationMetricVars struct {
	Operation   string           `json:"operation"`
	Verdict     string           `json:"verdict"`
	Reason      string           `json:"reason"`
	Disposition string           `json:"disposition"`
	By          string           `json:"by"` // which adjudicator decided (forensics)
	Latency     debugLatencyVars `json:"latency"`
}

type debugInKernelOOMVars struct {
	Class           string `json:"class"`
	Count           uint64 `json:"count"`
	FailedBytes     uint64 `json:"failed_bytes"`
	LastFailedBytes uint64 `json:"last_failed_bytes"`
	LastSite        string `json:"last_site,omitempty"`
}

type debugRequestMemoryMetricVars struct {
	Backend        string `json:"backend"`
	Class          string `json:"class"`
	Scope          string `json:"scope"`
	DType          string `json:"dtype"`
	Observations   uint64 `json:"observations"`
	TotalBytes     uint64 `json:"total_bytes"`
	HighWaterBytes int64  `json:"high_water_bytes"`
}

type debugRequestMemoryFitVars struct {
	Backend          string `json:"backend"`
	Scope            string `json:"scope"`
	Observations     uint64 `json:"observations"`
	WantHighWater    int64  `json:"want_high_water_bytes"`
	MarginLowWater   int64  `json:"margin_low_water_bytes,omitempty"`
	MarginLowWaterOK bool   `json:"margin_low_water_known"`
}

type debugRequestMemoryTokenVars struct {
	Backend      string `json:"backend"`
	Kind         string `json:"kind"`
	Observations uint64 `json:"observations"`
	Total        uint64 `json:"total"`
	HighWater    int    `json:"high_water"`
}

type debugInKernelOOMRetryVars struct {
	Backend         string `json:"backend"`
	Class           string `json:"class"`
	Attempts        uint64 `json:"attempts"`
	Successes       uint64 `json:"successes"`
	Failures        uint64 `json:"failures"`
	LastFailedBytes uint64 `json:"last_failed_bytes"`
	LastSite        string `json:"last_site,omitempty"`
}

type debugInKernelPressureTrimVars struct {
	Backend         string `json:"backend"`
	Scope           string `json:"scope"`
	Class           string `json:"class"`
	Reason          string `json:"reason"`
	Attempts        uint64 `json:"attempts"`
	Trimmed         uint64 `json:"trimmed"`
	NoHooks         uint64 `json:"no_hooks"`
	Resolved        uint64 `json:"resolved"`
	LastWantBytes   uint64 `json:"last_want_bytes"`
	LastBudgetBytes uint64 `json:"last_budget_bytes"`
	LastMarginBytes int64  `json:"last_margin_bytes"`
}

type debugLatencyVars struct {
	Count      uint64            `json:"count"`
	SumSeconds float64           `json:"sum_seconds"`
	Buckets    []debugBucketVars `json:"buckets"`
}

type debugBucketVars struct {
	LESeconds float64 `json:"le_seconds"`
	Count     uint64  `json:"count"`
}

func (s *Server) handleDebugVars(w http.ResponseWriter, r *http.Request) {
	if !requireMethod(w, r, http.MethodGet) {
		return
	}
	writeJSON(w, http.StatusOK, s.debugVarsContext(r.Context(), time.Now()))
}

func (s *Server) debugVars(now time.Time) debugVarsResponse {
	return s.debugVarsContext(context.Background(), now)
}

// debugWatchdogVars folds the process-global resume/heal watchdog expvars into the /debug/vars
// response, or nil on a cold process that has recorded no watchdog signal — the same
// nil-when-empty convention the other optional blocks keep, so the pane omits the block rather
// than rendering an all-zero one (#3803).
func debugWatchdogVars() *resumemetrics.Snapshot {
	if !resumemetrics.Active() {
		return nil
	}
	snap := resumemetrics.Read()
	return &snap
}

func (s *Server) debugVarsContext(ctx context.Context, now time.Time) debugVarsResponse {
	m := s.metrics
	if m == nil {
		m = newGatewayMetrics(now)
	}
	start := m.start
	if start.IsZero() {
		start = now
	}
	uptime := now.Sub(start).Seconds()
	if uptime < 0 {
		uptime = 0
	}

	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)

	c := s.k.Counters()
	ratio := 0.0
	if c.Submits > 0 {
		ratio = float64(c.VDSOHits) / float64(c.Submits)
	}
	httpRows, opRows := m.snapshot()
	compact := m.compactionSnapshotData()
	oomRows := m.inKernelOOMSnapshotData()
	reqMemoryRows := m.requestMemoryAggregateSnapshotData()
	infer := m.inferenceSnapshotData()
	vcacheTurns, vcacheCapped := m.vcacheTurnsSnapshot()
	_, inflightMaxAge := m.inflightSnapshot(now)

	upstream := m.debugUpstreamVars()
	upstream.ProviderExtraBodySet, upstream.ProviderExtraBodyKeys = debugProviderExtraBody(s.planner)
	modelLoad := debugModelLoadProfile(s.modelLoadProfile())
	startupReport := s.startupReportText()
	adjudication := m.adjudicationSummary()
	observation := s.buildObservationSnapshot(ctx, now, adjudication, c.VDSOHits, m.servedInlineSnapshot())

	return debugVarsResponse{
		Gateway: debugGatewayVars{
			Up:               true,
			Version:          s.version,
			Engine:           s.engineID,
			Model:            s.model,
			VDSO:             s.k.VDSOEnabled(),
			AuthRequired:     s.requireKey != "",
			StartTimeUnix:    start.Unix(),
			UptimeSeconds:    uptime,
			InflightRequests: atomic.LoadInt64(&m.inflight),
		},
		Runtime: debugRuntimeVars{
			GoVersion:    runtime.Version(),
			GOOS:         runtime.GOOS,
			GOARCH:       runtime.GOARCH,
			NumCPU:       runtime.NumCPU(),
			GOMAXPROCS:   runtime.GOMAXPROCS(0),
			NumGoroutine: runtime.NumGoroutine(),
			Memory:       newDebugMemoryVars(&mem),
		},
		Kernel: debugKernelVars{
			Submits:      c.Submits,
			VDSOHits:     c.VDSOHits,
			EngineCalls:  c.EngineCalls,
			Denies:       c.Denies,
			Transforms:   c.Transforms,
			Quarantines:  c.Quarantines,
			ResultDenies: c.ResultDenies,
			Admitted:     c.Admitted,
			VDSOHitRatio: ratio,
		},
		Inference:        inferenceVarsFromSnapshot(infer, inflightMaxAge),
		Upstream:         upstream,
		VCache:           vcacheVarsFromSnapshot(infer),
		CacheAttribution: observation.CacheAttribution,
		ManagedCache:     observation.ManagedCache,
		TokenSavings:     s.tokenSavingsVars(adjudication),
		ShrinkLevers: shrinkLeverVars(s.wireRunsShrinkLevers(), s.dualRoutesLocalModels(), s.provider,
			s.compactHistoryBudget, s.elideStaleReads, s.deferColdTools),
		VCacheFamilies:   vcacheFamiliesVars(vcacheTurns, vcacheCapped),
		VCacheGovernor:   m.vcacheGovernorDecisionRecords(),
		VCacheGovQuality: m.vcacheGovernorQualityVars(),
		VCacheWarmth:     m.vcacheWarmthDemotionRecords(),
		ModelLoad:        modelLoad,
		Startup:          debugStartupProfile(s.startup.snapshot(), startupReport, modelLoad),
		KVMemory:         debugKVMemory(s.planner),
		RequestMemory:    debugRequestMemory(s.planner),
		MoEResidency:     debugMoEResidency(s.planner),
		Sessions:         observation.Sessions,
		Assumptions:      s.debugAssumptions(ctx),
		ContextQueries:   s.contextQueryAuditSnapshot(),
		Endpoints:        s.debugEndpoints(),
		Adjudication:     s.debugAdjudication(),
		Harness:          observation.Harness,
		Fleet:            s.debugFleet(),
		StartupReport:    startupReport,
		Watchdog:         debugWatchdogVars(),
		Metrics: debugMetricsVars{
			HTTP:       debugHTTPRows(httpRows),
			Operations: debugOperationRows(opRows),
			Compaction: debugCompactionVars{
				Attempts:                    debugStableCompactionAttempts(compact.attempts),
				BailReasons:                 compact.bailReasons,
				DroppedTurns:                compact.dropped,
				ShedTokens:                  compact.shed,
				UncachedTrimResults:         compact.uncachedTrimResults,
				UncachedTrimShedTokens:      compact.uncachedTrimShed,
				CacheReadTokens:             compact.cacheReads,
				LastPostFireCacheReadTokens: compact.lastCacheRd,
				AnchorStarved:               compact.anchorStarved,
				SolvencyForced:              compact.solvencyForced,
				Budget:                      s.compactHistoryBudget,
				LastSuffixTokens:            compact.lastSuffixTokens,
				PeakSuffixTokens:            compact.peakSuffixTokens,
			},
			RequestMemory:        debugRequestMemoryMetricRows(reqMemoryRows.plans),
			RequestMemoryFit:     debugRequestMemoryFitRows(reqMemoryRows.fits),
			RequestMemoryTokens:  debugRequestMemoryTokenRows(reqMemoryRows.tokens),
			InKernelOOM:          debugInKernelOOMRows(oomRows),
			InKernelOOMRetries:   debugInKernelOOMRetryRows(s.planner),
			InKernelPressureTrim: debugInKernelPressureTrimRows(s.planner),
		},
	}
}

func debugProviderExtraBody(planner agent.Planner) (bool, []string) {
	hp := unwrapHTTPPlanner(planner)
	if hp == nil || len(hp.ExtraBody) == 0 {
		return false, nil
	}
	obj := map[string]json.RawMessage{}
	if err := json.Unmarshal(hp.ExtraBody, &obj); err != nil {
		return true, nil
	}
	keys := make([]string, 0, len(obj))
	for k := range obj {
		if strings.TrimSpace(k) != "" {
			keys = append(keys, k)
		}
	}
	sort.Strings(keys)
	return true, keys
}

// debugSessionVars is the per-session drive-state row /debug/vars mirrors for live
// operator panes (the `fak info` agents sub-pane): which sessions are running through
// this gateway RIGHT NOW, with the resource axes the session registry already tracks.
//
// ParentTrace/Generation carry CONTINUATION lineage, not spawn lineage: the registry
// writes them from session.Table.Recontinue alone, so a row with a parent is the same
// agent re-continued under a fresh trace after a budget/context reset, and Generation
// counts those resets. The sub-agent axis is SpawnCount, from the activity registry
// below — a parent-side count of admitted spawn calls. Reading a parent as a child
// reported sub-agents nobody spawned, at a depth that was a reset tally.
//
// The budget fields are the REMAINING allotment (what the operator seeded minus what the
// session consumed), and ElapsedSeconds is live wall-clock. Everything here is a
// projection of SessionState — no payloads, no transcript text — so the pane stays
// redaction-safe by construction.
// The wire shape lives in internal/guardvars so the `fak info` pane decodes the exact block
// this producer emits — one definition, no field-for-field hand-sync to drift (see guardvars).
type debugSessionVars = guardvars.SessionVars

// debugSessions folds the live session registry into /debug/vars rows. Stopped sessions
// are dropped (matching debugAssumptions: the pane shows what is running, not history),
// and rows keep the registry's own order so the main agent — registered first — leads.
// Each live row is joined against the per-trace activity registry (#2627) for its
// last-tool / spawn-count / in-flight-or-idle age, projected relative to now; a trace with
// no activity contributes no extra fields, so a pre-activity row's wire shape is unchanged.
func (s *Server) debugSessions(ctx context.Context, now time.Time) []debugSessionVars {
	if s.listSessions == nil {
		return nil
	}
	sessions := s.listSessions(ctx)
	live := make(map[string]struct{}, len(sessions))
	var out []debugSessionVars
	for _, st := range sessions {
		if strings.EqualFold(strings.TrimSpace(st.Run), "stopped") {
			continue
		}
		trace := strings.TrimSpace(st.TraceID)
		live[trace] = struct{}{}
		row := debugSessionVars{
			TraceID:           trace,
			Run:               strings.TrimSpace(st.Run),
			ParentTrace:       strings.TrimSpace(st.ParentTrace),
			Generation:        st.Generation,
			Priority:          st.Priority,
			TurnsLeft:         st.Budget.TurnsLeft,
			TokensLeft:        st.Budget.TokensLeft,
			ContextTokensLeft: st.Budget.ContextTokensLeft,
			ElapsedSeconds:    st.Time.ElapsedSeconds,
			Assumptions:       len(st.Assumptions),
		}
		if act, ok := s.activity.snapshot(trace, now); ok {
			row.LastTool = act.LastTool
			row.SpawnCount = act.SpawnCount
			row.InflightSeconds = act.InflightSeconds
			row.IdleSeconds = act.IdleSeconds
		}
		out = append(out, row)
	}
	// Fold stopped/vanished traces so the activity registry tracks at most the live
	// sessions — the read-path half of the bounded lifecycle (the write path caps it).
	s.activity.retain(live)
	return out
}

// debugEndpoints projects the host-supplied accounts+nodes provider into the
// /debug/vars "endpoints" block, or nil when no provider is set / nothing to report
// (the block is then omitted). See session_endpoints.go.
func (s *Server) debugEndpoints() *SessionEndpoints {
	ep, ok := s.sessionEndpoints()
	if !ok {
		return nil
	}
	return &ep
}

// debugAdjudication folds the live operation ledger into the verdict roll-up the guard
// exit summary also prints (via Server.AdjudicationSummary, which attaches the
// compaction budget), or nil on a cold gateway that has decided nothing and observed no
// tokens — so a fresh session omits the block rather than emitting an all-zero tally.
func (s *Server) debugAdjudication() *AdjudicationSummary {
	sum := s.AdjudicationSummary()
	if sum.Total == 0 && sum.CachedPromptTokens == 0 && sum.InputTokens == 0 &&
		sum.OutputTokens == 0 && sum.CompactionBudget == 0 {
		return nil
	}
	return &sum
}

// debugFleet projects the host-supplied cross-machine fleet provider into the
// /debug/vars "fleet" block, or nil when no provider is set / the fleet is empty (the
// block is then omitted). See session_fleet.go.
func (s *Server) debugFleet() *SessionFleet {
	f, ok := s.sessionFleet()
	if !ok {
		return nil
	}
	return &f
}

func (s *Server) debugAssumptions(ctx context.Context) []SessionAssumption {
	if s.listSessions == nil {
		return nil
	}
	sessions := s.listSessions(ctx)
	var out []SessionAssumption
	for _, st := range sessions {
		if strings.EqualFold(strings.TrimSpace(st.Run), "stopped") {
			continue
		}
		for _, a := range st.Assumptions {
			if strings.TrimSpace(a.TraceID) == "" {
				a.TraceID = strings.TrimSpace(st.TraceID)
			}
			out = append(out, a)
		}
	}
	return out
}

func (m *gatewayMetrics) debugUpstreamVars() debugUpstreamVars {
	out := debugUpstreamVars{
		ErrorsByKind: map[string]uint64{},
		AuthRefreshByOutcome: map[string]uint64{
			"recovered": 0,
			"exhausted": 0,
		},
		ForbiddenRetryByOutcome: map[string]uint64{
			"recovered": 0,
			"exhausted": 0,
		},
		AccountFailoverByOutcome: map[string]uint64{
			"recovered": 0,
			"exhausted": 0,
		},
	}
	if m == nil {
		return out
	}
	m.upstreamErrMu.Lock()
	for k, v := range m.upstreamErrors {
		out.ErrorsByKind[k] = v
	}
	for k, v := range m.upstreamAuthRefreshes {
		out.AuthRefreshByOutcome[k] = v
	}
	for k, v := range m.upstreamForbiddenRetries {
		out.ForbiddenRetryByOutcome[k] = v
	}
	for k, v := range m.upstreamAccountFailovers {
		out.AccountFailoverByOutcome[k] = v
	}
	out.LastForbiddenDetail = m.lastForbiddenDetail
	m.upstreamErrMu.Unlock()
	out.Retries = atomic.LoadUint64(&m.upstreamRetries)
	out.RetryWaitSeconds = time.Duration(atomic.LoadUint64(&m.upstreamRetryWaitNS)).Seconds()
	return out
}

// inferenceVarsFromSnapshot derives the /debug/vars inference block from the same
// snapshot the Prometheus renderer (writeInferenceMetrics) reads, so the two surfaces
// can never disagree. Every rate here uses the identical numerator/denominator the
// metric line uses: output t/s = completion/total wall-clock; prefill t/s =
// prefill-prompt-tokens/prefill-seconds over measured turns; decode t/s =
// measured-completion/measured-decode-seconds; mean TTFT = prefill-seconds/ttft-turns.
// A zero denominator yields 0 (no phantom throughput before the first measured turn).
func inferenceVarsFromSnapshot(snap inferenceSnapshot, inflightMaxAge float64) debugInferenceVars {
	var turns uint64
	for _, n := range snap.reqs {
		turns += n
	}
	out := debugInferenceVars{
		Turns:                 turns,
		PromptTokens:          snap.promptTok,
		CompletionTokens:      snap.complTok,
		DurationSeconds:       snap.decodeSecs,
		TTFTTurns:             snap.ttftTurns,
		PrefillSeconds:        snap.prefillSecs,
		InflightMaxAgeSeconds: inflightMaxAge,
	}
	if snap.decodeSecs > 0 {
		out.OutputTokensPerSecond = float64(snap.complTok) / snap.decodeSecs
	}
	if snap.prefillSecs > 0 {
		out.PrefillTokensPerSecond = float64(snap.prefillPromptTok) / snap.prefillSecs
	}
	if snap.measuredDecodeSecs > 0 {
		out.DecodeTokensPerSecond = float64(snap.measuredComplTok) / snap.measuredDecodeSecs
	}
	if snap.ttftTurns > 0 {
		out.MeanTTFTSeconds = snap.prefillSecs / float64(snap.ttftTurns)
	}
	return out
}

func debugRequestMemory(p agent.Planner) *debugRequestMemoryVars {
	reporter, ok := p.(agent.RequestMemoryReporter)
	if !ok {
		return nil
	}
	st := reporter.RequestMemoryStats()
	if !st.Observed {
		return nil
	}
	backend := defaultBackendLabel(st.Backend)
	out := &debugRequestMemoryVars{
		Backend:       backend,
		PromptTokens:  st.PromptTokens,
		MaxNewTokens:  st.MaxNewTokens,
		PlannedTokens: st.PlannedTokens,
		HeadroomRatio: st.HeadroomRatio,
	}
	for _, row := range st.MemoryPlan {
		out.MemoryPlan = appendDebugMemoryPlanVar(out.MemoryPlan, row.Class, row.Scope, row.Bytes, row.Detail, row.DType)
	}
	for _, cap := range st.Capacities {
		out.Capacities = appendDebugCapacityVar(out.Capacities, cap.Scope, cap.TotalBytes, cap.FreeBytes, cap.Known, cap.FreeKnown)
	}
	out.Fit = debugMemoryFitRows(requestMemoryFitRows(st.MemoryPlan, st.Capacities, st.HeadroomRatio))
	return out
}

func debugModelLoadProfile(p *ModelLoadProfile) *debugModelLoadVars {
	if p == nil {
		return nil
	}
	out := &debugModelLoadVars{
		Source:              p.Source,
		Mode:                p.Mode,
		TotalSeconds:        p.TotalSeconds,
		Bytes:               p.Bytes,
		Tensors:             p.Tensors,
		Bottleneck:          p.Bottleneck,
		MemoryHeadroomRatio: p.MemoryHeadroomRatio,
		Messages:            append([]StartupMessage(nil), p.Messages...),
	}
	for _, ph := range p.sorted() {
		out.Phases = append(out.Phases, debugModelLoadPhaseVars{
			Phase:   ph.Phase,
			Seconds: ph.Seconds,
			Bytes:   ph.Bytes,
			Tensors: ph.Tensors,
		})
	}
	for _, path := range p.LoadPaths {
		out.LoadPaths = append(out.LoadPaths, debugModelLoadPathVars{
			QuantType:       path.QuantType,
			Class:           modelLoadPathClass(path.Expert),
			ResidentTensors: path.ResidentTensors,
			ResidentBytes:   path.ResidentBytes,
			DequantTensors:  path.DequantTensors,
			DequantBytes:    path.DequantBytes,
		})
	}
	for _, row := range p.MemoryPlan {
		out.MemoryPlan = appendDebugMemoryPlanVar(out.MemoryPlan, row.Class, row.Scope, row.Bytes, row.Detail, row.DType)
	}
	for _, cap := range p.sortedMemoryCapacities() {
		out.MemoryCapacities = appendDebugCapacityVar(out.MemoryCapacities, cap.Scope, cap.TotalBytes, cap.FreeBytes, cap.Known, cap.FreeKnown)
	}
	out.MemoryFit = debugMemoryFitRows(modelLoadMemoryFitRows(p.MemoryPlan, p.MemoryCapacities, p.MemoryHeadroomRatio))
	return out
}

func debugStartupProfile(s startupSnapshot, report string, modelLoad *debugModelLoadVars) debugStartupVars {
	status := "starting"
	if !s.ready.IsZero() {
		status = "ready"
	}
	out := debugStartupVars{
		Status:             status,
		TimeToReadySeconds: s.timeToReady(),
		UnaccountedSeconds: s.unaccountedSeconds(),
		ModelLoad:          modelLoad,
		Messages:           append([]StartupMessage(nil), s.messages...),
	}
	if !s.start.IsZero() {
		out.StartedAt = s.start.UTC().Format(time.RFC3339Nano)
	}
	if !s.ready.IsZero() {
		out.ReadyAt = s.ready.UTC().Format(time.RFC3339Nano)
	}
	for _, ph := range s.phases {
		provenance := ph.Provenance
		if provenance == "" {
			provenance = "measured"
		}
		stage := "gateway-boot"
		if ph.PostReady {
			stage = "post-ready"
		}
		out.Phases = append(out.Phases, debugStartupPhaseVars{
			Name:       ph.Name,
			Seconds:    ph.Dur.Seconds(),
			Provenance: provenance,
			Stage:      stage,
		})
	}
	if modelLoad != nil {
		out.Messages = append(out.Messages, modelLoad.Messages...)
	}
	if strings.TrimSpace(report) != "" {
		out.Messages = append(out.Messages, StartupMessage{
			Source: "guard",
			Kind:   "startup-report",
			Level:  "info",
			Text:   report,
		})
	}
	return out
}

// appendDebugMemoryPlanVar folds one memory-plan demand row (from either the
// request-memory or model-load reporter, which carry structurally identical rows)
// into the shared debug var shape, dropping zero/negative-byte rows. Single source
// of the class/scope/dtype label-mapping the request and model-load paths shared.
func appendDebugMemoryPlanVar(out []debugModelLoadMemoryPlanVars, class, scope string, bytes int64, detail, dtype string) []debugModelLoadMemoryPlanVars {
	if bytes <= 0 {
		return out
	}
	return append(out, debugModelLoadMemoryPlanVars{
		Class:  modelLoadClass(class),
		Scope:  modelLoadScope(scope),
		Bytes:  bytes,
		Detail: detail,
		DType:  modelLoadDType(dtype),
	})
}

// appendDebugCapacityVar folds one memory-capacity row into the shared debug var
// shape. FreeKnown is gated on Known to match the prior inline behavior.
func appendDebugCapacityVar(out []debugModelLoadCapacityVars, scope string, totalBytes, freeBytes int64, known, freeKnown bool) []debugModelLoadCapacityVars {
	return append(out, debugModelLoadCapacityVars{
		Scope:      modelLoadScope(scope),
		TotalBytes: totalBytes,
		FreeBytes:  freeBytes,
		Known:      known,
		FreeKnown:  known && freeKnown,
	})
}

func debugMemoryFitRows(rows []memoryFitRow) []debugMemoryFitVars {
	if len(rows) == 0 {
		return nil
	}
	out := make([]debugMemoryFitVars, 0, len(rows))
	for _, row := range rows {
		out = append(out, debugMemoryFitVars{
			Scope:         row.Scope,
			WantBytes:     row.WantBytes,
			BudgetBytes:   row.BudgetBytes,
			MarginBytes:   row.MarginBytes,
			CapacityKnown: row.CapacityKnown,
			FreeKnown:     row.FreeKnown,
		})
	}
	return out
}

func debugStableCompactionAttempts(in map[string]uint64) map[string]uint64 {
	out := map[string]uint64{}
	for _, outcome := range []string{"fired", "bailed", "off"} {
		out[outcome] = in[outcome]
	}
	return out
}

func debugHTTPRows(rows []httpMetricSnapshot) []debugHTTPMetricVars {
	out := make([]debugHTTPMetricVars, 0, len(rows))
	for _, row := range rows {
		out = append(out, debugHTTPMetricVars{
			Route:   row.key.route,
			Method:  row.key.method,
			Status:  row.key.status,
			Latency: debugLatency(row.val),
		})
	}
	return out
}

func debugOperationRows(rows []operationMetricSnapshot) []debugOperationMetricVars {
	out := make([]debugOperationMetricVars, 0, len(rows))
	for _, row := range rows {
		out = append(out, debugOperationMetricVars{
			Operation:   row.key.operation,
			Verdict:     row.key.verdict,
			Reason:      row.key.reason,
			Disposition: row.key.disposition,
			By:          row.key.by,
			Latency:     debugLatency(row.val),
		})
	}
	return out
}

// debugMoEResidency renders a serve's activated-expert residency, or nil when there is nothing to
// render — a proxy planner (no reporter) or a local one whose operator declared no expert budget,
// so no session ever built a ring. Both are ordinary configurations, and reporting them as a block
// of zeros would claim a measurement nobody took.
func debugMoEResidency(p agent.Planner) *debugMoEResidencyVars {
	reporter, ok := p.(agent.MoEResidencyReporter)
	if !ok {
		return nil
	}
	l := reporter.MoEResidencyStats()
	if l.Requests == 0 {
		return nil
	}
	last := l.Last
	out := &debugMoEResidencyVars{
		Requests:               l.Requests,
		Tokens:                 l.Tokens,
		Lookups:                l.Lookups,
		Hits:                   l.Hits,
		PageIns:                l.PageIns,
		Evictions:              l.Evictions,
		Refusals:               l.Refusals,
		PageInBytes:            l.PageInBytes,
		BudgetBytes:            l.BudgetBytes,
		PeakBytes:              l.PeakBytes,
		HitRate:                l.HitRate(),
		RefusalRate:            l.RefusalRate(),
		ExpertBytesPerToken:    l.ExpertBytesPerToken(),
		PeakBudgetUsed:         l.PeakBudgetUsed(),
		Experts:                last.Shape.Experts,
		ExpertsPerToken:        last.Shape.ExpertsPerToken,
		ActivatedFraction:      last.Shape.ActivatedFraction,
		PlacementDrift:         last.Placement.Drift,
		PlacementServedShare:   last.Placement.Coverage,
		AgentsPerPageIn:        last.Rates.AgentsPerPageIn,
		ReconciliationFailures: l.ReconciliationFailures,
	}
	if basis := strings.TrimSpace(last.Placement.Basis); basis != "" && basis != "none" {
		out.PlacementBasis = basis
	}
	if last.Shared != nil {
		out.SharedRingAgents = last.Shared.Agents
	}
	for _, c := range last.Reconciliation.Checks {
		if !c.OK {
			out.FailedChecks = append(out.FailedChecks, c.Name)
		}
	}
	return out
}

func debugKVMemory(p agent.Planner) *debugKVMemoryVars {
	reporter, ok := p.(agent.KVMemoryReporter)
	if !ok {
		return nil
	}
	st := reporter.KVMemoryStats()
	class := strings.TrimSpace(st.MemoryClass)
	if class == "" {
		class = "kv_cache"
	}
	scope := strings.TrimSpace(st.Scope)
	if scope == "" {
		scope = "host"
	}
	backend := defaultBackendLabel(st.Backend)
	dtype := modelLoadDType(st.DType)
	return &debugKVMemoryVars{
		Enabled:               st.Enabled,
		Backend:               backend,
		MemoryClass:           class,
		Scope:                 scope,
		DType:                 dtype,
		BytesPerToken:         st.BytesPerToken,
		ResidentTokens:        st.ResidentTokens,
		ResidentBytes:         st.ResidentBytes,
		CapacityKnown:         st.CapacityKnown,
		CapacityFreeKnown:     st.CapacityKnown && st.CapacityFreeKnown,
		CapacityTotalBytes:    st.CapacityTotalBytes,
		CapacityFreeBytes:     st.CapacityFreeBytes,
		HeadroomRatio:         st.HeadroomRatio,
		FitBudgetBytes:        st.FitBudgetBytes,
		FitMarginBytes:        st.FitMarginBytes,
		BudgetTokens:          st.BudgetTokens,
		LRUTokens:             st.LRUTokens,
		MaxDepthTokens:        st.MaxDepthTokens,
		Nodes:                 st.Nodes,
		Leaves:                st.Leaves,
		Evictions:             st.Evictions,
		PolicyEvictions:       st.PolicyEvictions,
		Splits:                st.Splits,
		L1DeviceResidentBytes: st.L1DeviceResidentBytes,
		L1HostResidentBytes:   st.L1HostResidentBytes,
		L2HostResidentBytes:   st.L2HostResidentBytes,
		L2HostCapacityBytes:   st.L2HostCapacityBytes,
		L1Hits:                st.L1Hits,
		L1Misses:              st.L1Misses,
		L1Faults:              st.L1Faults,
		L1HitTokens:           st.L1HitTokens,
		L2Hits:                st.L2Hits,
		L2Misses:              st.L2Misses,
		L2Faults:              st.L2Faults,
		L2HitTokens:           st.L2HitTokens,
		L2StageBytes:          st.L2StageBytes,
		L2RestoreBytes:        st.L2RestoreBytes,
		L2Evictions:           st.L2Evictions,
		L3Enabled:             st.L3Enabled,
		L3ReferencedBytes:     st.L3ReferencedBytes,
		L3Hits:                st.L3Hits,
		L3Misses:              st.L3Misses,
		L3Faults:              st.L3Faults,
		L3HitTokens:           st.L3HitTokens,
		L3StageBytes:          st.L3StageBytes,
		L3RestoreBytes:        st.L3RestoreBytes,
		L3StageNanos:          st.L3StageNanos,
		L3RestoreNanos:        st.L3RestoreNanos,
		L3StageFaults:         st.L3StageFaults,
		L3RestoreFaults:       st.L3RestoreFaults,
	}
}

func debugInKernelOOMRows(rows []inKernelOOMSnapshot) []debugInKernelOOMVars {
	out := make([]debugInKernelOOMVars, 0, len(rows))
	for _, row := range rows {
		out = append(out, debugInKernelOOMVars{
			Class:           row.class,
			Count:           row.count,
			FailedBytes:     row.failedBytes,
			LastFailedBytes: row.lastFailedBytes,
			LastSite:        row.lastSite,
		})
	}
	return out
}

func debugRequestMemoryMetricRows(rows []requestMemoryPlanSnapshot) []debugRequestMemoryMetricVars {
	if len(rows) == 0 {
		return nil
	}
	out := make([]debugRequestMemoryMetricVars, 0, len(rows))
	for _, row := range rows {
		out = append(out, debugRequestMemoryMetricVars{
			Backend:        row.key.backend,
			Class:          row.key.class,
			Scope:          row.key.scope,
			DType:          row.key.dtype,
			Observations:   row.observations,
			TotalBytes:     row.totalBytes,
			HighWaterBytes: row.highWaterBytes,
		})
	}
	return out
}

func debugRequestMemoryFitRows(rows []requestMemoryFitSnapshot) []debugRequestMemoryFitVars {
	if len(rows) == 0 {
		return nil
	}
	out := make([]debugRequestMemoryFitVars, 0, len(rows))
	for _, row := range rows {
		out = append(out, debugRequestMemoryFitVars{
			Backend:          row.key.backend,
			Scope:            row.key.scope,
			Observations:     row.observations,
			WantHighWater:    row.wantHighWater,
			MarginLowWater:   row.marginLowWater,
			MarginLowWaterOK: row.marginKnown,
		})
	}
	return out
}

func debugRequestMemoryTokenRows(rows []requestMemoryTokenSnapshot) []debugRequestMemoryTokenVars {
	if len(rows) == 0 {
		return nil
	}
	out := make([]debugRequestMemoryTokenVars, 0, len(rows))
	for _, row := range rows {
		out = append(out, debugRequestMemoryTokenVars{
			Backend:      row.key.backend,
			Kind:         row.key.kind,
			Observations: row.observations,
			Total:        row.total,
			HighWater:    row.highWater,
		})
	}
	return out
}

func debugInKernelOOMRetryRows(p agent.Planner) []debugInKernelOOMRetryVars {
	reporter, ok := p.(agent.InKernelOOMRetryReporter)
	if !ok {
		return nil
	}
	st := reporter.InKernelOOMRetryStats()
	if len(st.Rows) == 0 {
		return nil
	}
	backend := defaultBackendLabel(st.Backend)
	out := make([]debugInKernelOOMRetryVars, 0, len(st.Rows))
	for _, row := range st.Rows {
		out = append(out, debugInKernelOOMRetryVars{
			Backend:         backend,
			Class:           oomClassLabel(row.Class),
			Attempts:        row.Attempts,
			Successes:       row.Successes,
			Failures:        row.Failures,
			LastFailedBytes: row.LastFailedBytes,
			LastSite:        row.LastSite,
		})
	}
	return out
}

func debugInKernelPressureTrimRows(p agent.Planner) []debugInKernelPressureTrimVars {
	reporter, ok := p.(agent.InKernelMemoryPressureTrimReporter)
	if !ok {
		return nil
	}
	st := reporter.InKernelMemoryPressureTrimStats()
	if len(st.Rows) == 0 {
		return nil
	}
	backend := defaultBackendLabel(st.Backend)
	out := make([]debugInKernelPressureTrimVars, 0, len(st.Rows))
	for _, row := range st.Rows {
		out = append(out, debugInKernelPressureTrimVars{
			Backend:         backend,
			Scope:           modelLoadScope(row.Scope),
			Class:           oomClassLabel(row.Class),
			Reason:          pressureTrimReasonLabel(row.Reason),
			Attempts:        row.Attempts,
			Trimmed:         row.Trimmed,
			NoHooks:         row.NoHooks,
			Resolved:        row.Resolved,
			LastWantBytes:   row.LastWantBytes,
			LastBudgetBytes: row.LastBudgetBytes,
			LastMarginBytes: row.LastMarginBytes,
		})
	}
	return out
}

func debugLatency(s latencySnapshot) debugLatencyVars {
	buckets := make([]debugBucketVars, 0, len(gatewayLatencyBuckets))
	for i, le := range gatewayLatencyBuckets {
		buckets = append(buckets, debugBucketVars{LESeconds: le, Count: s.buckets[i]})
	}
	return debugLatencyVars{Count: s.count, SumSeconds: s.sum, Buckets: buckets}
}
