package gateway

import (
	"net/http"
	"runtime"
	"strings"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

type debugVarsResponse struct {
	Gateway        debugGatewayVars         `json:"gateway"`
	Runtime        debugRuntimeVars         `json:"runtime"`
	Kernel         debugKernelVars          `json:"kernel"`
	Inference      debugInferenceVars       `json:"inference"`
	VCache         *debugVCacheVars         `json:"vcache,omitempty"`
	VCacheFamilies *debugVCacheFamiliesVars `json:"vcache_families,omitempty"`
	ModelLoad      *debugModelLoadVars      `json:"model_load,omitempty"`
	KVMemory       *debugKVMemoryVars       `json:"kv_memory,omitempty"`
	RequestMemory  *debugRequestMemoryVars  `json:"request_memory,omitempty"`
	Metrics        debugMetricsVars         `json:"metrics"`
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
	Enabled            bool    `json:"enabled"`
	Backend            string  `json:"backend"`
	MemoryClass        string  `json:"memory_class"`
	Scope              string  `json:"scope"`
	DType              string  `json:"dtype,omitempty"`
	BytesPerToken      int64   `json:"bytes_per_token"`
	ResidentTokens     int     `json:"resident_tokens,omitempty"`
	ResidentBytes      int64   `json:"resident_bytes,omitempty"`
	CapacityKnown      bool    `json:"capacity_known"`
	CapacityFreeKnown  bool    `json:"capacity_free_known"`
	CapacityTotalBytes int64   `json:"capacity_total_bytes,omitempty"`
	CapacityFreeBytes  int64   `json:"capacity_free_bytes,omitempty"`
	HeadroomRatio      float64 `json:"headroom_ratio,omitempty"`
	FitBudgetBytes     int64   `json:"fit_budget_bytes,omitempty"`
	FitMarginBytes     int64   `json:"fit_margin_bytes,omitempty"`
	BudgetTokens       int     `json:"budget_tokens,omitempty"`
	LRUTokens          int     `json:"lru_tokens,omitempty"`
	MaxDepthTokens     int     `json:"max_depth_tokens,omitempty"`
	Nodes              int     `json:"nodes,omitempty"`
	Leaves             int     `json:"leaves,omitempty"`
	Evictions          int     `json:"evictions,omitempty"`
	PolicyEvictions    int     `json:"policy_evictions,omitempty"`
	Splits             int     `json:"splits,omitempty"`
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
	CacheReadTokens             uint64            `json:"cache_read_tokens"`
	LastPostFireCacheReadTokens float64           `json:"last_post_fire_cache_read_tokens"`
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
	writeJSON(w, http.StatusOK, s.debugVars(time.Now()))
}

func (s *Server) debugVars(now time.Time) debugVarsResponse {
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
			Memory: debugMemoryVars{
				AllocBytes:      mem.Alloc,
				TotalAllocBytes: mem.TotalAlloc,
				SysBytes:        mem.Sys,
				HeapAllocBytes:  mem.HeapAlloc,
				HeapSysBytes:    mem.HeapSys,
				HeapObjects:     mem.HeapObjects,
				StackInuseBytes: mem.StackInuse,
				NextGCBytes:     mem.NextGC,
				LastGCUnixNano:  mem.LastGC,
				NumGC:           mem.NumGC,
			},
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
		Inference:      inferenceVarsFromSnapshot(infer, inflightMaxAge),
		VCache:         vcacheVarsFromSnapshot(infer),
		VCacheFamilies: vcacheFamiliesVars(vcacheTurns, vcacheCapped),
		ModelLoad:      debugModelLoadProfile(s.modelLoadProfile()),
		KVMemory:       debugKVMemory(s.planner),
		RequestMemory:  debugRequestMemory(s.planner),
		Metrics: debugMetricsVars{
			HTTP:       debugHTTPRows(httpRows),
			Operations: debugOperationRows(opRows),
			Compaction: debugCompactionVars{
				Attempts:                    debugStableCompactionAttempts(compact.attempts),
				BailReasons:                 compact.bailReasons,
				DroppedTurns:                compact.dropped,
				ShedTokens:                  compact.shed,
				CacheReadTokens:             compact.cacheReads,
				LastPostFireCacheReadTokens: compact.lastCacheRd,
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

// debugVCacheVars surfaces the NET realized provider-cache economics (read rebate MINUS
// write premium) the same way `fak vcache observe` does — computed via
// vcachegov.ProveTelemetrySavings over the session's cumulative cache counters, so the
// /debug/vars block, the fak_vcache_* metrics, and the offline observe Aggregate all
// agree on the same totals. Every value is OBSERVED (provider-relayed); a hit is a
// realized rebate, never local trust. Nil until a turn carries provider cache activity.
type debugVCacheVars struct {
	CacheReadTokens     uint64  `json:"cache_read_tokens"`     // OBSERVED read axis
	CacheCreationTokens uint64  `json:"cache_creation_tokens"` // OBSERVED write axis
	InputTokens         uint64  `json:"input_tokens"`          // OBSERVED uncached remainder
	BaselineTokenEquiv  float64 `json:"baseline_token_equiv"`
	ActualTokenEquiv    float64 `json:"actual_token_equiv"`
	SavedTokenEquiv     float64 `json:"saved_token_equiv"` // NET; negative until reads repay writes
	SavedPct            float64 `json:"saved_pct"`
	HitRate             float64 `json:"hit_rate"`
	Multiplier          float64 `json:"multiplier"`
	Status              string  `json:"status"` // PROVEN / REFUTED
}

// vcacheVarsFromSnapshot builds the /debug/vars vcache block from the same inference
// snapshot the Prometheus writeVCacheMetrics reads, so the two surfaces never disagree.
// It returns nil (the block is omitted) until a turn carried provider cache activity.
func vcacheVarsFromSnapshot(snap inferenceSnapshot) *debugVCacheVars {
	if snap.cachedTok == 0 && snap.cacheCreateTok == 0 {
		return nil
	}
	proof := vcacheProofFromCounters(snap.promptTok, snap.cachedTok, snap.cacheCreateTok)
	hit, mult := 0.0, 0.0
	if proof.BaselineTokenEquiv > 0 {
		hit = proof.CacheReadTokens / proof.BaselineTokenEquiv
	}
	if proof.ActualTokenEquiv > 0 {
		mult = proof.BaselineTokenEquiv / proof.ActualTokenEquiv
	}
	return &debugVCacheVars{
		CacheReadTokens:     snap.cachedTok,
		CacheCreationTokens: snap.cacheCreateTok,
		InputTokens:         snap.promptTok,
		BaselineTokenEquiv:  proof.BaselineTokenEquiv,
		ActualTokenEquiv:    proof.ActualTokenEquiv,
		SavedTokenEquiv:     proof.SavedTokenEquiv,
		SavedPct:            proof.SavedPct,
		HitRate:             hit,
		Multiplier:          mult,
		Status:              string(proof.Status),
	}
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
	}
	for _, ph := range p.sorted() {
		out.Phases = append(out.Phases, debugModelLoadPhaseVars{
			Phase:   ph.Phase,
			Seconds: ph.Seconds,
			Bytes:   ph.Bytes,
			Tensors: ph.Tensors,
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
		Enabled:            st.Enabled,
		Backend:            backend,
		MemoryClass:        class,
		Scope:              scope,
		DType:              dtype,
		BytesPerToken:      st.BytesPerToken,
		ResidentTokens:     st.ResidentTokens,
		ResidentBytes:      st.ResidentBytes,
		CapacityKnown:      st.CapacityKnown,
		CapacityFreeKnown:  st.CapacityKnown && st.CapacityFreeKnown,
		CapacityTotalBytes: st.CapacityTotalBytes,
		CapacityFreeBytes:  st.CapacityFreeBytes,
		HeadroomRatio:      st.HeadroomRatio,
		FitBudgetBytes:     st.FitBudgetBytes,
		FitMarginBytes:     st.FitMarginBytes,
		BudgetTokens:       st.BudgetTokens,
		LRUTokens:          st.LRUTokens,
		MaxDepthTokens:     st.MaxDepthTokens,
		Nodes:              st.Nodes,
		Leaves:             st.Leaves,
		Evictions:          st.Evictions,
		PolicyEvictions:    st.PolicyEvictions,
		Splits:             st.Splits,
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
