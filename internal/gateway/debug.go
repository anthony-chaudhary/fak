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
	Gateway       debugGatewayVars        `json:"gateway"`
	Runtime       debugRuntimeVars        `json:"runtime"`
	Kernel        debugKernelVars         `json:"kernel"`
	ModelLoad     *debugModelLoadVars     `json:"model_load,omitempty"`
	KVMemory      *debugKVMemoryVars      `json:"kv_memory,omitempty"`
	RequestMemory *debugRequestMemoryVars `json:"request_memory,omitempty"`
	Metrics       debugMetricsVars        `json:"metrics"`
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
	HTTP        []debugHTTPMetricVars      `json:"http"`
	Operations  []debugOperationMetricVars `json:"operations"`
	Compaction  debugCompactionVars        `json:"compaction"`
	InKernelOOM []debugInKernelOOMVars     `json:"in_kernel_oom"`
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
	if r.Method != http.MethodGet {
		writeErr(w, http.StatusMethodNotAllowed, "use GET")
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
		ModelLoad:     debugModelLoadProfile(s.modelLoadProfile()),
		KVMemory:      debugKVMemory(s.planner),
		RequestMemory: debugRequestMemory(s.planner),
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
			InKernelOOM: debugInKernelOOMRows(oomRows),
		},
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
	backend := strings.TrimSpace(st.Backend)
	if backend == "" {
		backend = "unknown"
	}
	out := &debugRequestMemoryVars{
		Backend:       backend,
		PromptTokens:  st.PromptTokens,
		MaxNewTokens:  st.MaxNewTokens,
		PlannedTokens: st.PlannedTokens,
		HeadroomRatio: st.HeadroomRatio,
	}
	for _, row := range st.MemoryPlan {
		if row.Bytes <= 0 {
			continue
		}
		out.MemoryPlan = append(out.MemoryPlan, debugModelLoadMemoryPlanVars{
			Class:  modelLoadClass(row.Class),
			Scope:  modelLoadScope(row.Scope),
			Bytes:  row.Bytes,
			Detail: row.Detail,
			DType:  modelLoadDType(row.DType),
		})
	}
	for _, cap := range st.Capacities {
		out.Capacities = append(out.Capacities, debugModelLoadCapacityVars{
			Scope:      modelLoadScope(cap.Scope),
			TotalBytes: cap.TotalBytes,
			FreeBytes:  cap.FreeBytes,
			Known:      cap.Known,
			FreeKnown:  cap.Known && cap.FreeKnown,
		})
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
		if row.Bytes <= 0 {
			continue
		}
		out.MemoryPlan = append(out.MemoryPlan, debugModelLoadMemoryPlanVars{
			Class:  modelLoadClass(row.Class),
			Scope:  modelLoadScope(row.Scope),
			Bytes:  row.Bytes,
			Detail: row.Detail,
			DType:  modelLoadDType(row.DType),
		})
	}
	for _, cap := range p.sortedMemoryCapacities() {
		out.MemoryCapacities = append(out.MemoryCapacities, debugModelLoadCapacityVars{
			Scope:      modelLoadScope(cap.Scope),
			TotalBytes: cap.TotalBytes,
			FreeBytes:  cap.FreeBytes,
			Known:      cap.Known,
			FreeKnown:  cap.Known && cap.FreeKnown,
		})
	}
	out.MemoryFit = debugMemoryFitRows(modelLoadMemoryFitRows(p.MemoryPlan, p.MemoryCapacities, p.MemoryHeadroomRatio))
	return out
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
	backend := strings.TrimSpace(st.Backend)
	if backend == "" {
		backend = "unknown"
	}
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

func debugLatency(s latencySnapshot) debugLatencyVars {
	buckets := make([]debugBucketVars, 0, len(gatewayLatencyBuckets))
	for i, le := range gatewayLatencyBuckets {
		buckets = append(buckets, debugBucketVars{LESeconds: le, Count: s.buckets[i]})
	}
	return debugLatencyVars{Count: s.count, SumSeconds: s.sum, Buckets: buckets}
}
