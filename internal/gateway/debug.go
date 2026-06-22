package gateway

import (
	"net/http"
	"runtime"
	"sync/atomic"
	"time"
)

type debugVarsResponse struct {
	Gateway debugGatewayVars `json:"gateway"`
	Runtime debugRuntimeVars `json:"runtime"`
	Kernel  debugKernelVars  `json:"kernel"`
	Metrics debugMetricsVars `json:"metrics"`
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
	Admitted     int64   `json:"admitted"`
	VDSOHitRatio float64 `json:"vdso_hit_ratio"`
}

type debugMetricsVars struct {
	HTTP       []debugHTTPMetricVars      `json:"http"`
	Operations []debugOperationMetricVars `json:"operations"`
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
			Admitted:     c.Admitted,
			VDSOHitRatio: ratio,
		},
		Metrics: debugMetricsVars{
			HTTP:       debugHTTPRows(httpRows),
			Operations: debugOperationRows(opRows),
		},
	}
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

func debugLatency(s latencySnapshot) debugLatencyVars {
	buckets := make([]debugBucketVars, 0, len(gatewayLatencyBuckets))
	for i, le := range gatewayLatencyBuckets {
		buckets = append(buckets, debugBucketVars{LESeconds: le, Count: s.buckets[i]})
	}
	return debugLatencyVars{Count: s.count, SumSeconds: s.sum, Buckets: buckets}
}
