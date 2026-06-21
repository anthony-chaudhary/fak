// Package metrics is the KPI layer + the A/B report shape. It owns:
//
//   - Hist: a latency histogram with p50/p99 + fixed log-spaced buckets (unit 85)
//   - the five v0.1 KPI fields (unit 77): tool-call p50/p99, vDSO-hit-rate,
//     pre-flight-catch-rate, context-pollution-rate, tokens-per-task
//   - Report: the provenance-stamped report.json the bench emits, with the
//     identical-workload guard (unit 81) and the legacy gate_primary field for
//     the syscall subsystem boundary-tax check (unit 82)
//
// The package is pure (no engine, no kernel import) so it can be unit-tested and
// reused by the bench, the steward kpi-regression check, and any external tool
// that ingests fak's /metrics.
package metrics

import (
	"encoding/json"
	"errors"
	"sort"
	"time"
)

// Hist is a latency histogram in nanoseconds.
type Hist struct {
	samples []int64
}

func (h *Hist) Record(d time.Duration) { h.samples = append(h.samples, int64(d)) }
func (h *Hist) RecordNs(ns int64)      { h.samples = append(h.samples, ns) }
func (h *Hist) Count() int             { return len(h.samples) }

func (h *Hist) pct(p float64) int64 {
	if len(h.samples) == 0 {
		return 0
	}
	s := append([]int64(nil), h.samples...)
	sort.Slice(s, func(i, j int) bool { return s[i] < s[j] })
	idx := int(p / 100 * float64(len(s)-1))
	if idx < 0 {
		idx = 0
	}
	if idx >= len(s) {
		idx = len(s) - 1
	}
	return s[idx]
}

func (h *Hist) P50() int64 { return h.pct(50) }
func (h *Hist) P99() int64 { return h.pct(99) }
func (h *Hist) Mean() int64 {
	if len(h.samples) == 0 {
		return 0
	}
	var sum int64
	for _, v := range h.samples {
		sum += v
	}
	return sum / int64(len(h.samples))
}

// Buckets returns fixed log-spaced bucket counts (ns thresholds) for the report.
func (h *Hist) Buckets() []Bucket {
	edges := []int64{100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000, 1_000_000_000}
	out := make([]Bucket, len(edges)+1)
	labels := []string{"<100ns", "<1µs", "<10µs", "<100µs", "<1ms", "<10ms", "<100ms", "<1s", ">=1s"}
	for i := range out {
		out[i].Label = labels[i]
	}
	for _, v := range h.samples {
		placed := false
		for i, e := range edges {
			if v < e {
				out[i].Count++
				placed = true
				break
			}
		}
		if !placed {
			out[len(out)-1].Count++
		}
	}
	return out
}

type Bucket struct {
	Label string `json:"label"`
	Count int    `json:"count"`
}

// Arm is one side of the A/B (vdso on or off).
type Arm struct {
	Label       string   `json:"label"`
	Calls       int      `json:"calls"`
	P50Ns       int64    `json:"p50_ns"`
	P99Ns       int64    `json:"p99_ns"`
	MeanNs      int64    `json:"mean_ns"`
	EngineCalls int64    `json:"engine_calls"`
	VDSOHits    int64    `json:"vdso_hits"`
	Denies      int64    `json:"denies"`
	Quarantines int64    `json:"quarantines"`
	InTokens    int64    `json:"input_tokens"`
	OutTokens   int64    `json:"output_tokens"`
	Buckets     []Bucket `json:"buckets"`

	// Provider prompt-prefix cache telemetry (issue #112), kept DISTINCT from the
	// local-reuse counters above. These are cost/latency savings the remote
	// provider (Anthropic/OpenAI) granted on a cached prompt prefix — they are NOT
	// a fak-local cache win and must never be folded into VDSOHits or a local
	// token-saved total. A benchmark reports them under their own labels.
	ProviderCacheHits       int64 `json:"provider_cache_hits"`
	ProviderCacheReadTokens int64 `json:"provider_cache_read_tokens"`
}

// Baseline is the recorded spawned-hook decide latency (unit 23).
type Baseline struct {
	Source     string `json:"source"`
	P50Ns      int64  `json:"p50_ns"`
	P99Ns      int64  `json:"p99_ns"`
	Calls      int    `json:"calls"`
	SpawnModel string `json:"spawn_model"`
}

// Provenance pins what produced the report (unit 80).
type Provenance struct {
	AppVersion   string `json:"app_version"`
	Command      string `json:"command"`
	EngineModel  string `json:"engine_model"`
	SliceID      string `json:"slice_id"`
	WorkloadHash string `json:"workload_hash"`
	GoVersion    string `json:"go_version"`
	OS           string `json:"os"`
	GeneratedBy  string `json:"generated_by"`
}

// KPIs is the five-counter KPI set (unit 77).
type KPIs struct {
	ToolCallP50Ns        int64   `json:"tool_call_p50_ns"`
	ToolCallP99Ns        int64   `json:"tool_call_p99_ns"`
	VDSOHitRate          float64 `json:"vdso_hit_rate"`
	PreflightCatchRate   float64 `json:"preflight_catch_rate"`
	ContextPollutionRate float64 `json:"context_pollution_rate"`
	TokensPerTask        float64 `json:"tokens_per_task"`
}

// Report is the full A/B artifact (report.json).
type Report struct {
	Provenance    Provenance `json:"provenance"`
	On            Arm        `json:"vdso_on"`
	Off           Arm        `json:"vdso_off"`
	Baseline      Baseline   `json:"spawned_hook_baseline"`
	KPIs          KPIs       `json:"kpis"`
	GatePrimary   string     `json:"gate_primary"` // "pass"/"fail" for the syscall subsystem check (unit 82)
	PrimaryDetail string     `json:"primary_detail"`
	TokenDeltaPct float64    `json:"token_delta_pct"` // secondary, soft (unit 83)
	DollarPerTask float64    `json:"dollar_per_task"` // tokencost (unit 84)
	LiveSeam      string     `json:"live_seam"`       // transcript hash xor "live_seam_unverified"
}

// ComputeGate fills the legacy gate_primary field for the syscall subsystem
// boundary-tax check (unit 82): in-process adjudication p50 should beat the
// spawned-hook baseline. This is a regression sentinel for the decide path, not
// a production-readiness or serving-throughput gate.
func (r *Report) ComputeGate() {
	on := r.On.P50Ns
	base := r.Baseline.P50Ns
	if base > 0 && on < base {
		r.GatePrimary = "pass"
	} else {
		r.GatePrimary = "fail"
	}
	r.PrimaryDetail = "in-process adjudication p50 (" + itoa(on) + "ns) vs spawned-hook p50 (" + itoa(base) + "ns)"
}

// Validate enforces the identical-workload guard (unit 81): the two arms must
// share a workload hash, else the comparison is refused.
func (r *Report) Validate(onHash, offHash string) error {
	if onHash != offHash {
		return errors.New("metrics: refusing to compare arms with different workload hashes (" + onHash + " != " + offHash + ")")
	}
	return nil
}

// JSON renders the report.
func (r *Report) JSON() []byte {
	b, _ := json.MarshalIndent(r, "", "  ")
	return b
}

func itoa(n int64) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [24]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}
