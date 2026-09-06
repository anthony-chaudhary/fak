package nativeperf

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"time"
)

const (
	// TurnTraceSchema identifies the nanosecond-resolution turn latency decomposition contract.
	TurnTraceSchema = "fak.trace.turn_latency/v1"
	// TurnLatencySchema is an alias for TurnTraceSchema.
	TurnLatencySchema = TurnTraceSchema

	// MaxTracerOverheadPct is the maximum allowed tracer measurement overhead (< 1% of total turn latency).
	MaxTracerOverheadPct = 1.0

	// MaxSumInvariantTolerancePct is the maximum allowed discrepancy between the sum of phases
	// and total wall-clock turn duration (<= 0.5% tolerance).
	MaxSumInvariantTolerancePct = 0.5
)

// TurnPhase is a bounded phase label for turn latency profiling.
type TurnPhase string

const (
	TurnPhaseHostDispatch  TurnPhase = "host_dispatch"
	TurnPhasePrefixLookup  TurnPhase = "prefix_lookup"
	TurnPhaseKVAllocation  TurnPhase = "kv_allocation"
	TurnPhaseGPUKernel     TurnPhase = "gpu_kernel"
	TurnPhaseTokenSampling TurnPhase = "token_sampling"
)

// Type and constant aliases for flexible invocation.
type TracePhase = TurnPhase

const (
	TracePhaseHostDispatch  = TurnPhaseHostDispatch
	TracePhasePrefixLookup  = TurnPhasePrefixLookup
	TracePhaseKVAllocation  = TurnPhaseKVAllocation
	TracePhaseGPUKernel     = TurnPhaseGPUKernel
	TracePhaseTokenSampling = TurnPhaseTokenSampling

	// PhasePrefixLookup is exported for consistency with other Phase* constants.
	PhasePrefixLookup = TurnPhasePrefixLookup

	// PascalCase bucket name aliases.
	PhaseHostDispatchNs  = TurnPhaseHostDispatch
	PhasePrefixLookupNs  = TurnPhasePrefixLookup
	PhaseKVAllocationNs  = TurnPhaseKVAllocation
	PhaseGPUKernelNs     = TurnPhaseGPUKernel
	PhaseTokenSamplingNs = TurnPhaseTokenSampling
)

var CanonicalTurnPhases = [...]TurnPhase{
	TurnPhaseHostDispatch,
	TurnPhasePrefixLookup,
	TurnPhaseKVAllocation,
	TurnPhaseGPUKernel,
	TurnPhaseTokenSampling,
}

// TurnPhases returns the canonical list of turn latency phase labels.
func TurnPhases() []TurnPhase {
	return append([]TurnPhase(nil), CanonicalTurnPhases[:]...)
}

// TurnPhaseDescriptions provides human-readable explanations for each phase bucket.
var TurnPhaseDescriptions = map[TurnPhase]string{
	TurnPhaseHostDispatch:  "CPU process orchestration, IPC, and JSON serialization/deserialization",
	TurnPhasePrefixLookup:  "Radix/Context MMU tree prefix matching and traversal",
	TurnPhaseKVAllocation:  "Physical page reservation and block table allocation",
	TurnPhaseGPUKernel:     "Raw GPU prefill and decode GEMM/GEMV execution",
	TurnPhaseTokenSampling: "Logit sampling, temperature, and detokenization",
}

// NormalizeTurnPhase normalizes various representations of phase names into a canonical TurnPhase.
func NormalizeTurnPhase(v any) (TurnPhase, error) {
	var s string
	switch val := v.(type) {
	case TurnPhase:
		s = string(val)
	case SubagentPhase:
		s = string(val)
	case Phase:
		s = string(val)
	case string:
		s = val
	case fmt.Stringer:
		s = val.String()
	default:
		return "", fmt.Errorf("unsupported phase type %T", v)
	}

	cleaned := strings.ToLower(strings.TrimSpace(s))
	cleaned = strings.ReplaceAll(cleaned, "-", "_")

	switch cleaned {
	case "host_dispatch", "hostdispatch", "hostdispatchns":
		return TurnPhaseHostDispatch, nil
	case "prefix_lookup", "prefixlookup", "prefixlookupns", "prefix_tree_lookup", "prefixtreelookup":
		return TurnPhasePrefixLookup, nil
	case "kv_allocation", "kvallocation", "kvallocationns":
		return TurnPhaseKVAllocation, nil
	case "gpu_kernel", "gpukernel", "gpukernelns", "kernel":
		return TurnPhaseGPUKernel, nil
	case "token_sampling", "tokensampling", "tokensamplingns", "sampling":
		return TurnPhaseTokenSampling, nil
	default:
		return TurnPhase(cleaned), fmt.Errorf("unknown turn phase %q", s)
	}
}

// TurnPhaseMetrics represents timing and percentage breakdown for one phase.
type TurnPhaseMetrics struct {
	Phase        TurnPhase `json:"phase"`
	Nanoseconds  int64     `json:"nanoseconds"`
	Microseconds float64   `json:"microseconds"`
	Percentage   float64   `json:"percentage"`
	Description  string    `json:"description,omitempty"`
}

// TurnTraceReport captures the nanosecond-resolution turn latency decomposition,
// per-phase breakdown in nanoseconds, microseconds, and percentage of total turn time,
// as well as overhead and sum invariant verification results.
type TurnTraceReport struct {
	Schema    string    `json:"schema"`
	TurnID    string    `json:"turn_id,omitempty"`
	Timestamp time.Time `json:"timestamp,omitempty"`

	// Total turn wall clock duration
	TotalWallNs     int64         `json:"total_wall_ns"`
	TotalWallUS     float64       `json:"total_wall_us"`
	TotalDurationNs int64         `json:"total_duration_ns"`
	TotalDurationUS float64       `json:"total_duration_us"`
	TotalDuration   time.Duration `json:"total_duration"`

	// Deconstructed nanosecond-resolution phase buckets
	HostDispatchNs  int64 `json:"host_dispatch_ns"`
	PrefixLookupNs  int64 `json:"prefix_lookup_ns"`
	KVAllocationNs  int64 `json:"kv_allocation_ns"`
	GPUKernelNs     int64 `json:"gpu_kernel_ns"`
	TokenSamplingNs int64 `json:"token_sampling_ns"`

	// Microsecond phase buckets
	HostDispatchUS  float64 `json:"host_dispatch_us"`
	PrefixLookupUS  float64 `json:"prefix_lookup_us"`
	KVAllocationUS  float64 `json:"kv_allocation_us"`
	GPUKernelUS     float64 `json:"gpu_kernel_us"`
	TokenSamplingUS float64 `json:"token_sampling_us"`

	// Percentage breakdown (0.0 to 100.0)
	HostDispatchPct  float64 `json:"host_dispatch_pct"`
	PrefixLookupPct  float64 `json:"prefix_lookup_pct"`
	KVAllocationPct  float64 `json:"kv_allocation_pct"`
	GPUKernelPct     float64 `json:"gpu_kernel_pct"`
	TokenSamplingPct float64 `json:"token_sampling_pct"`

	// Per-phase breakdown map
	Phases    map[TurnPhase]TurnPhaseMetrics `json:"phases"`
	Breakdown map[TurnPhase]TurnPhaseMetrics `json:"breakdown,omitempty"`

	// Tracer measurement overhead metrics
	OverheadNs  int64   `json:"overhead_ns"`
	OverheadUS  float64 `json:"overhead_us"`
	OverheadPct float64 `json:"overhead_pct"`

	// Validation invariants
	OverheadValid bool    `json:"overhead_valid"`
	SumValid      bool    `json:"sum_valid"`
	SumDeltaNs    int64   `json:"sum_delta_ns"`
	SumDeltaPct   float64 `json:"sum_delta_pct"`
}

func buildTurnTraceReport(turnID string, ts time.Time, totalWallNs int64, phases map[TurnPhase]int64, overheadNs int64) TurnTraceReport {
	if ts.IsZero() {
		ts = time.Now().UTC()
	}

	report := TurnTraceReport{
		Schema:          TurnTraceSchema,
		TurnID:          turnID,
		Timestamp:       ts,
		TotalWallNs:     totalWallNs,
		TotalDurationNs: totalWallNs,
		HostDispatchNs:  phases[TurnPhaseHostDispatch],
		PrefixLookupNs:  phases[TurnPhasePrefixLookup],
		KVAllocationNs:  phases[TurnPhaseKVAllocation],
		GPUKernelNs:     phases[TurnPhaseGPUKernel],
		TokenSamplingNs: phases[TurnPhaseTokenSampling],
		OverheadNs:      overheadNs,
	}

	report.recomputeDerived()
	return report
}

// recomputeDerived recalculates all unit conversions, percentages, and invariant metrics.
func (r *TurnTraceReport) recomputeDerived() {
	if r.Schema == "" {
		r.Schema = TurnTraceSchema
	}

	if r.TotalWallNs == 0 && r.TotalDurationNs > 0 {
		r.TotalWallNs = r.TotalDurationNs
	}
	if r.TotalDurationNs == 0 && r.TotalWallNs > 0 {
		r.TotalDurationNs = r.TotalWallNs
	}

	sumPhases := r.HostDispatchNs + r.PrefixLookupNs + r.KVAllocationNs + r.GPUKernelNs + r.TokenSamplingNs
	if r.TotalWallNs == 0 && sumPhases > 0 {
		r.TotalWallNs = sumPhases
		r.TotalDurationNs = sumPhases
	}

	r.TotalWallUS = float64(r.TotalWallNs) / 1000.0
	r.TotalDurationUS = float64(r.TotalDurationNs) / 1000.0
	r.TotalDuration = time.Duration(r.TotalWallNs) * time.Nanosecond

	r.HostDispatchUS = float64(r.HostDispatchNs) / 1000.0
	r.PrefixLookupUS = float64(r.PrefixLookupNs) / 1000.0
	r.KVAllocationUS = float64(r.KVAllocationNs) / 1000.0
	r.GPUKernelUS = float64(r.GPUKernelNs) / 1000.0
	r.TokenSamplingUS = float64(r.TokenSamplingNs) / 1000.0

	r.OverheadUS = float64(r.OverheadNs) / 1000.0

	if r.TotalWallNs > 0 {
		totalF := float64(r.TotalWallNs)
		r.HostDispatchPct = (float64(r.HostDispatchNs) / totalF) * 100.0
		r.PrefixLookupPct = (float64(r.PrefixLookupNs) / totalF) * 100.0
		r.KVAllocationPct = (float64(r.KVAllocationNs) / totalF) * 100.0
		r.GPUKernelPct = (float64(r.GPUKernelNs) / totalF) * 100.0
		r.TokenSamplingPct = (float64(r.TokenSamplingNs) / totalF) * 100.0
		r.OverheadPct = (float64(r.OverheadNs) / totalF) * 100.0
	} else {
		r.HostDispatchPct = 0
		r.PrefixLookupPct = 0
		r.KVAllocationPct = 0
		r.GPUKernelPct = 0
		r.TokenSamplingPct = 0
		r.OverheadPct = 0
	}

	r.OverheadValid = r.OverheadPct < MaxTracerOverheadPct

	diffNs := sumPhases - r.TotalWallNs
	if diffNs < 0 {
		diffNs = -diffNs
	}
	r.SumDeltaNs = diffNs

	if r.TotalWallNs > 0 {
		r.SumDeltaPct = (float64(diffNs) / float64(r.TotalWallNs)) * 100.0
	} else {
		r.SumDeltaPct = 0
	}
	r.SumValid = r.SumDeltaPct <= MaxSumInvariantTolerancePct

	// Populate phases maps
	if r.Phases == nil {
		r.Phases = make(map[TurnPhase]TurnPhaseMetrics, len(CanonicalTurnPhases))
	}
	r.Phases[TurnPhaseHostDispatch] = TurnPhaseMetrics{
		Phase:        TurnPhaseHostDispatch,
		Nanoseconds:  r.HostDispatchNs,
		Microseconds: r.HostDispatchUS,
		Percentage:   r.HostDispatchPct,
		Description:  TurnPhaseDescriptions[TurnPhaseHostDispatch],
	}
	r.Phases[TurnPhasePrefixLookup] = TurnPhaseMetrics{
		Phase:        TurnPhasePrefixLookup,
		Nanoseconds:  r.PrefixLookupNs,
		Microseconds: r.PrefixLookupUS,
		Percentage:   r.PrefixLookupPct,
		Description:  TurnPhaseDescriptions[TurnPhasePrefixLookup],
	}
	r.Phases[TurnPhaseKVAllocation] = TurnPhaseMetrics{
		Phase:        TurnPhaseKVAllocation,
		Nanoseconds:  r.KVAllocationNs,
		Microseconds: r.KVAllocationUS,
		Percentage:   r.KVAllocationPct,
		Description:  TurnPhaseDescriptions[TurnPhaseKVAllocation],
	}
	r.Phases[TurnPhaseGPUKernel] = TurnPhaseMetrics{
		Phase:        TurnPhaseGPUKernel,
		Nanoseconds:  r.GPUKernelNs,
		Microseconds: r.GPUKernelUS,
		Percentage:   r.GPUKernelPct,
		Description:  TurnPhaseDescriptions[TurnPhaseGPUKernel],
	}
	r.Phases[TurnPhaseTokenSampling] = TurnPhaseMetrics{
		Phase:        TurnPhaseTokenSampling,
		Nanoseconds:  r.TokenSamplingNs,
		Microseconds: r.TokenSamplingUS,
		Percentage:   r.TokenSamplingPct,
		Description:  TurnPhaseDescriptions[TurnPhaseTokenSampling],
	}

	r.Breakdown = r.Phases
}

// AssertLowOverhead proves that the tracer measurement overhead is < 1% of total turn latency.
func (r TurnTraceReport) AssertLowOverhead() error {
	if r.TotalWallNs <= 0 {
		return nil
	}
	if r.OverheadPct >= MaxTracerOverheadPct {
		return fmt.Errorf("tracer measurement overhead %.4f%% exceeds limit of %.2f%% (overhead: %d ns, total: %d ns)",
			r.OverheadPct, MaxTracerOverheadPct, r.OverheadNs, r.TotalWallNs)
	}
	return nil
}

// AssertSumInvariant proves that Sum(Phases) == TotalWallClock within 0.5% tolerance.
func (r TurnTraceReport) AssertSumInvariant() error {
	sumPhases := r.HostDispatchNs + r.PrefixLookupNs + r.KVAllocationNs + r.GPUKernelNs + r.TokenSamplingNs
	if r.TotalWallNs <= 0 {
		if sumPhases == 0 {
			return nil
		}
		return fmt.Errorf("sum invariant violated: phases sum %d ns but total wall is 0", sumPhases)
	}

	diffNs := sumPhases - r.TotalWallNs
	if diffNs < 0 {
		diffNs = -diffNs
	}
	pct := (float64(diffNs) / float64(r.TotalWallNs)) * 100.0
	if pct > MaxSumInvariantTolerancePct {
		return fmt.Errorf("sum invariant violated: phases sum %d ns differs from total wall %d ns by %.4f%% (tolerance: %.2f%%)",
			sumPhases, r.TotalWallNs, pct, MaxSumInvariantTolerancePct)
	}
	return nil
}

// CheckOverhead is an alias for AssertLowOverhead.
func (r TurnTraceReport) CheckOverhead() error {
	return r.AssertLowOverhead()
}

// CheckSumInvariant is an alias for AssertSumInvariant.
func (r TurnTraceReport) CheckSumInvariant() error {
	return r.AssertSumInvariant()
}

// Validate checks schema conformance, non-negativity, sum invariant, and tracer overhead.
func (r TurnTraceReport) Validate() error {
	if r.Schema != TurnTraceSchema {
		return fmt.Errorf("invalid schema %q, want %q", r.Schema, TurnTraceSchema)
	}
	if r.TotalWallNs < 0 {
		return fmt.Errorf("total_wall_ns must be non-negative, got %d", r.TotalWallNs)
	}
	if r.HostDispatchNs < 0 || r.PrefixLookupNs < 0 || r.KVAllocationNs < 0 || r.GPUKernelNs < 0 || r.TokenSamplingNs < 0 {
		return errors.New("phase durations must be non-negative")
	}
	if r.OverheadNs < 0 {
		return fmt.Errorf("overhead_ns must be non-negative, got %d", r.OverheadNs)
	}
	if err := r.AssertSumInvariant(); err != nil {
		return err
	}
	if err := r.AssertLowOverhead(); err != nil {
		return err
	}
	return nil
}

// JSON returns formatted, indented JSON serialization conforming to schema fak.trace.turn_latency/v1.
func (r TurnTraceReport) JSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

// UnmarshalJSON implements custom deserialization supporting snake_case, PascalCase, and legacy forms.
func (r *TurnTraceReport) UnmarshalJSON(data []byte) error {
	type Alias TurnTraceReport
	var aux struct {
		Alias
		AltHostDispatchNs  *int64   `json:"HostDispatchNs"`
		AltPrefixLookupNs  *int64   `json:"PrefixLookupNs"`
		AltKVAllocationNs  *int64   `json:"KVAllocationNs"`
		AltGPUKernelNs     *int64   `json:"GPUKernelNs"`
		AltTokenSamplingNs *int64   `json:"TokenSamplingNs"`
		AltTotalWallNs     *int64   `json:"TotalWallNs"`
		AltTotalDurationNs *int64   `json:"TotalDurationNs"`
		AltTotalWallUS     *float64 `json:"TotalWallUS"`
		AltTotalDurationUS *float64 `json:"TotalDurationUS"`
		AltOverheadNs      *int64   `json:"OverheadNs"`
	}

	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}

	*r = TurnTraceReport(aux.Alias)

	if aux.AltHostDispatchNs != nil && r.HostDispatchNs == 0 {
		r.HostDispatchNs = *aux.AltHostDispatchNs
	}
	if aux.AltPrefixLookupNs != nil && r.PrefixLookupNs == 0 {
		r.PrefixLookupNs = *aux.AltPrefixLookupNs
	}
	if aux.AltKVAllocationNs != nil && r.KVAllocationNs == 0 {
		r.KVAllocationNs = *aux.AltKVAllocationNs
	}
	if aux.AltGPUKernelNs != nil && r.GPUKernelNs == 0 {
		r.GPUKernelNs = *aux.AltGPUKernelNs
	}
	if aux.AltTokenSamplingNs != nil && r.TokenSamplingNs == 0 {
		r.TokenSamplingNs = *aux.AltTokenSamplingNs
	}
	if aux.AltTotalWallNs != nil && r.TotalWallNs == 0 {
		r.TotalWallNs = *aux.AltTotalWallNs
	}
	if aux.AltTotalDurationNs != nil && r.TotalDurationNs == 0 {
		r.TotalDurationNs = *aux.AltTotalDurationNs
	}
	if aux.AltTotalWallUS != nil && r.TotalWallUS == 0 {
		r.TotalWallUS = *aux.AltTotalWallUS
	}
	if aux.AltTotalDurationUS != nil && r.TotalDurationUS == 0 {
		r.TotalDurationUS = *aux.AltTotalDurationUS
	}
	if aux.AltOverheadNs != nil && r.OverheadNs == 0 {
		r.OverheadNs = *aux.AltOverheadNs
	}

	r.recomputeDerived()
	return nil
}

// DecodeTurnTraceReport parses and validates JSON bytes conforming to fak.trace.turn_latency/v1.
func DecodeTurnTraceReport(data []byte) (TurnTraceReport, error) {
	var report TurnTraceReport
	if err := json.Unmarshal(data, &report); err != nil {
		return TurnTraceReport{}, fmt.Errorf("decode turn trace report: %w", err)
	}
	if err := report.Validate(); err != nil {
		return report, fmt.Errorf("validate turn trace report: %w", err)
	}
	return report, nil
}

// NewTurnTraceReport creates and validates a TurnTraceReport from raw nanosecond phase buckets.
func NewTurnTraceReport(turnID string, totalWallNs int64, phases map[TurnPhase]int64, overheadNs int64) (TurnTraceReport, error) {
	report := buildTurnTraceReport(turnID, time.Now().UTC(), totalWallNs, phases, overheadNs)
	if err := report.Validate(); err != nil {
		return report, err
	}
	return report, nil
}

// TurnLatencyTracer provides nanosecond-resolution phase timing to isolate subagent dispatch
// and prefix-tree ingestion overhead from raw GPU execution.
type TurnLatencyTracer struct {
	mu          sync.Mutex
	turnID      string
	startTime   time.Time
	stopTime    time.Time
	totalWallNs int64

	activePhase TurnPhase
	phaseStart  time.Time

	phasesNs   map[TurnPhase]int64
	overheadNs int64
}

// NewTurnLatencyTracer creates a new nanosecond-resolution turn latency tracer.
func NewTurnLatencyTracer(turnID ...string) *TurnLatencyTracer {
	id := ""
	if len(turnID) > 0 {
		id = turnID[0]
	}
	phases := make(map[TurnPhase]int64, len(CanonicalTurnPhases))
	for _, p := range CanonicalTurnPhases {
		phases[p] = 0
	}
	return &TurnLatencyTracer{
		turnID:    id,
		startTime: time.Now(),
		phasesNs:  phases,
	}
}

// StartTurn resets and starts the turn-level wall clock timer.
func (t *TurnLatencyTracer) StartTurn() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.startTime = time.Now()
	t.stopTime = time.Time{}
}

// StartPhase begins timing a phase and returns a stop callback function.
func (t *TurnLatencyTracer) StartPhase(phase TurnPhase) func() {
	t0 := time.Now()
	t.mu.Lock()
	defer func() {
		overhead := time.Since(t0)
		t.overheadNs += overhead.Nanoseconds()
		t.mu.Unlock()
	}()

	if t.phasesNs == nil {
		t.phasesNs = make(map[TurnPhase]int64, len(CanonicalTurnPhases))
	}
	canon, _ := NormalizeTurnPhase(phase)
	t.activePhase = canon
	t.phaseStart = t0

	return func() {
		t.EndPhase(canon)
	}
}

// Start is an alias for StartPhase.
func (t *TurnLatencyTracer) Start(phase TurnPhase) func() {
	return t.StartPhase(phase)
}

// EndPhase finishes timing the active phase (or optionally specified phase),
// accumulates the elapsed nanoseconds into the phase bucket, and returns elapsed duration.
func (t *TurnLatencyTracer) EndPhase(phase ...TurnPhase) time.Duration {
	t0 := time.Now()
	t.mu.Lock()
	defer func() {
		overhead := time.Since(t0)
		t.overheadNs += overhead.Nanoseconds()
		t.mu.Unlock()
	}()

	if t.phasesNs == nil {
		t.phasesNs = make(map[TurnPhase]int64, len(CanonicalTurnPhases))
	}

	p := t.activePhase
	if len(phase) > 0 && phase[0] != "" {
		canon, _ := NormalizeTurnPhase(phase[0])
		p = canon
	}
	if p == "" {
		return 0
	}

	dur := t0.Sub(t.phaseStart)
	if dur < 0 {
		dur = 0
	}
	t.phasesNs[p] += dur.Nanoseconds()
	t.activePhase = ""
	return dur
}

// RecordPhase adds a time.Duration to the specified phase bucket.
func (t *TurnLatencyTracer) RecordPhase(phase TurnPhase, d time.Duration) {
	t0 := time.Now()
	t.mu.Lock()
	defer func() {
		overhead := time.Since(t0)
		t.overheadNs += overhead.Nanoseconds()
		t.mu.Unlock()
	}()

	if t.phasesNs == nil {
		t.phasesNs = make(map[TurnPhase]int64, len(CanonicalTurnPhases))
	}

	canon, _ := NormalizeTurnPhase(phase)
	ns := d.Nanoseconds()
	if ns < 0 {
		ns = 0
	}
	t.phasesNs[canon] += ns
}

// RecordPhaseNs adds nanoseconds directly to the specified phase bucket.
func (t *TurnLatencyTracer) RecordPhaseNs(phase TurnPhase, ns int64) {
	t.RecordPhase(phase, time.Duration(ns)*time.Nanosecond)
}

// TimePhase executes fn while recording its execution duration into phase.
func (t *TurnLatencyTracer) TimePhase(phase TurnPhase, fn func()) {
	stop := t.StartPhase(phase)
	defer stop()
	fn()
}

// SetTotalWall explicitly sets the total turn wall clock duration.
func (t *TurnLatencyTracer) SetTotalWall(d time.Duration) {
	t.SetTotalWallNs(d.Nanoseconds())
}

// SetTotalWallNs explicitly sets the total turn wall clock duration in nanoseconds.
func (t *TurnLatencyTracer) SetTotalWallNs(ns int64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.totalWallNs = ns
}

// EndTurn marks the completion of the turn and records total wall time.
func (t *TurnLatencyTracer) EndTurn() time.Duration {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.stopTime = time.Now()
	dur := t.stopTime.Sub(t.startTime)
	if dur < 0 {
		dur = 0
	}
	t.totalWallNs = dur.Nanoseconds()
	return dur
}

// Stop is an alias for EndTurn.
func (t *TurnLatencyTracer) Stop() time.Duration {
	return t.EndTurn()
}

// Overhead returns total accumulated tracer measurement overhead duration.
func (t *TurnLatencyTracer) Overhead() time.Duration {
	t.mu.Lock()
	defer t.mu.Unlock()
	return time.Duration(t.overheadNs) * time.Nanosecond
}

// OverheadNs returns total accumulated tracer measurement overhead in nanoseconds.
func (t *TurnLatencyTracer) OverheadNs() int64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.overheadNs
}

// SetOverheadNs explicitly sets the overhead in nanoseconds (useful for deterministic tests).
func (t *TurnLatencyTracer) SetOverheadNs(ns int64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.overheadNs = ns
}

// Reset clears accumulated timings and resets state for a new turn.
func (t *TurnLatencyTracer) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.startTime = time.Now()
	t.stopTime = time.Time{}
	t.totalWallNs = 0
	t.activePhase = ""
	t.overheadNs = 0
	for _, p := range CanonicalTurnPhases {
		t.phasesNs[p] = 0
	}
}

// Report builds and validates the TurnTraceReport from accumulated timings.
func (t *TurnLatencyTracer) Report() (TurnTraceReport, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	totalNs := t.totalWallNs
	if totalNs <= 0 {
		if !t.stopTime.IsZero() && t.stopTime.After(t.startTime) {
			totalNs = t.stopTime.Sub(t.startTime).Nanoseconds()
		} else {
			var sum int64
			for _, p := range CanonicalTurnPhases {
				sum += t.phasesNs[p]
			}
			if sum > 0 {
				totalNs = sum
			} else if !t.startTime.IsZero() {
				totalNs = time.Since(t.startTime).Nanoseconds()
			}
		}
	}

	phasesCopy := make(map[TurnPhase]int64, len(CanonicalTurnPhases))
	for _, p := range CanonicalTurnPhases {
		phasesCopy[p] = t.phasesNs[p]
	}

	report := buildTurnTraceReport(t.turnID, t.startTime, totalNs, phasesCopy, t.overheadNs)
	if err := report.Validate(); err != nil {
		return report, err
	}
	return report, nil
}

// Finalize is an alias for Report.
func (t *TurnLatencyTracer) Finalize() (TurnTraceReport, error) {
	return t.Report()
}

// ToJSON builds the report and serializes it to indented JSON bytes.
func (t *TurnLatencyTracer) ToJSON() ([]byte, error) {
	report, err := t.Report()
	if err != nil {
		return nil, err
	}
	return report.JSON()
}
