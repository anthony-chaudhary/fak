package nativeperf

import (
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"time"
)

const (
	// SubagentTraceSchema identifies the subagent timing decomposition contract.
	SubagentTraceSchema = "fak.subagent.trace/v1"

	// Subagent phase bucket identifiers.
	SubagentPhaseHostDispatch     = "host_dispatch"
	SubagentPhasePrefixTreeLookup = "prefix_tree_lookup"
	SubagentPhaseKVAllocation     = "kv_allocation"
	SubagentPhaseGPUKernel        = "gpu_kernel"
	SubagentPhaseTokenSampling    = "token_sampling"

	// DefaultSubagentTraceToleranceUS is the clock quantization tolerance for microsecond reconciliation.
	DefaultSubagentTraceToleranceUS = 0.05
)

// SubagentPhase is a stable phase label for subagent execution profiling.
type SubagentPhase string

const (
	PhaseHostDispatch     SubagentPhase = SubagentPhaseHostDispatch
	PhasePrefixTreeLookup SubagentPhase = SubagentPhasePrefixTreeLookup
	PhaseKVAllocation     SubagentPhase = SubagentPhaseKVAllocation
	PhaseGPUKernel        SubagentPhase = SubagentPhaseGPUKernel
	PhaseTokenSampling    SubagentPhase = SubagentPhaseTokenSampling
)

var CanonicalSubagentPhases = [...]string{
	SubagentPhaseHostDispatch,
	SubagentPhasePrefixTreeLookup,
	SubagentPhaseKVAllocation,
	SubagentPhaseGPUKernel,
	SubagentPhaseTokenSampling,
}

// SubagentPhases returns the canonical list of subagent trace phase bucket names.
func SubagentPhases() []string {
	return append([]string(nil), CanonicalSubagentPhases[:]...)
}

// SubagentTraceReceipt represents high-resolution phase timing decomposing total subagent turn
// latency into host CPU orchestration, prefix tree lookup, KV allocation, GPU kernel execution,
// and token sampling.
type SubagentTraceReceipt struct {
	Schema                 string             `json:"schema"`
	Turn                   int                `json:"turn"`
	SubagentID             string             `json:"subagent_id"`
	TotalWallUS            float64            `json:"total_wall_us"`
	PhasesUS               map[string]float64 `json:"phases_us"`
	HostCPUOverheadUS      float64            `json:"host_cpu_overhead_us"`
	GPUKernelWallUS        float64            `json:"gpu_kernel_wall_us"`
	HostCPUOverheadPercent float64            `json:"host_cpu_overhead_percent"`
	GPUKernelWallPercent   float64            `json:"gpu_kernel_wall_percent"`
}

// Validate asserts that the receipt conforms to schema fak.subagent.trace/v1, has non-negative
// timings, contains all canonical phase buckets, and that individual phase durations reconcile
// to total wall time within DefaultSubagentTraceToleranceUS.
func (r SubagentTraceReceipt) Validate() error {
	return r.ValidateWithTolerance(DefaultSubagentTraceToleranceUS)
}

// ValidateWithTolerance asserts receipt validity against a caller-specified tolerance in microseconds.
func (r SubagentTraceReceipt) ValidateWithTolerance(toleranceUS float64) error {
	if r.Schema != SubagentTraceSchema {
		return fmt.Errorf("invalid schema %q, want %q", r.Schema, SubagentTraceSchema)
	}
	if strings.TrimSpace(r.SubagentID) == "" {
		return errors.New("subagent_id is empty")
	}
	if r.Turn < 0 {
		return fmt.Errorf("negative turn %d", r.Turn)
	}
	if toleranceUS < 0 || math.IsNaN(toleranceUS) || math.IsInf(toleranceUS, 0) {
		return errors.New("tolerance must be non-negative and finite")
	}
	if r.TotalWallUS < 0 || math.IsNaN(r.TotalWallUS) || math.IsInf(r.TotalWallUS, 0) {
		return errors.New("total_wall_us must be non-negative and finite")
	}
	if r.HostCPUOverheadUS < 0 || math.IsNaN(r.HostCPUOverheadUS) || math.IsInf(r.HostCPUOverheadUS, 0) {
		return errors.New("host_cpu_overhead_us must be non-negative and finite")
	}
	if r.GPUKernelWallUS < 0 || math.IsNaN(r.GPUKernelWallUS) || math.IsInf(r.GPUKernelWallUS, 0) {
		return errors.New("gpu_kernel_wall_us must be non-negative and finite")
	}
	if r.HostCPUOverheadPercent < 0 || math.IsNaN(r.HostCPUOverheadPercent) || math.IsInf(r.HostCPUOverheadPercent, 0) {
		return errors.New("host_cpu_overhead_percent must be non-negative and finite")
	}
	if r.GPUKernelWallPercent < 0 || math.IsNaN(r.GPUKernelWallPercent) || math.IsInf(r.GPUKernelWallPercent, 0) {
		return errors.New("gpu_kernel_wall_percent must be non-negative and finite")
	}
	if r.PhasesUS == nil {
		return errors.New("phases_us map is nil")
	}

	for _, req := range CanonicalSubagentPhases {
		v, ok := r.PhasesUS[req]
		if !ok {
			return fmt.Errorf("missing required phase %q", req)
		}
		if v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("phase %q duration must be non-negative and finite, got %v", req, v)
		}
	}

	var sum float64
	for k, v := range r.PhasesUS {
		if v < 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("phase %q duration must be non-negative and finite, got %v", k, v)
		}
		sum += v
	}

	if math.Abs(sum-r.TotalWallUS) > toleranceUS {
		return fmt.Errorf("phase sum %.4f µs does not reconcile to total wall %.4f µs (tolerance: %.4f µs)", sum, r.TotalWallUS, toleranceUS)
	}

	gpuVal := r.PhasesUS[SubagentPhaseGPUKernel]
	if math.Abs(r.GPUKernelWallUS-gpuVal) > toleranceUS {
		return fmt.Errorf("gpu_kernel_wall_us %.4f µs does not reconcile to gpu_kernel phase %.4f µs", r.GPUKernelWallUS, gpuVal)
	}

	expectedHost := sum - gpuVal
	if math.Abs(r.HostCPUOverheadUS-expectedHost) > toleranceUS {
		return fmt.Errorf("host_cpu_overhead_us %.4f µs does not reconcile to non-gpu phases sum %.4f µs", r.HostCPUOverheadUS, expectedHost)
	}

	if r.TotalWallUS > toleranceUS {
		expectedHostPct := (r.HostCPUOverheadUS / r.TotalWallUS) * 100.0
		expectedGPUPct := (r.GPUKernelWallUS / r.TotalWallUS) * 100.0
		const pctTolerance = 0.05
		if math.Abs(r.HostCPUOverheadPercent-expectedHostPct) > pctTolerance {
			return fmt.Errorf("host_cpu_overhead_percent %.2f%% does not reconcile to expected %.2f%%", r.HostCPUOverheadPercent, expectedHostPct)
		}
		if math.Abs(r.GPUKernelWallPercent-expectedGPUPct) > pctTolerance {
			return fmt.Errorf("gpu_kernel_wall_percent %.2f%% does not reconcile to expected %.2f%%", r.GPUKernelWallPercent, expectedGPUPct)
		}
		if math.Abs((r.HostCPUOverheadPercent+r.GPUKernelWallPercent)-100.0) > (2 * pctTolerance) {
			return fmt.Errorf("host and gpu percentages (%.2f%% + %.2f%%) do not sum to 100%%", r.HostCPUOverheadPercent, r.GPUKernelWallPercent)
		}
	} else {
		if r.HostCPUOverheadPercent != 0 || r.GPUKernelWallPercent != 0 {
			return fmt.Errorf("expected 0%% overhead and kernel percentages for 0 total wall, got %.2f%% and %.2f%%", r.HostCPUOverheadPercent, r.GPUKernelWallPercent)
		}
	}

	return nil
}

// JSON serializes the receipt to indented JSON bytes.
func (r SubagentTraceReceipt) JSON() ([]byte, error) {
	return json.MarshalIndent(r, "", "  ")
}

// NewSubagentTraceReceipt builds and validates a SubagentTraceReceipt from phase microsecond values.
// If totalWallUS is <= 0, it is computed as the sum of all phase durations.
func NewSubagentTraceReceipt(turn int, subagentID string, phases map[string]float64, totalWallUS float64) (SubagentTraceReceipt, error) {
	phasesCopy := make(map[string]float64, len(CanonicalSubagentPhases))
	for _, p := range CanonicalSubagentPhases {
		phasesCopy[p] = 0.0
	}
	var sum float64
	for k, v := range phases {
		phasesCopy[k] = v
		sum += v
	}

	if totalWallUS <= 0 {
		totalWallUS = sum
	}

	gpuUS := phasesCopy[SubagentPhaseGPUKernel]
	hostUS := sum - gpuUS
	if hostUS < 0 {
		hostUS = 0
	}

	var hostPct, gpuPct float64
	if totalWallUS > 0 {
		hostPct = (hostUS / totalWallUS) * 100.0
		gpuPct = (gpuUS / totalWallUS) * 100.0
	}

	receipt := SubagentTraceReceipt{
		Schema:                 SubagentTraceSchema,
		Turn:                   turn,
		SubagentID:             subagentID,
		TotalWallUS:            totalWallUS,
		PhasesUS:               phasesCopy,
		HostCPUOverheadUS:      hostUS,
		GPUKernelWallUS:        gpuUS,
		HostCPUOverheadPercent: hostPct,
		GPUKernelWallPercent:   gpuPct,
	}

	if err := receipt.Validate(); err != nil {
		return SubagentTraceReceipt{}, err
	}
	return receipt, nil
}

// SubagentTraceTimer provides high-resolution phase timing for subagent execution turns.
// It tracks durations spent in individual phases and computes reconciled trace receipts.
type SubagentTraceTimer struct {
	mu          sync.Mutex
	turn        int
	subagentID  string
	phasesUS    map[string]float64
	startTime   time.Time
	totalWallUS float64
}

// NewSubagentTraceTimer creates a new high-resolution phase timer for the given turn and subagent ID.
func NewSubagentTraceTimer(turn int, subagentID string) *SubagentTraceTimer {
	phases := make(map[string]float64, len(CanonicalSubagentPhases))
	for _, p := range CanonicalSubagentPhases {
		phases[p] = 0.0
	}
	return &SubagentTraceTimer{
		turn:       turn,
		subagentID: subagentID,
		phasesUS:   phases,
		startTime:  time.Now(),
	}
}

// StartPhase begins timing a phase and returns a stop callback that accumulates the elapsed
// duration into that phase bucket.
func (t *SubagentTraceTimer) StartPhase(phase string) func() {
	start := time.Now()
	return func() {
		elapsed := time.Since(start)
		t.RecordDuration(phase, elapsed)
	}
}

// Start is an alias for StartPhase.
func (t *SubagentTraceTimer) Start(phase string) func() {
	return t.StartPhase(phase)
}

// TimePhase executes fn while recording its execution duration into phase.
func (t *SubagentTraceTimer) TimePhase(phase string, fn func()) {
	stop := t.StartPhase(phase)
	defer stop()
	fn()
}

// RecordDuration adds a time.Duration to the specified phase bucket.
func (t *SubagentTraceTimer) RecordDuration(phase string, d time.Duration) {
	t.RecordUS(phase, float64(d.Nanoseconds())/1000.0)
}

// RecordPhase adds a time.Duration to the specified phase bucket.
func (t *SubagentTraceTimer) RecordPhase(phase string, d time.Duration) {
	t.RecordDuration(phase, d)
}

// RecordUS adds microseconds to the specified phase bucket.
func (t *SubagentTraceTimer) RecordUS(phase string, us float64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.phasesUS[phase] += us
}

// SetTotalWallUS explicitly specifies the total turn wall clock duration in microseconds.
func (t *SubagentTraceTimer) SetTotalWallUS(us float64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.totalWallUS = us
}

// SetTotalWall explicitly specifies the total turn wall clock duration.
func (t *SubagentTraceTimer) SetTotalWall(d time.Duration) {
	t.SetTotalWallUS(float64(d.Nanoseconds()) / 1000.0)
}

// Finalize builds and validates the SubagentTraceReceipt from accumulated timings.
func (t *SubagentTraceTimer) Finalize() (SubagentTraceReceipt, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	phasesCopy := make(map[string]float64, len(t.phasesUS))
	for k, v := range t.phasesUS {
		phasesCopy[k] = v
	}

	return NewSubagentTraceReceipt(t.turn, t.subagentID, phasesCopy, t.totalWallUS)
}

// SubagentTraceFromPhaseReceipt converts a bounded fak-native PhaseReceipt into a SubagentTraceReceipt,
// mapping exclusive phase accounting into subagent trace buckets:
//   - prefix_tree_lookup: kv_lookup
//   - kv_allocation: kv_restore, kv_evict
//   - gpu_kernel: kernel
//   - token_sampling: sampling
//   - host_dispatch: queue_admission, model_load, tokenization, prefill, decode, host transfers, sync, output
func SubagentTraceFromPhaseReceipt(turn int, subagentID string, receipt PhaseReceipt) (SubagentTraceReceipt, error) {
	phases := make(map[string]float64, len(CanonicalSubagentPhases))
	for _, p := range CanonicalSubagentPhases {
		phases[p] = 0.0
	}

	for _, p := range receipt.Phases {
		us := float64(p.Exclusive.Wall.Nanoseconds()) / 1000.0
		switch p.Phase {
		case PhaseKVLookup:
			phases[SubagentPhasePrefixTreeLookup] += us
		case PhaseKVRestore, PhaseKVEvict:
			phases[SubagentPhaseKVAllocation] += us
		case PhaseKernel:
			phases[SubagentPhaseGPUKernel] += us
		case PhaseSampling:
			phases[SubagentPhaseTokenSampling] += us
		default:
			phases[SubagentPhaseHostDispatch] += us
		}
	}

	totalWallUS := float64(receipt.Wall.Nanoseconds()) / 1000.0
	return NewSubagentTraceReceipt(turn, subagentID, phases, totalWallUS)
}

// SubagentTrace finalizes the PhaseRecorder and projects the bounded phase receipt
// into an isolated SubagentTraceReceipt decomposing host CPU overhead from GPU execution.
func (r *PhaseRecorder) SubagentTrace(turn int, subagentID string, tolerance time.Duration) (SubagentTraceReceipt, error) {
	receipt, err := r.Finalize(tolerance)
	if err != nil {
		return SubagentTraceReceipt{}, err
	}
	return SubagentTraceFromPhaseReceipt(turn, subagentID, receipt)
}

// ToPhaseRecorder maps the SubagentTraceReceipt into a PhaseRecorder with contiguous intervals
// representing the 5 subagent phases and zero-length intervals for other bounded phases.
func (r SubagentTraceReceipt) ToPhaseRecorder(engine, backend, forwardPath string) (*PhaseRecorder, error) {
	if err := r.Validate(); err != nil {
		return nil, fmt.Errorf("cannot convert invalid receipt: %w", err)
	}

	totalWall := time.Duration(math.Round(r.TotalWallUS * 1000.0))
	recorder := NewPhaseRecorder(engine, backend, forwardPath, totalWall)

	type phaseMapping struct {
		subagentPhase string
		targetPhase   Phase
		workKind      WorkKind
	}

	mappings := []phaseMapping{
		{SubagentPhaseHostDispatch, PhaseTokenization, WorkActive},
		{SubagentPhasePrefixTreeLookup, PhaseKVLookup, WorkActive},
		{SubagentPhaseKVAllocation, PhaseKVRestore, WorkWait},
		{SubagentPhaseGPUKernel, PhaseKernel, WorkActive},
		{SubagentPhaseTokenSampling, PhaseSampling, WorkActive},
	}

	usedPhases := make(map[Phase]bool, len(mappings))
	var cursor time.Duration

	for i, m := range mappings {
		us := r.PhasesUS[m.subagentPhase]
		dur := time.Duration(math.Round(us * 1000.0))
		start := cursor
		end := cursor + dur
		if i == len(mappings)-1 {
			// Ensure final interval reaches exact totalWall to absorb clock quantization rounding
			end = totalWall
		}
		if end > totalWall {
			end = totalWall
		}
		if err := recorder.Add(m.targetPhase, "", start, end, m.workKind); err != nil {
			return nil, fmt.Errorf("failed to add phase %s: %w", m.targetPhase, err)
		}
		usedPhases[m.targetPhase] = true
		cursor = end
	}

	// Add 0-length intervals for remaining bounded phases so Finalize sees the full phaseOrder set.
	for _, p := range phaseOrder {
		if !usedPhases[p] {
			if err := recorder.Add(p, "", totalWall, totalWall, WorkActive); err != nil {
				return nil, fmt.Errorf("failed to register placeholder phase %s: %w", p, err)
			}
		}
	}

	return recorder, nil
}
