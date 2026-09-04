package modelengine

// workercoupling.go — dynamic runtime-capacity worker coupling with Go 1.26 scheduler metrics.
//
// In high-concurrency and mixed-workload environments (where prompt prefill and token decode
// run concurrently with background GC, disk I/O, and agent subprocesses on multi-core hosts),
// static thread allocations cause severe thread contention, OS context switching, and high
// p99 latency cliffs.
//
// WorkerCoupler dynamically couples model forward-pass thread concurrency to live Go runtime
// scheduler pressure (/sched/latencies:seconds, /sched/goroutines:runnable,
// /cpu/classes/idle:cpu-seconds, /sched/gomaxprocs:threads).
//
// It differentiates:
//   - OpPrefill: compute-heavy prompt prefill GEMM, which scales with cores when quiet but
//     must throttle down when runnable run-queues backlog to prevent CPU saturation storms.
//   - OpDecode: lightweight memory-bandwidth-bound single-token decode GEMV, which saturates
//     at modest thread counts and must clamp aggressively (down to 1-2 workers) under scheduler
//     contention to eliminate barrier stalls and thread-park churn.
//   - OpBatch: continuous batching multi-lane StepBatch, scaling between decode and prefill.
//   - OpPipeline: network PP forward worker dispatch.
//
// Telemetry and Prometheus-formatted metrics are exposed for real-time observability.

import (
	"fmt"
	"math"
	"os"
	"runtime"
	rmetrics "runtime/metrics"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// WorkerOpKind identifies the model forward pass operation kind.
type WorkerOpKind uint8

const (
	// OpPrefill represents compute-heavy prompt prefill GEMM.
	OpPrefill WorkerOpKind = iota
	// OpDecode represents lightweight memory-bandwidth-bound single-token decode GEMV.
	OpDecode
	// OpBatch represents continuous batching multi-lane StepBatch.
	OpBatch
	// OpPipeline represents multi-stage pipeline forward dispatch.
	OpPipeline
)

func (k WorkerOpKind) String() string {
	switch k {
	case OpPrefill:
		return "prefill"
	case OpDecode:
		return "decode"
	case OpBatch:
		return "batch"
	case OpPipeline:
		return "pipeline"
	default:
		return "unknown"
	}
}

// WorkerCouplingMode selects the dynamic coupling regime.
type WorkerCouplingMode uint8

const (
	// CouplingModeDynamic dynamically tunes concurrency based on Go runtime scheduler metrics.
	CouplingModeDynamic WorkerCouplingMode = iota
	// CouplingModeStatic locks worker concurrency to configured static maximums.
	CouplingModeStatic
	// CouplingModeDisabled bypasses coupling and leaves model worker settings untouched.
	CouplingModeDisabled
)

func (m WorkerCouplingMode) String() string {
	switch m {
	case CouplingModeDynamic:
		return "dynamic"
	case CouplingModeStatic:
		return "static"
	case CouplingModeDisabled:
		return "disabled"
	default:
		return "unknown"
	}
}

// RuntimePressureClass categorizes Go runtime scheduler contention.
type RuntimePressureClass string

const (
	PressureNominal  RuntimePressureClass = "nominal"
	PressureLight    RuntimePressureClass = "light"
	PressureModerate RuntimePressureClass = "moderate"
	PressureHigh     RuntimePressureClass = "high"
	PressureCritical RuntimePressureClass = "critical"
)

// RuntimeMetricsSnapshot captures a point-in-time sample of Go runtime scheduler metrics.
type RuntimeMetricsSnapshot struct {
	RunnableGoroutines int64
	GOMAXPROCS         int
	NumCPU             int
	SchedLatencyP50    time.Duration
	SchedLatencyP95    time.Duration
	CPUIdleFraction    float64
	SampledAt          time.Time
}

// MetricsSampler reads or synthesizes Go runtime scheduler metrics.
type MetricsSampler func() RuntimeMetricsSnapshot

// WorkerCouplingConfig configures dynamic worker coupling.
type WorkerCouplingConfig struct {
	Mode               WorkerCouplingMode
	MaxPrefillWorkers  int
	MaxDecodeWorkers   int
	MaxBatchWorkers    int
	MaxPipelineWorkers int
	MinWorkers         int
	SampleInterval     time.Duration
	MaxRunningCap      int
}

// DefaultWorkerCouplingConfig returns production defaults backed by host hardware topology.
func DefaultWorkerCouplingConfig() WorkerCouplingConfig {
	procs := runtime.GOMAXPROCS(0)
	return WorkerCouplingConfig{
		Mode:               CouplingModeDynamic,
		MaxPrefillWorkers:  procs,
		MaxDecodeWorkers:   decodeMaxWorkersFor(procs),
		MaxBatchWorkers:    batchMaxWorkersFor(procs),
		MaxPipelineWorkers: pipelineMaxWorkersFor(procs),
		MinWorkers:         1,
		SampleInterval:     25 * time.Millisecond,
		MaxRunningCap:      0,
	}
}

// WorkerCouplingConfigFromEnv constructs a configuration from environment variables.
func WorkerCouplingConfigFromEnv() WorkerCouplingConfig {
	p := DefaultWorkerCouplingConfig()
	if raw := os.Getenv("FAK_WORKER_COUPLING"); raw != "" {
		switch strings.ToLower(strings.TrimSpace(raw)) {
		case "0", "false", "off", "disabled", "no":
			p.Mode = CouplingModeDisabled
		case "static":
			p.Mode = CouplingModeStatic
		case "1", "true", "on", "enabled", "yes", "dynamic", "auto":
			p.Mode = CouplingModeDynamic
		}
	}
	if raw := os.Getenv("FAK_WORKER_COUPLING_MODE"); raw != "" {
		switch strings.ToLower(strings.TrimSpace(raw)) {
		case "disabled", "off":
			p.Mode = CouplingModeDisabled
		case "static":
			p.Mode = CouplingModeStatic
		case "dynamic", "auto":
			p.Mode = CouplingModeDynamic
		}
	}
	if raw := os.Getenv("FAK_WORKER_COUPLING_PREFILL_MAX"); raw != "" {
		if n, err := strconv.Atoi(strings.TrimSpace(raw)); err == nil && n > 0 {
			p.MaxPrefillWorkers = n
		}
	}
	if raw := os.Getenv("FAK_WORKER_COUPLING_DECODE_MAX"); raw != "" {
		if n, err := strconv.Atoi(strings.TrimSpace(raw)); err == nil && n > 0 {
			p.MaxDecodeWorkers = n
		}
	}
	if raw := os.Getenv("FAK_WORKER_COUPLING_MIN"); raw != "" {
		if n, err := strconv.Atoi(strings.TrimSpace(raw)); err == nil && n > 0 {
			p.MinWorkers = n
		}
	}
	if raw := os.Getenv("FAK_WORKER_COUPLING_INTERVAL_MS"); raw != "" {
		if n, err := strconv.Atoi(strings.TrimSpace(raw)); err == nil && n >= 0 {
			p.SampleInterval = time.Duration(n) * time.Millisecond
		}
	}
	return p
}

// WorkerCouplingStats reports telemetry for the active worker coupling state.
type WorkerCouplingStats struct {
	Enabled            bool                 `json:"enabled"`
	Mode               string               `json:"mode"`
	GOMAXPROCS         int                  `json:"gomaxprocs"`
	NumCPU             int                  `json:"num_cpu"`
	PressureClass      RuntimePressureClass `json:"pressure_class"`
	PressureScore      float64              `json:"pressure_score"`
	RunnableGoroutines int64                `json:"runnable_goroutines"`
	SchedLatencyP50    time.Duration        `json:"sched_latency_p50"`
	SchedLatencyP95    time.Duration        `json:"sched_latency_p95"`
	CPUIdleFraction    float64              `json:"cpu_idle_fraction"`
	SampleCount        int64                `json:"sample_count"`
	ThrottleEvents     int64                `json:"throttle_events"`
	PrefillWorkers     int                  `json:"prefill_workers"`
	DecodeWorkers      int                  `json:"decode_workers"`
	BatchWorkers       int                  `json:"batch_workers"`
	PipelineWorkers    int                  `json:"pipeline_workers"`
	LastSampleTime     time.Time            `json:"last_sample_time"`
}

// WorkerCoupler orchestrates dynamic runtime-capacity worker coupling.
type WorkerCoupler struct {
	mu     sync.RWMutex
	opMu   sync.Mutex // serializes model worker configuration during operations
	policy WorkerCouplingConfig

	sampler MetricsSampler

	lastSnapshot      RuntimeMetricsSnapshot
	lastSampleTime    time.Time
	lastPressureScore float64
	lastPressureClass RuntimePressureClass

	sampleCount        int64
	throttleEvents     int64
	lastAppliedWorkers int
}

// NewWorkerCoupler builds a coupler with the given configuration.
func NewWorkerCoupler(policy WorkerCouplingConfig) *WorkerCoupler {
	normalizePolicy(&policy)
	c := &WorkerCoupler{
		policy:            policy,
		sampler:           newRuntimeMetricsReader().Sample,
		lastPressureClass: PressureNominal,
	}
	c.sampleLocked(time.Now(), true)
	return c
}

// NewDefaultWorkerCoupler builds a coupler with environment or topology defaults.
func NewDefaultWorkerCoupler() *WorkerCoupler {
	return NewWorkerCoupler(WorkerCouplingConfigFromEnv())
}

func normalizePolicy(p *WorkerCouplingConfig) {
	procs := runtime.GOMAXPROCS(0)
	if procs < 1 {
		procs = 1
	}
	if p.MaxPrefillWorkers <= 0 {
		p.MaxPrefillWorkers = procs
	}
	if p.MaxDecodeWorkers <= 0 {
		p.MaxDecodeWorkers = decodeMaxWorkersFor(procs)
	}
	if p.MaxBatchWorkers <= 0 {
		p.MaxBatchWorkers = batchMaxWorkersFor(procs)
	}
	if p.MaxPipelineWorkers <= 0 {
		p.MaxPipelineWorkers = pipelineMaxWorkersFor(procs)
	}
	if p.MinWorkers < 1 {
		p.MinWorkers = 1
	}
	if p.SampleInterval < 0 {
		p.SampleInterval = 25 * time.Millisecond
	}
}

// Enabled reports whether dynamic worker coupling is active.
func (c *WorkerCoupler) Enabled() bool {
	if c == nil {
		return false
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.policy.Mode != CouplingModeDisabled
}

// SetConfig updates the coupler configuration.
func (c *WorkerCoupler) SetConfig(p WorkerCouplingConfig) {
	if c == nil {
		return
	}
	normalizePolicy(&p)
	c.mu.Lock()
	c.policy = p
	c.mu.Unlock()
}

// Config returns the current configuration.
func (c *WorkerCoupler) Config() WorkerCouplingConfig {
	if c == nil {
		return DefaultWorkerCouplingConfig()
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.policy
}

// SetSampler installs a custom scheduler metrics sampler (e.g. for testing).
func (c *WorkerCoupler) SetSampler(s MetricsSampler) {
	if c == nil {
		return
	}
	c.mu.Lock()
	if s == nil {
		s = newRuntimeMetricsReader().Sample
	}
	c.sampler = s
	c.mu.Unlock()
	c.SampleNow()
}

// SampleNow immediately forces a fresh metrics sample and recalculates budgets.
func (c *WorkerCoupler) SampleNow() RuntimeMetricsSnapshot {
	if c == nil {
		return RuntimeMetricsSnapshot{}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.sampleLocked(time.Now(), true)
}

func (c *WorkerCoupler) sampleLocked(now time.Time, force bool) RuntimeMetricsSnapshot {
	if !force && c.policy.SampleInterval > 0 && !c.lastSampleTime.IsZero() && now.Sub(c.lastSampleTime) < c.policy.SampleInterval {
		return c.lastSnapshot
	}
	var snap RuntimeMetricsSnapshot
	if c.sampler != nil {
		snap = c.sampler()
	} else {
		snap = RuntimeMetricsSnapshot{
			GOMAXPROCS: runtime.GOMAXPROCS(0),
			NumCPU:     runtime.NumCPU(),
			SampledAt:  now,
		}
	}
	if snap.GOMAXPROCS <= 0 {
		snap.GOMAXPROCS = runtime.GOMAXPROCS(0)
	}
	if snap.NumCPU <= 0 {
		snap.NumCPU = runtime.NumCPU()
	}
	if snap.SampledAt.IsZero() {
		snap.SampledAt = now
	}

	class, score := computePressureScore(snap)
	c.lastSnapshot = snap
	c.lastSampleTime = now
	c.lastPressureClass = class
	c.lastPressureScore = score
	c.sampleCount++
	return snap
}

// CurrentPressure reports the current scheduler pressure class and continuous score [0.0, 1.0].
func (c *WorkerCoupler) CurrentPressure() (RuntimePressureClass, float64) {
	if c == nil {
		return PressureNominal, 0.0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.sampleLocked(time.Now(), false)
	return c.lastPressureClass, c.lastPressureScore
}

// WorkersFor returns the active worker budget for the specified operation kind.
func (c *WorkerCoupler) WorkersFor(op WorkerOpKind) int {
	if c == nil {
		return runtime.GOMAXPROCS(0)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.sampleLocked(time.Now(), false)
	return c.workersForOpLocked(op)
}

func (c *WorkerCoupler) maxWorkersForOpLocked(op WorkerOpKind) int {
	switch op {
	case OpPrefill:
		return c.policy.MaxPrefillWorkers
	case OpDecode:
		return c.policy.MaxDecodeWorkers
	case OpBatch:
		return c.policy.MaxBatchWorkers
	case OpPipeline:
		return c.policy.MaxPipelineWorkers
	default:
		return runtime.GOMAXPROCS(0)
	}
}

func (c *WorkerCoupler) workersForOpLocked(op WorkerOpKind) int {
	maxW := c.maxWorkersForOpLocked(op)
	minW := c.policy.MinWorkers
	if minW < 1 {
		minW = 1
	}
	if maxW < minW {
		maxW = minW
	}

	if c.policy.Mode != CouplingModeDynamic {
		return maxW
	}

	score := c.lastPressureScore
	var factor float64

	switch op {
	case OpPrefill:
		// Compute-heavy GEMM: scales smoothly with pressure.
		switch {
		case score < 0.20:
			factor = 1.00
		case score < 0.45:
			factor = 0.85
		case score < 0.70:
			factor = 0.60
		case score < 0.85:
			factor = 0.35
		default:
			factor = 0.20
		}
		w := int(math.Round(float64(maxW) * factor))
		if w < minW {
			w = minW
		}
		if w > maxW {
			w = maxW
		}
		return w

	case OpDecode:
		// Memory-bandwidth-bound single-token GEMV: sensitive to barrier stalls, clamps aggressively.
		switch {
		case score < 0.20:
			factor = 1.00
		case score < 0.45:
			factor = 0.75
		case score < 0.70:
			factor = 0.50
		case score < 0.85:
			// Under high contention, clamp to min(2, maxW).
			w := 2
			if w > maxW {
				w = maxW
			}
			if w < minW {
				w = minW
			}
			return w
		default:
			// Critical contention: single worker avoids parFor barrier entirely.
			return minW
		}
		w := int(math.Round(float64(maxW) * factor))
		if w < minW {
			w = minW
		}
		if w > maxW {
			w = maxW
		}
		return w

	case OpBatch:
		// Multi-lane StepBatch: intermediate compute intensity.
		switch {
		case score < 0.20:
			factor = 1.00
		case score < 0.45:
			factor = 0.80
		case score < 0.70:
			factor = 0.55
		case score < 0.85:
			factor = 0.30
		default:
			factor = 0.15
		}
		w := int(math.Round(float64(maxW) * factor))
		if w < minW {
			w = minW
		}
		if w > maxW {
			w = maxW
		}
		return w

	case OpPipeline:
		// Pipeline stage forward dispatch.
		switch {
		case score < 0.20:
			factor = 1.00
		case score < 0.45:
			factor = 0.80
		case score < 0.70:
			factor = 0.50
		case score < 0.85:
			factor = 0.30
		default:
			factor = 0.15
		}
		w := int(math.Round(float64(maxW) * factor))
		if w < minW {
			w = minW
		}
		if w > maxW {
			w = maxW
		}
		return w

	default:
		return maxW
	}
}

// ActiveWorkerBudget returns the current worker budget for (prefill, decode, batch, pipeline).
func (c *WorkerCoupler) ActiveWorkerBudget() (prefill, decode, batch, pipeline int) {
	if c == nil {
		procs := runtime.GOMAXPROCS(0)
		return procs, procs, procs, procs
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.sampleLocked(time.Now(), false)
	return c.workersForOpLocked(OpPrefill),
		c.workersForOpLocked(OpDecode),
		c.workersForOpLocked(OpBatch),
		c.workersForOpLocked(OpPipeline)
}

// IsThrottled reports whether any operation kind is currently throttled below its maximum.
func (c *WorkerCoupler) IsThrottled() bool {
	if c == nil {
		return false
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.sampleLocked(time.Now(), false)
	return c.workersForOpLocked(OpPrefill) < c.maxWorkersForOpLocked(OpPrefill) ||
		c.workersForOpLocked(OpDecode) < c.maxWorkersForOpLocked(OpDecode)
}

// CoupledMaxRunning computes the dynamic running-set cap for NativeScheduler admission.
func (c *WorkerCoupler) CoupledMaxRunning(configuredMaxRunning int) int {
	if c == nil {
		return configuredMaxRunning
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.sampleLocked(time.Now(), false)

	if c.policy.Mode != CouplingModeDynamic {
		return configuredMaxRunning
	}

	score := c.lastPressureScore
	procs := c.lastSnapshot.GOMAXPROCS
	if procs <= 0 {
		procs = runtime.GOMAXPROCS(0)
	}

	if configuredMaxRunning > 0 {
		switch {
		case score < 0.45:
			return configuredMaxRunning
		case score < 0.70:
			w := (configuredMaxRunning * 3) / 4
			if w < 1 {
				w = 1
			}
			return w
		case score < 0.85:
			w := configuredMaxRunning / 2
			if w < 1 {
				w = 1
			}
			return w
		default:
			w := configuredMaxRunning / 4
			if w < 1 {
				w = 1
			}
			return w
		}
	}

	// For unbounded (configuredMaxRunning == 0), impose protective upper bounds under severe pressure.
	switch {
	case score < 0.70:
		return 0 // unbounded under nominal / light / moderate
	case score < 0.85:
		cap := procs * 4
		if cap < 2 {
			cap = 2
		}
		return cap
	default:
		cap := procs * 2
		if cap < 1 {
			cap = 1
		}
		return cap
	}
}

// WithOp applies dynamic worker coupling for the duration of the supplied forward function.
func (c *WorkerCoupler) WithOp(op WorkerOpKind, fn func()) {
	if c == nil || !c.Enabled() {
		fn()
		return
	}
	c.opMu.Lock()
	defer c.opMu.Unlock()

	c.mu.Lock()
	c.sampleLocked(time.Now(), false)
	targetWorkers := c.workersForOpLocked(op)
	maxW := c.maxWorkersForOpLocked(op)
	if targetWorkers < maxW {
		c.throttleEvents++
	}
	if targetWorkers != c.lastAppliedWorkers {
		_ = model.SetWorkers(targetWorkers)
		c.lastAppliedWorkers = targetWorkers
	}
	c.mu.Unlock()

	fn()
}

// RunWithOp wraps a value-returning function with dynamic worker coupling.
func RunWithOp[T any](c *WorkerCoupler, op WorkerOpKind, fn func() T) T {
	if c == nil || !c.Enabled() {
		return fn()
	}
	var res T
	c.WithOp(op, func() {
		res = fn()
	})
	return res
}

// RunWithOpErr wraps an error-returning function with dynamic worker coupling.
func RunWithOpErr[T any](c *WorkerCoupler, op WorkerOpKind, fn func() (T, error)) (T, error) {
	if c == nil || !c.Enabled() {
		return fn()
	}
	var (
		res T
		err error
	)
	c.WithOp(op, func() {
		res, err = fn()
	})
	return res, err
}

// Stats snapshots telemetry for this coupler.
func (c *WorkerCoupler) Stats() WorkerCouplingStats {
	if c == nil {
		return WorkerCouplingStats{
			Enabled: false,
			Mode:    CouplingModeDisabled.String(),
		}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.sampleLocked(time.Now(), false)

	return WorkerCouplingStats{
		Enabled:            c.policy.Mode != CouplingModeDisabled,
		Mode:               c.policy.Mode.String(),
		GOMAXPROCS:         c.lastSnapshot.GOMAXPROCS,
		NumCPU:             c.lastSnapshot.NumCPU,
		PressureClass:      c.lastPressureClass,
		PressureScore:      c.lastPressureScore,
		RunnableGoroutines: c.lastSnapshot.RunnableGoroutines,
		SchedLatencyP50:    c.lastSnapshot.SchedLatencyP50,
		SchedLatencyP95:    c.lastSnapshot.SchedLatencyP95,
		CPUIdleFraction:    c.lastSnapshot.CPUIdleFraction,
		SampleCount:        c.sampleCount,
		ThrottleEvents:     c.throttleEvents,
		PrefillWorkers:     c.workersForOpLocked(OpPrefill),
		DecodeWorkers:      c.workersForOpLocked(OpDecode),
		BatchWorkers:       c.workersForOpLocked(OpBatch),
		PipelineWorkers:    c.workersForOpLocked(OpPipeline),
		LastSampleTime:     c.lastSampleTime,
	}
}

// WriteMetrics renders Prometheus-format metrics for the active worker coupling state.
func (c *WorkerCoupler) WriteMetrics(b *strings.Builder) {
	if c == nil || b == nil {
		return
	}
	st := c.Stats()
	const p = "fak_worker_coupling_"
	writeNativeHelpType(b, p+"pressure_score", "Current Go runtime scheduler contention score in range [0.0, 1.0].", "gauge")
	fmt.Fprintf(b, "%spressure_score %.4f\n", p, st.PressureScore)

	writeNativeHelpType(b, p+"runnable_goroutines", "Current count of runnable goroutines on Go runtime run queues.", "gauge")
	fmt.Fprintf(b, "%srunnable_goroutines %d\n", p, st.RunnableGoroutines)

	writeNativeHelpType(b, p+"sched_latency_p95_seconds", "P95 Go runtime goroutine scheduler latency in seconds.", "gauge")
	fmt.Fprintf(b, "%ssched_latency_p95_seconds %.6f\n", p, st.SchedLatencyP95.Seconds())

	writeNativeHelpType(b, p+"sched_latency_p50_seconds", "P50 Go runtime goroutine scheduler latency in seconds.", "gauge")
	fmt.Fprintf(b, "%ssched_latency_p50_seconds %.6f\n", p, st.SchedLatencyP50.Seconds())

	writeNativeHelpType(b, p+"cpu_idle_fraction", "Recent CPU idle time fraction in range [0.0, 1.0].", "gauge")
	fmt.Fprintf(b, "%scpu_idle_fraction %.4f\n", p, st.CPUIdleFraction)

	writeNativeHelpType(b, p+"active_workers", "Active parallel worker allocations by operation kind.", "gauge")
	fmt.Fprintf(b, "%sactive_workers{op=\"prefill\"} %d\n", p, st.PrefillWorkers)
	fmt.Fprintf(b, "%sactive_workers{op=\"decode\"} %d\n", p, st.DecodeWorkers)
	fmt.Fprintf(b, "%sactive_workers{op=\"batch\"} %d\n", p, st.BatchWorkers)
	fmt.Fprintf(b, "%sactive_workers{op=\"pipeline\"} %d\n", p, st.PipelineWorkers)

	writeNativeCounter(b, p+"samples_total", "Cumulative scheduler metrics samples taken.", st.SampleCount)
	writeNativeCounter(b, p+"throttle_events_total", "Cumulative parallel worker throttle events under scheduler pressure.", st.ThrottleEvents)
}

// Internal hardware defaults.
func decodeMaxWorkersFor(procs int) int {
	if procs <= 1 {
		return 1
	}
	if runtime.GOOS == "darwin" && runtime.GOARCH == "arm64" && procs >= 8 {
		w := (procs + 1) / 2
		if w > 6 {
			w = 6
		}
		return w
	}
	if runtime.GOARCH == "amd64" && procs >= 64 {
		w := procs / 8
		if w > 16 {
			w = 16
		}
		if w < 4 {
			w = 4
		}
		return w
	}
	w := (procs + 1) / 2
	if w > 6 {
		w = 6
	}
	if w < 1 {
		w = 1
	}
	return w
}

func batchMaxWorkersFor(procs int) int {
	if procs <= 1 {
		return 1
	}
	w := (procs * 3) / 4
	if w < 2 {
		w = 2
	}
	if w > procs {
		w = procs
	}
	return w
}

func pipelineMaxWorkersFor(procs int) int {
	if procs <= 1 {
		return 1
	}
	w := procs / 2
	if w < 1 {
		w = 1
	}
	return w
}

// computePressureScore evaluates normalized pressure from metrics.
func computePressureScore(snap RuntimeMetricsSnapshot) (RuntimePressureClass, float64) {
	procs := snap.GOMAXPROCS
	if procs <= 0 {
		procs = runtime.GOMAXPROCS(0)
	}
	if procs <= 0 {
		procs = 1
	}

	// 1. Runnable Goroutines Contention:
	runnableRatio := float64(snap.RunnableGoroutines) / float64(procs)
	var runnableScore float64
	if runnableRatio <= 0.25 {
		runnableScore = runnableRatio * 0.4
	} else if runnableRatio <= 1.0 {
		runnableScore = 0.10 + (runnableRatio-0.25)*0.40
	} else if runnableRatio <= 2.5 {
		runnableScore = 0.40 + (runnableRatio-1.00)*0.30
	} else {
		runnableScore = math.Min(1.0, 0.85+(runnableRatio-2.50)*0.10)
	}

	// 2. Scheduler Latency:
	var latencyScore float64
	latMS := float64(snap.SchedLatencyP95) / float64(time.Millisecond)
	if latMS <= 0.1 {
		latencyScore = 0.0
	} else if latMS <= 1.0 {
		latencyScore = latMS * 0.25
	} else if latMS <= 5.0 {
		latencyScore = 0.25 + (latMS-1.0)/4.0*0.25
	} else if latMS <= 15.0 {
		latencyScore = 0.50 + (latMS-5.0)/10.0*0.30
	} else {
		latencyScore = math.Min(1.0, 0.80+(latMS-15.0)/15.0*0.20)
	}

	// 3. CPU Idle Fraction:
	var cpuScore float64
	hasCPU := snap.CPUIdleFraction >= 0
	if hasCPU {
		cpuScore = 1.0 - snap.CPUIdleFraction
		if cpuScore < 0 {
			cpuScore = 0
		} else if cpuScore > 1 {
			cpuScore = 1
		}
	}

	var score float64
	if hasCPU {
		score = 0.50*runnableScore + 0.30*latencyScore + 0.20*cpuScore
	} else {
		score = 0.65*runnableScore + 0.35*latencyScore
	}
	score = math.Max(0.0, math.Min(1.0, score))

	var class RuntimePressureClass
	switch {
	case score < 0.20:
		class = PressureNominal
	case score < 0.45:
		class = PressureLight
	case score < 0.70:
		class = PressureModerate
	case score < 0.85:
		class = PressureHigh
	default:
		class = PressureCritical
	}
	return class, score
}

// runtimeMetricsReader provides stateful Go 1.26 runtime/metrics sampling.
type runtimeMetricsReader struct {
	mu         sync.Mutex
	prevIdle   float64
	prevTotal  float64
	prevCounts []uint64
	prevTime   time.Time
}

func newRuntimeMetricsReader() *runtimeMetricsReader {
	return &runtimeMetricsReader{
		prevTime: time.Now(),
	}
}

const (
	metricSchedLatencies  = "/sched/latencies:seconds"
	metricSchedRunnable   = "/sched/goroutines:runnable"
	metricSchedGOMAXPROCS = "/sched/gomaxprocs:threads"
	metricCPUIdle         = "/cpu/classes/idle:cpu-seconds"
	metricCPUTotal        = "/cpu/classes/total:cpu-seconds"
)

func (r *runtimeMetricsReader) Sample() RuntimeMetricsSnapshot {
	samples := []rmetrics.Sample{
		{Name: metricSchedLatencies},
		{Name: metricSchedRunnable},
		{Name: metricSchedGOMAXPROCS},
		{Name: metricCPUIdle},
		{Name: metricCPUTotal},
	}
	rmetrics.Read(samples)

	now := time.Now()
	snap := RuntimeMetricsSnapshot{
		GOMAXPROCS: runtime.GOMAXPROCS(0),
		NumCPU:     runtime.NumCPU(),
		SampledAt:  now,
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	for _, sample := range samples {
		switch sample.Name {
		case metricSchedRunnable:
			if sample.Value.Kind() == rmetrics.KindUint64 {
				snap.RunnableGoroutines = int64(sample.Value.Uint64())
			}
		case metricSchedGOMAXPROCS:
			if sample.Value.Kind() == rmetrics.KindUint64 {
				procs := int(sample.Value.Uint64())
				if procs > 0 {
					snap.GOMAXPROCS = procs
				}
			}
		case metricCPUIdle:
			if sample.Value.Kind() == rmetrics.KindFloat64 {
				idle := sample.Value.Float64()
				for _, other := range samples {
					if other.Name == metricCPUTotal && other.Value.Kind() == rmetrics.KindFloat64 {
						total := other.Value.Float64()
						dIdle := idle - r.prevIdle
						dTotal := total - r.prevTotal
						if dTotal > 0 && dIdle >= 0 {
							frac := dIdle / dTotal
							if frac > 1.0 {
								frac = 1.0
							}
							snap.CPUIdleFraction = frac
						}
						r.prevIdle = idle
						r.prevTotal = total
						break
					}
				}
			}
		case metricSchedLatencies:
			if sample.Value.Kind() == rmetrics.KindFloat64Histogram {
				h := sample.Value.Float64Histogram()
				if h != nil && len(h.Counts) > 0 {
					snap.SchedLatencyP50 = histogramQuantile(h, r.prevCounts, 0.50)
					snap.SchedLatencyP95 = histogramQuantile(h, r.prevCounts, 0.95)
					r.prevCounts = append([]uint64(nil), h.Counts...)
				}
			}
		}
	}
	r.prevTime = now
	return snap
}

func histogramQuantile(h *rmetrics.Float64Histogram, prevCounts []uint64, q float64) time.Duration {
	if h == nil || len(h.Counts) == 0 || len(h.Buckets) < 2 {
		return 0
	}
	// Use delta counts if available to isolate interval latency.
	counts := h.Counts
	useDelta := len(prevCounts) == len(h.Counts)
	var total uint64
	if useDelta {
		deltaCounts := make([]uint64, len(h.Counts))
		var deltaTotal uint64
		for i := range h.Counts {
			if h.Counts[i] >= prevCounts[i] {
				deltaCounts[i] = h.Counts[i] - prevCounts[i]
			}
			deltaTotal += deltaCounts[i]
		}
		if deltaTotal > 0 {
			counts = deltaCounts
			total = deltaTotal
		} else {
			useDelta = false
		}
	}
	if !useDelta {
		for _, c := range h.Counts {
			total += c
		}
	}
	if total == 0 {
		return 0
	}

	target := float64(total) * q
	var cum uint64
	for i, count := range counts {
		if count == 0 {
			continue
		}
		cum += count
		if float64(cum) >= target {
			lo := h.Buckets[i]
			hi := h.Buckets[i+1]
			if math.IsInf(lo, -1) {
				lo = 0
			}
			if math.IsInf(hi, 1) {
				if lo < 0 {
					lo = 0
				}
				return time.Duration(lo * float64(time.Second))
			}
			prev := float64(cum - count)
			frac := 0.0
			if count > 0 {
				frac = (target - prev) / float64(count)
			}
			sec := lo + frac*(hi-lo)
			if sec < 0 {
				sec = 0
			}
			return time.Duration(sec * float64(time.Second))
		}
	}
	last := h.Buckets[len(h.Buckets)-1]
	if math.IsInf(last, 1) {
		last = h.Buckets[len(h.Buckets)-2]
	}
	if last < 0 {
		last = 0
	}
	return time.Duration(last * float64(time.Second))
}
