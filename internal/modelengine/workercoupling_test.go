package modelengine

import (
	"context"
	"fmt"
	"math"
	"runtime"
	rmetrics "runtime/metrics"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestWorkerCouplerConfigFromEnv(t *testing.T) {
	t.Setenv("FAK_WORKER_COUPLING", "dynamic")
	t.Setenv("FAK_WORKER_COUPLING_PREFILL_MAX", "12")
	t.Setenv("FAK_WORKER_COUPLING_DECODE_MAX", "4")
	t.Setenv("FAK_WORKER_COUPLING_MIN", "2")
	t.Setenv("FAK_WORKER_COUPLING_INTERVAL_MS", "50")

	p := WorkerCouplingConfigFromEnv()
	if p.Mode != CouplingModeDynamic {
		t.Fatalf("Mode = %v, want %v", p.Mode, CouplingModeDynamic)
	}
	if p.MaxPrefillWorkers != 12 {
		t.Fatalf("MaxPrefillWorkers = %d, want 12", p.MaxPrefillWorkers)
	}
	if p.MaxDecodeWorkers != 4 {
		t.Fatalf("MaxDecodeWorkers = %d, want 4", p.MaxDecodeWorkers)
	}
	if p.MinWorkers != 2 {
		t.Fatalf("MinWorkers = %d, want 2", p.MinWorkers)
	}
	if p.SampleInterval != 50*time.Millisecond {
		t.Fatalf("SampleInterval = %v, want 50ms", p.SampleInterval)
	}
}

func TestRuntimeMetricsSamplingLive(t *testing.T) {
	reader := newRuntimeMetricsReader()
	snap := reader.Sample()

	if snap.GOMAXPROCS <= 0 {
		t.Fatalf("snap.GOMAXPROCS = %d, want > 0", snap.GOMAXPROCS)
	}
	if snap.NumCPU <= 0 {
		t.Fatalf("snap.NumCPU = %d, want > 0", snap.NumCPU)
	}
	if snap.SampledAt.IsZero() {
		t.Fatalf("snap.SampledAt is zero")
	}
	if snap.RunnableGoroutines < 0 {
		t.Fatalf("snap.RunnableGoroutines = %d, want >= 0", snap.RunnableGoroutines)
	}
}

func TestHistogramQuantileMath(t *testing.T) {
	// 4 buckets: [0, 0.001), [0.001, 0.010), [0.010, 0.100), [0.100, +Inf)
	h := &rmetrics.Float64Histogram{
		Buckets: []float64{0.0, 0.001, 0.010, 0.100, math.Inf(1)},
		Counts:  []uint64{100, 80, 15, 5},
	}
	// Total count = 200.
	// P50 is target 100 -> bucket 0 (at boundary or within [0, 0.001)).
	p50 := histogramQuantile(h, nil, 0.50)
	if p50 <= 0 || p50 > time.Millisecond {
		t.Fatalf("p50 = %v, expected in (0, 1ms]", p50)
	}

	// P95 is target 190. Buckets: 100 + 80 + 10 = 190 -> inside bucket 2 [0.010, 0.100).
	p95 := histogramQuantile(h, nil, 0.95)
	if p95 < 10*time.Millisecond || p95 > 100*time.Millisecond {
		t.Fatalf("p95 = %v, expected in [10ms, 100ms]", p95)
	}

	// Test delta handling.
	prevCounts := []uint64{100, 80, 15, 5}
	h2 := &rmetrics.Float64Histogram{
		Buckets: []float64{0.0, 0.001, 0.010, 0.100, math.Inf(1)},
		Counts:  []uint64{110, 80, 15, 5}, // 10 new samples in [0, 0.001)
	}
	pDelta := histogramQuantile(h2, prevCounts, 0.50)
	if pDelta <= 0 || pDelta > time.Millisecond {
		t.Fatalf("pDelta = %v, expected in (0, 1ms]", pDelta)
	}
}

func TestWorkerCouplerPressureClassification(t *testing.T) {
	tests := []struct {
		name      string
		snap      RuntimeMetricsSnapshot
		wantClass RuntimePressureClass
	}{
		{
			name: "nominal",
			snap: RuntimeMetricsSnapshot{
				RunnableGoroutines: 0,
				GOMAXPROCS:         16,
				NumCPU:             16,
				SchedLatencyP95:    50 * time.Microsecond,
				CPUIdleFraction:    0.95,
			},
			wantClass: PressureNominal,
		},
		{
			name: "light",
			snap: RuntimeMetricsSnapshot{
				RunnableGoroutines: 8,
				GOMAXPROCS:         16,
				NumCPU:             16,
				SchedLatencyP95:    800 * time.Microsecond,
				CPUIdleFraction:    0.70,
			},
			wantClass: PressureLight,
		},
		{
			name: "moderate",
			snap: RuntimeMetricsSnapshot{
				RunnableGoroutines: 24,
				GOMAXPROCS:         16,
				NumCPU:             16,
				SchedLatencyP95:    4 * time.Millisecond,
				CPUIdleFraction:    0.35,
			},
			wantClass: PressureModerate,
		},
		{
			name: "high",
			snap: RuntimeMetricsSnapshot{
				RunnableGoroutines: 40,
				GOMAXPROCS:         16,
				NumCPU:             16,
				SchedLatencyP95:    12 * time.Millisecond,
				CPUIdleFraction:    0.10,
			},
			wantClass: PressureHigh,
		},
		{
			name: "critical",
			snap: RuntimeMetricsSnapshot{
				RunnableGoroutines: 80,
				GOMAXPROCS:         16,
				NumCPU:             16,
				SchedLatencyP95:    35 * time.Millisecond,
				CPUIdleFraction:    0.01,
			},
			wantClass: PressureCritical,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			class, score := computePressureScore(tc.snap)
			if class != tc.wantClass {
				t.Fatalf("computePressureScore() class = %v (score = %.4f), want %v", class, score, tc.wantClass)
			}
		})
	}
}

func TestWorkerCouplerOperationDifferentiation(t *testing.T) {
	// High-core host: 16 cores.
	// Prefill max: 16 workers.
	// Decode max: 6 workers (memory bandwidth ceiling).
	policy := WorkerCouplingConfig{
		Mode:               CouplingModeDynamic,
		MaxPrefillWorkers:  16,
		MaxDecodeWorkers:   6,
		MaxBatchWorkers:    12,
		MaxPipelineWorkers: 8,
		MinWorkers:         1,
		SampleInterval:     0, // sample on every call for deterministic testing
	}
	coupler := NewWorkerCoupler(policy)

	// Step 1: Nominal pressure.
	coupler.SetSampler(func() RuntimeMetricsSnapshot {
		return RuntimeMetricsSnapshot{
			RunnableGoroutines: 0,
			GOMAXPROCS:         16,
			NumCPU:             16,
			SchedLatencyP95:    50 * time.Microsecond,
			CPUIdleFraction:    0.95,
			SampledAt:          time.Now(),
		}
	})

	prefillNominal := coupler.WorkersFor(OpPrefill)
	decodeNominal := coupler.WorkersFor(OpDecode)
	batchNominal := coupler.WorkersFor(OpBatch)
	pipelineNominal := coupler.WorkersFor(OpPipeline)

	if prefillNominal != 16 {
		t.Fatalf("prefillNominal = %d, want 16", prefillNominal)
	}
	if decodeNominal != 6 {
		t.Fatalf("decodeNominal = %d, want 6 (memory-bound ceiling)", decodeNominal)
	}
	if batchNominal != 12 {
		t.Fatalf("batchNominal = %d, want 12", batchNominal)
	}
	if pipelineNominal != 8 {
		t.Fatalf("pipelineNominal = %d, want 8", pipelineNominal)
	}
	if prefillNominal == decodeNominal {
		t.Fatalf("prefill (%d) and decode (%d) must be differentiated", prefillNominal, decodeNominal)
	}
	if coupler.IsThrottled() {
		t.Fatalf("IsThrottled() = true under nominal pressure, want false")
	}

	// Step 2: High pressure (runnable backlog = 40 goroutines).
	coupler.SetSampler(func() RuntimeMetricsSnapshot {
		return RuntimeMetricsSnapshot{
			RunnableGoroutines: 40,
			GOMAXPROCS:         16,
			NumCPU:             16,
			SchedLatencyP95:    12 * time.Millisecond,
			CPUIdleFraction:    0.10,
			SampledAt:          time.Now(),
		}
	})

	prefillHigh := coupler.WorkersFor(OpPrefill)
	decodeHigh := coupler.WorkersFor(OpDecode)

	// Prefill scaled down: 16 * 0.35 ~= 6
	if prefillHigh >= prefillNominal {
		t.Fatalf("prefillHigh = %d must be scaled down from prefillNominal = %d", prefillHigh, prefillNominal)
	}
	// Decode clamped aggressively to 2 workers to prevent barrier collapse.
	if decodeHigh > 2 {
		t.Fatalf("decodeHigh = %d, want <= 2 under high pressure", decodeHigh)
	}
	if !coupler.IsThrottled() {
		t.Fatalf("IsThrottled() = false under high pressure, want true")
	}

	// Step 3: Critical pressure (severe run-queue thrash = 80 goroutines).
	coupler.SetSampler(func() RuntimeMetricsSnapshot {
		return RuntimeMetricsSnapshot{
			RunnableGoroutines: 80,
			GOMAXPROCS:         16,
			NumCPU:             16,
			SchedLatencyP95:    35 * time.Millisecond,
			CPUIdleFraction:    0.01,
			SampledAt:          time.Now(),
		}
	})

	prefillCritical := coupler.WorkersFor(OpPrefill)
	decodeCritical := coupler.WorkersFor(OpDecode)

	if prefillCritical > prefillHigh {
		t.Fatalf("prefillCritical = %d, want <= prefillHigh = %d", prefillCritical, prefillHigh)
	}
	if decodeCritical != 1 {
		t.Fatalf("decodeCritical = %d, want 1 (single-worker serialization under critical pressure)", decodeCritical)
	}
}

func TestWorkerCouplerThrottleScaleAndCap(t *testing.T) {
	policy := WorkerCouplingConfig{
		Mode:              CouplingModeDynamic,
		MaxPrefillWorkers: 16,
		MaxDecodeWorkers:  6,
		MinWorkers:        1,
		SampleInterval:    0,
	}
	coupler := NewWorkerCoupler(policy)

	// Under critical pressure, test CoupledMaxRunning.
	coupler.SetSampler(func() RuntimeMetricsSnapshot {
		return RuntimeMetricsSnapshot{
			RunnableGoroutines: 90,
			GOMAXPROCS:         16,
			NumCPU:             16,
			SchedLatencyP95:    40 * time.Millisecond,
			CPUIdleFraction:    0.0,
			SampledAt:          time.Now(),
		}
	})

	// Configured max running = 8 -> under critical pressure, scales to 8 / 4 = 2.
	cappedRun := coupler.CoupledMaxRunning(8)
	if cappedRun != 2 {
		t.Fatalf("CoupledMaxRunning(8) = %d, want 2", cappedRun)
	}

	// Unbounded (0) -> under critical pressure, bounded to 2 * GOMAXPROCS = 32.
	unboundedCapped := coupler.CoupledMaxRunning(0)
	if unboundedCapped != 32 {
		t.Fatalf("CoupledMaxRunning(0) = %d, want 32", unboundedCapped)
	}
}

func TestWorkerCouplerWithOpAndModelWorkers(t *testing.T) {
	policy := WorkerCouplingConfig{
		Mode:              CouplingModeDynamic,
		MaxPrefillWorkers: 8,
		MaxDecodeWorkers:  2,
		MinWorkers:        1,
		SampleInterval:    0,
	}
	coupler := NewWorkerCoupler(policy)
	coupler.SetSampler(func() RuntimeMetricsSnapshot {
		return RuntimeMetricsSnapshot{
			RunnableGoroutines: 0,
			GOMAXPROCS:         16,
			NumCPU:             16,
			SchedLatencyP95:    50 * time.Microsecond,
			CPUIdleFraction:    0.95,
			SampledAt:          time.Now(),
		}
	})

	var recordedPrefillWorkers int
	coupler.WithOp(OpPrefill, func() {
		recordedPrefillWorkers = model.NumWorkers()
	})
	if recordedPrefillWorkers != 8 {
		t.Fatalf("recordedPrefillWorkers = %d, want 8", recordedPrefillWorkers)
	}

	var recordedDecodeWorkers int
	coupler.WithOp(OpDecode, func() {
		recordedDecodeWorkers = model.NumWorkers()
	})
	if recordedDecodeWorkers != 2 {
		t.Fatalf("recordedDecodeWorkers = %d, want 2", recordedDecodeWorkers)
	}

	// RunWithOp generic test.
	val := RunWithOp(coupler, OpDecode, func() string {
		return fmt.Sprintf("workers=%d", model.NumWorkers())
	})
	if val != "workers=2" {
		t.Fatalf("RunWithOp val = %q, want workers=2", val)
	}

	// RunWithOpErr generic test.
	valErr, err := RunWithOpErr(coupler, OpPrefill, func() (int, error) {
		return model.NumWorkers(), nil
	})
	if err != nil || valErr != 8 {
		t.Fatalf("RunWithOpErr = (%d, %v), want (8, nil)", valErr, err)
	}
}

func TestNativeSchedulerWithDynamicCoupling(t *testing.T) {
	cfg := SyntheticConfig()
	m := model.NewSynthetic(cfg)
	sched := NewNativeScheduler(m)
	defer sched.Close()

	coupler := sched.WorkerCoupler()
	if coupler == nil {
		t.Fatal("scheduler must have non-nil WorkerCoupler")
	}

	// Set sample interval to 0 for instantaneous test evaluation.
	p := coupler.Config()
	p.SampleInterval = 0
	coupler.SetConfig(p)

	ctx := context.Background()
	call := inlineCall("search_flights", `{"from":"SFO","to":"JFK"}`)
	req, err := sched.Admit(ctx, call)
	if err != nil {
		t.Fatalf("Admit: %v", err)
	}

	var tokens []int
	for tok := range req.Tokens() {
		tokens = append(tokens, tok.ID)
	}

	res, err := req.Result()
	if err != nil {
		t.Fatalf("Result: %v", err)
	}
	if res == nil || res.Status != abi.StatusOK {
		t.Fatalf("Result = %+v, want StatusOK", res)
	}
	if len(tokens) != genTokens {
		t.Fatalf("streamed %d tokens, want %d", len(tokens), genTokens)
	}

	// Verify telemetry was recorded.
	st := sched.WorkerCouplingStats()
	if !st.Enabled {
		t.Fatal("WorkerCouplingStats.Enabled = false, want true")
	}
	if st.SampleCount <= 0 {
		t.Fatalf("WorkerCouplingStats.SampleCount = %d, want > 0", st.SampleCount)
	}
}

func TestEngineWorkerCouplingSeams(t *testing.T) {
	eng := New()
	coupler := eng.WorkerCoupler()
	if coupler == nil {
		t.Fatal("engine.WorkerCoupler() is nil")
	}

	stats := eng.WorkerCouplingStats()
	if !stats.Enabled {
		t.Fatal("eng.WorkerCouplingStats().Enabled is false")
	}

	var b strings.Builder
	eng.WriteWorkerCouplingMetrics(&b)
	out := b.String()

	if !strings.Contains(out, "fak_worker_coupling_pressure_score") {
		t.Fatalf("missing pressure_score metric:\n%s", out)
	}
	if !strings.Contains(out, "fak_worker_coupling_active_workers{op=\"prefill\"}") {
		t.Fatalf("missing prefill active_workers metric:\n%s", out)
	}
	if !strings.Contains(out, "fak_worker_coupling_active_workers{op=\"decode\"}") {
		t.Fatalf("missing decode active_workers metric:\n%s", out)
	}
}

func TestWorkerCouplingWriteMetricsFormatting(t *testing.T) {
	coupler := NewDefaultWorkerCoupler()
	var b strings.Builder
	coupler.WriteMetrics(&b)
	metrics := b.String()

	requiredSubstrings := []string{
		"# HELP fak_worker_coupling_pressure_score",
		"# TYPE fak_worker_coupling_pressure_score gauge",
		"fak_worker_coupling_pressure_score",
		"fak_worker_coupling_runnable_goroutines",
		"fak_worker_coupling_sched_latency_p95_seconds",
		"fak_worker_coupling_sched_latency_p50_seconds",
		"fak_worker_coupling_cpu_idle_fraction",
		"fak_worker_coupling_active_workers{op=\"prefill\"}",
		"fak_worker_coupling_active_workers{op=\"decode\"}",
		"fak_worker_coupling_active_workers{op=\"batch\"}",
		"fak_worker_coupling_active_workers{op=\"pipeline\"}",
		"fak_worker_coupling_samples_total",
		"fak_worker_coupling_throttle_events_total",
	}

	for _, sub := range requiredSubstrings {
		if !strings.Contains(metrics, sub) {
			t.Errorf("metrics output missing expected substring %q:\n%s", sub, metrics)
		}
	}
}

func TestWorkerCouplerConcurrentAccessRace(t *testing.T) {
	policy := DefaultWorkerCouplingConfig()
	policy.SampleInterval = time.Millisecond
	coupler := NewWorkerCoupler(policy)

	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()

	var wg sync.WaitGroup
	const goroutines = 8

	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			op := WorkerOpKind(id % 4)
			for {
				select {
				case <-ctx.Done():
					return
				default:
					coupler.WithOp(op, func() {
						_ = coupler.WorkersFor(op)
						_ = coupler.CurrentPressure
						_ = coupler.Stats()
						_ = coupler.IsThrottled()
					})
				}
			}
		}(i)
	}

	// Background sampler mutator.
	wg.Add(1)
	go func() {
		defer wg.Done()
		tick := time.NewTicker(2 * time.Millisecond)
		defer tick.Stop()
		i := 0
		for {
			select {
			case <-ctx.Done():
				return
			case <-tick.C:
				i++
				val := i
				coupler.SetSampler(func() RuntimeMetricsSnapshot {
					return RuntimeMetricsSnapshot{
						RunnableGoroutines: int64(val % 50),
						GOMAXPROCS:         runtime.GOMAXPROCS(0),
						NumCPU:             runtime.NumCPU(),
						SchedLatencyP95:    time.Duration(val%20) * time.Millisecond,
						CPUIdleFraction:    0.5,
						SampledAt:          time.Now(),
					}
				})
			}
		}
	}()

	wg.Wait()
}

// Benchmarks

func BenchmarkWorkerCouplerWorkersFor(b *testing.B) {
	coupler := NewDefaultWorkerCoupler()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = coupler.WorkersFor(OpPrefill)
		_ = coupler.WorkersFor(OpDecode)
	}
}

func BenchmarkWorkerCouplerSample(b *testing.B) {
	coupler := NewDefaultWorkerCoupler()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = coupler.SampleNow()
	}
}

func BenchmarkWorkerCouplerWithOp(b *testing.B) {
	coupler := NewDefaultWorkerCoupler()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		coupler.WithOp(OpDecode, func() {
			// micro-work
		})
	}
}

func BenchmarkNativeSchedulerCoupledBatch(b *testing.B) {
	cfg := SyntheticConfig()
	m := model.NewSynthetic(cfg)
	sched := NewNativeScheduler(m)
	defer sched.Close()

	ctx := context.Background()
	call := inlineCall("bench_call", `{"x":1}`)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		req, err := sched.Admit(ctx, call)
		if err != nil {
			b.Fatalf("Admit: %v", err)
		}
		for range req.Tokens() {
		}
		if _, err := req.Result(); err != nil {
			b.Fatalf("Result: %v", err)
		}
	}
}
