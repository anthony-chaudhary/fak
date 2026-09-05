package nativeperf

import (
	"encoding/json"
	"math"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestSubagentTraceValidation(t *testing.T) {
	validPhases := map[string]float64{
		SubagentPhaseHostDispatch:     120.5,
		SubagentPhasePrefixTreeLookup: 45.5,
		SubagentPhaseKVAllocation:     34.0,
		SubagentPhaseGPUKernel:        750.0,
		SubagentPhaseTokenSampling:    50.0,
	}
	const totalWall = 1000.0 // 120.5 + 45.5 + 34.0 + 750.0 + 50.0 = 1000.0

	receipt, err := NewSubagentTraceReceipt(1, "explore-agent", validPhases, totalWall)
	if err != nil {
		t.Fatalf("expected valid receipt, got error: %v", err)
	}

	if err := receipt.Validate(); err != nil {
		t.Fatalf("receipt.Validate() failed: %v", err)
	}

	// Test invalid schema
	invalidSchema := receipt
	invalidSchema.Schema = "wrong.schema/v1"
	if err := invalidSchema.Validate(); err == nil || !strings.Contains(err.Error(), "invalid schema") {
		t.Fatalf("expected invalid schema error, got %v", err)
	}

	// Test empty subagent ID
	emptyID := receipt
	emptyID.SubagentID = "  "
	if err := emptyID.Validate(); err == nil || !strings.Contains(err.Error(), "subagent_id is empty") {
		t.Fatalf("expected empty subagent_id error, got %v", err)
	}

	// Test negative turn
	negTurn := receipt
	negTurn.Turn = -1
	if err := negTurn.Validate(); err == nil || !strings.Contains(err.Error(), "negative turn") {
		t.Fatalf("expected negative turn error, got %v", err)
	}

	// Test negative total wall
	negWall := receipt
	negWall.TotalWallUS = -10.0
	if err := negWall.Validate(); err == nil || !strings.Contains(err.Error(), "total_wall_us must be non-negative") {
		t.Fatalf("expected negative wall error, got %v", err)
	}

	// Test negative phase duration
	negPhase := receipt
	negPhase.PhasesUS = make(map[string]float64)
	for k, v := range receipt.PhasesUS {
		negPhase.PhasesUS[k] = v
	}
	negPhase.PhasesUS[SubagentPhaseHostDispatch] = -5.0
	if err := negPhase.Validate(); err == nil || !strings.Contains(err.Error(), "must be non-negative") {
		t.Fatalf("expected negative phase error, got %v", err)
	}

	// Test missing required phase
	missingPhase := receipt
	missingPhase.PhasesUS = make(map[string]float64)
	for k, v := range receipt.PhasesUS {
		if k != SubagentPhasePrefixTreeLookup {
			missingPhase.PhasesUS[k] = v
		}
	}
	if err := missingPhase.Validate(); err == nil || !strings.Contains(err.Error(), "missing required phase") {
		t.Fatalf("expected missing phase error, got %v", err)
	}

	// Test unreconciled phase sum
	unreconciledSum := receipt
	unreconciledSum.TotalWallUS = 1500.0
	if err := unreconciledSum.Validate(); err == nil || !strings.Contains(err.Error(), "does not reconcile to total wall") {
		t.Fatalf("expected unreconciled sum error, got %v", err)
	}

	// Test unreconciled GPU kernel wall
	unreconciledGPU := receipt
	unreconciledGPU.GPUKernelWallUS = 500.0
	if err := unreconciledGPU.Validate(); err == nil || !strings.Contains(err.Error(), "does not reconcile to gpu_kernel phase") {
		t.Fatalf("expected unreconciled GPU error, got %v", err)
	}

	// Test unreconciled host CPU overhead
	unreconciledHost := receipt
	unreconciledHost.HostCPUOverheadUS = 100.0
	if err := unreconciledHost.Validate(); err == nil || !strings.Contains(err.Error(), "does not reconcile to non-gpu phases sum") {
		t.Fatalf("expected unreconciled host error, got %v", err)
	}

	// Test unreconciled percentages
	unreconciledPct := receipt
	unreconciledPct.HostCPUOverheadPercent = 50.0
	if err := unreconciledPct.Validate(); err == nil || !strings.Contains(err.Error(), "does not reconcile to expected") {
		t.Fatalf("expected unreconciled percentage error, got %v", err)
	}

	// Test NaN handling
	nanWall := receipt
	nanWall.TotalWallUS = math.NaN()
	if err := nanWall.Validate(); err == nil {
		t.Fatalf("expected error on NaN wall time")
	}

	// Test nil phases map
	nilPhases := receipt
	nilPhases.PhasesUS = nil
	if err := nilPhases.Validate(); err == nil || !strings.Contains(err.Error(), "phases_us map is nil") {
		t.Fatalf("expected error on nil phases map, got %v", err)
	}
}

func TestSubagentTraceMicrosecondAccuracy(t *testing.T) {
	timer := NewSubagentTraceTimer(42, "tester-subagent")

	// 1500 nanoseconds = 1.5 microseconds
	timer.RecordDuration(SubagentPhaseHostDispatch, 1500*time.Nanosecond)
	// 2250 nanoseconds = 2.25 microseconds
	timer.RecordDuration(SubagentPhasePrefixTreeLookup, 2250*time.Nanosecond)
	// 1000 nanoseconds = 1.0 microsecond
	timer.RecordDuration(SubagentPhaseKVAllocation, 1000*time.Nanosecond)
	// 10000 nanoseconds = 10.0 microseconds
	timer.RecordDuration(SubagentPhaseGPUKernel, 10000*time.Nanosecond)
	// 250 nanoseconds = 0.25 microseconds
	timer.RecordDuration(SubagentPhaseTokenSampling, 250*time.Nanosecond)

	receipt, err := timer.Finalize()
	if err != nil {
		t.Fatalf("timer.Finalize failed: %v", err)
	}

	if err := receipt.Validate(); err != nil {
		t.Fatalf("receipt.Validate failed: %v", err)
	}

	const expectedTotal = 1.5 + 2.25 + 1.0 + 10.0 + 0.25 // 15.0 µs
	if math.Abs(receipt.TotalWallUS-expectedTotal) > 1e-6 {
		t.Errorf("TotalWallUS = %f, want %f", receipt.TotalWallUS, expectedTotal)
	}

	if receipt.PhasesUS[SubagentPhaseHostDispatch] != 1.5 {
		t.Errorf("host_dispatch = %f, want 1.5", receipt.PhasesUS[SubagentPhaseHostDispatch])
	}
	if receipt.PhasesUS[SubagentPhasePrefixTreeLookup] != 2.25 {
		t.Errorf("prefix_tree_lookup = %f, want 2.25", receipt.PhasesUS[SubagentPhasePrefixTreeLookup])
	}
	if receipt.PhasesUS[SubagentPhaseKVAllocation] != 1.0 {
		t.Errorf("kv_allocation = %f, want 1.0", receipt.PhasesUS[SubagentPhaseKVAllocation])
	}
	if receipt.PhasesUS[SubagentPhaseGPUKernel] != 10.0 {
		t.Errorf("gpu_kernel = %f, want 10.0", receipt.PhasesUS[SubagentPhaseGPUKernel])
	}
	if receipt.PhasesUS[SubagentPhaseTokenSampling] != 0.25 {
		t.Errorf("token_sampling = %f, want 0.25", receipt.PhasesUS[SubagentPhaseTokenSampling])
	}

	// Host overhead: 1.5 + 2.25 + 1.0 + 0.25 = 5.0 µs
	const expectedHost = 5.0
	if math.Abs(receipt.HostCPUOverheadUS-expectedHost) > 1e-6 {
		t.Errorf("HostCPUOverheadUS = %f, want %f", receipt.HostCPUOverheadUS, expectedHost)
	}

	// GPU kernel: 10.0 µs
	const expectedGPU = 10.0
	if math.Abs(receipt.GPUKernelWallUS-expectedGPU) > 1e-6 {
		t.Errorf("GPUKernelWallUS = %f, want %f", receipt.GPUKernelWallUS, expectedGPU)
	}

	// Percentages: Host = 5.0 / 15.0 * 100 = 33.3333%, GPU = 10.0 / 15.0 * 100 = 66.6667%
	expectedHostPct := (expectedHost / expectedTotal) * 100.0
	expectedGPUPct := (expectedGPU / expectedTotal) * 100.0
	if math.Abs(receipt.HostCPUOverheadPercent-expectedHostPct) > 1e-4 {
		t.Errorf("HostCPUOverheadPercent = %f, want %f", receipt.HostCPUOverheadPercent, expectedHostPct)
	}
	if math.Abs(receipt.GPUKernelWallPercent-expectedGPUPct) > 1e-4 {
		t.Errorf("GPUKernelWallPercent = %f, want %f", receipt.GPUKernelWallPercent, expectedGPUPct)
	}
}

func TestSubagentTraceOverheadPercentageCalculation(t *testing.T) {
	tests := []struct {
		name            string
		phases          map[string]float64
		wantTotal       float64
		wantHost        float64
		wantGPU         float64
		wantHostPercent float64
		wantGPUPercent  float64
	}{
		{
			name: "balanced",
			phases: map[string]float64{
				SubagentPhaseHostDispatch:     200.0,
				SubagentPhasePrefixTreeLookup: 50.0,
				SubagentPhaseKVAllocation:     50.0,
				SubagentPhaseGPUKernel:        600.0,
				SubagentPhaseTokenSampling:    100.0,
			},
			wantTotal:       1000.0,
			wantHost:        400.0,
			wantGPU:         600.0,
			wantHostPercent: 40.0,
			wantGPUPercent:  60.0,
		},
		{
			name: "pure-gpu",
			phases: map[string]float64{
				SubagentPhaseHostDispatch:     0.0,
				SubagentPhasePrefixTreeLookup: 0.0,
				SubagentPhaseKVAllocation:     0.0,
				SubagentPhaseGPUKernel:        500.0,
				SubagentPhaseTokenSampling:    0.0,
			},
			wantTotal:       500.0,
			wantHost:        0.0,
			wantGPU:         500.0,
			wantHostPercent: 0.0,
			wantGPUPercent:  100.0,
		},
		{
			name: "pure-host",
			phases: map[string]float64{
				SubagentPhaseHostDispatch:     100.0,
				SubagentPhasePrefixTreeLookup: 50.0,
				SubagentPhaseKVAllocation:     25.0,
				SubagentPhaseGPUKernel:        0.0,
				SubagentPhaseTokenSampling:    25.0,
			},
			wantTotal:       200.0,
			wantHost:        200.0,
			wantGPU:         0.0,
			wantHostPercent: 100.0,
			wantGPUPercent:  0.0,
		},
		{
			name: "zero-duration",
			phases: map[string]float64{
				SubagentPhaseHostDispatch:     0.0,
				SubagentPhasePrefixTreeLookup: 0.0,
				SubagentPhaseKVAllocation:     0.0,
				SubagentPhaseGPUKernel:        0.0,
				SubagentPhaseTokenSampling:    0.0,
			},
			wantTotal:       0.0,
			wantHost:        0.0,
			wantGPU:         0.0,
			wantHostPercent: 0.0,
			wantGPUPercent:  0.0,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			receipt, err := NewSubagentTraceReceipt(1, "worker", tc.phases, tc.wantTotal)
			if err != nil {
				t.Fatalf("NewSubagentTraceReceipt failed: %v", err)
			}
			if err := receipt.Validate(); err != nil {
				t.Fatalf("Validate failed: %v", err)
			}

			if math.Abs(receipt.TotalWallUS-tc.wantTotal) > 1e-6 {
				t.Errorf("TotalWallUS = %f, want %f", receipt.TotalWallUS, tc.wantTotal)
			}
			if math.Abs(receipt.HostCPUOverheadUS-tc.wantHost) > 1e-6 {
				t.Errorf("HostCPUOverheadUS = %f, want %f", receipt.HostCPUOverheadUS, tc.wantHost)
			}
			if math.Abs(receipt.GPUKernelWallUS-tc.wantGPU) > 1e-6 {
				t.Errorf("GPUKernelWallUS = %f, want %f", receipt.GPUKernelWallUS, tc.wantGPU)
			}
			if math.Abs(receipt.HostCPUOverheadPercent-tc.wantHostPercent) > 1e-4 {
				t.Errorf("HostCPUOverheadPercent = %f, want %f", receipt.HostCPUOverheadPercent, tc.wantHostPercent)
			}
			if math.Abs(receipt.GPUKernelWallPercent-tc.wantGPUPercent) > 1e-4 {
				t.Errorf("GPUKernelWallPercent = %f, want %f", receipt.GPUKernelWallPercent, tc.wantGPUPercent)
			}
		})
	}
}

func TestSubagentTraceJSONSerialization(t *testing.T) {
	phases := map[string]float64{
		SubagentPhaseHostDispatch:     110.25,
		SubagentPhasePrefixTreeLookup: 40.75,
		SubagentPhaseKVAllocation:     29.0,
		SubagentPhaseGPUKernel:        800.0,
		SubagentPhaseTokenSampling:    20.0,
	}
	original, err := NewSubagentTraceReceipt(3, "json-test-agent", phases, 1000.0)
	if err != nil {
		t.Fatalf("failed to create receipt: %v", err)
	}

	data, err := original.JSON()
	if err != nil {
		t.Fatalf("failed to marshal receipt: %v", err)
	}

	jsonStr := string(data)
	requiredKeys := []string{
		`"schema": "fak.subagent.trace/v1"`,
		`"turn": 3`,
		`"subagent_id": "json-test-agent"`,
		`"total_wall_us": 1000`,
		`"phases_us"`,
		`"host_dispatch": 110.25`,
		`"prefix_tree_lookup": 40.75`,
		`"kv_allocation": 29`,
		`"gpu_kernel": 800`,
		`"token_sampling": 20`,
		`"host_cpu_overhead_us": 200`,
		`"gpu_kernel_wall_us": 800`,
		`"host_cpu_overhead_percent": 20`,
		`"gpu_kernel_wall_percent": 80`,
	}
	for _, key := range requiredKeys {
		if !strings.Contains(jsonStr, key) {
			t.Errorf("JSON output missing expected string %q\nJSON:\n%s", key, jsonStr)
		}
	}

	var parsed SubagentTraceReceipt
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("failed to unmarshal JSON: %v", err)
	}

	if err := parsed.Validate(); err != nil {
		t.Fatalf("unmarshaled receipt failed Validate: %v", err)
	}

	if parsed.Turn != original.Turn || parsed.SubagentID != original.SubagentID {
		t.Errorf("mismatch in identity: got Turn=%d ID=%s, want Turn=%d ID=%s",
			parsed.Turn, parsed.SubagentID, original.Turn, original.SubagentID)
	}
	if parsed.TotalWallUS != original.TotalWallUS {
		t.Errorf("TotalWallUS mismatch: got %f, want %f", parsed.TotalWallUS, original.TotalWallUS)
	}
	if parsed.HostCPUOverheadUS != original.HostCPUOverheadUS {
		t.Errorf("HostCPUOverheadUS mismatch: got %f, want %f", parsed.HostCPUOverheadUS, original.HostCPUOverheadUS)
	}
	if parsed.GPUKernelWallUS != original.GPUKernelWallUS {
		t.Errorf("GPUKernelWallUS mismatch: got %f, want %f", parsed.GPUKernelWallUS, original.GPUKernelWallUS)
	}
}

func TestSubagentTracePhaseRecorderIntegration(t *testing.T) {
	// 1. Build a PhaseRecorder with realistic execution phases
	recorder := NewPhaseRecorder("inkernel", "cuda", "qwen3.8_cuda", 20*time.Millisecond)
	adds := []struct {
		phase, parent Phase
		start, end    time.Duration
		kind          WorkKind
	}{
		{PhaseQueueAdmission, "", 0, 1 * time.Millisecond, WorkWait},
		{PhaseModelLoad, "", 1 * time.Millisecond, 2 * time.Millisecond, WorkActive},
		{PhaseTokenization, "", 2 * time.Millisecond, 3 * time.Millisecond, WorkActive},
		{PhaseKVLookup, "", 3 * time.Millisecond, 5 * time.Millisecond, WorkActive},   // 2ms prefix_tree_lookup
		{PhaseKVRestore, "", 5 * time.Millisecond, 7 * time.Millisecond, WorkWait},    // 2ms kv_allocation
		{PhaseKernel, "", 7 * time.Millisecond, 17 * time.Millisecond, WorkActive},    // 10ms gpu_kernel
		{PhaseSampling, "", 17 * time.Millisecond, 19 * time.Millisecond, WorkActive}, // 2ms token_sampling
		{PhaseOutput, "", 19 * time.Millisecond, 20 * time.Millisecond, WorkActive},   // 1ms host_dispatch
		// Remaining bounded phases with 0 duration:
		{PhasePrefill, "", 20 * time.Millisecond, 20 * time.Millisecond, WorkActive},
		{PhaseDecode, "", 20 * time.Millisecond, 20 * time.Millisecond, WorkActive},
		{PhaseKVEvict, "", 20 * time.Millisecond, 20 * time.Millisecond, WorkActive},
		{PhaseHostUpload, "", 20 * time.Millisecond, 20 * time.Millisecond, WorkWait},
		{PhaseHostDownload, "", 20 * time.Millisecond, 20 * time.Millisecond, WorkWait},
		{PhaseSynchronization, "", 20 * time.Millisecond, 20 * time.Millisecond, WorkWait},
	}
	for _, a := range adds {
		if err := recorder.Add(a.phase, a.parent, a.start, a.end, a.kind); err != nil {
			t.Fatalf("recorder.Add(%s): %v", a.phase, err)
		}
	}

	subReceipt, err := recorder.SubagentTrace(10, "subagent-integration", 0)
	if err != nil {
		t.Fatalf("recorder.SubagentTrace failed: %v", err)
	}

	if err := subReceipt.Validate(); err != nil {
		t.Fatalf("subReceipt.Validate failed: %v", err)
	}

	// 20ms total wall = 20000 µs
	if math.Abs(subReceipt.TotalWallUS-20000.0) > 1e-3 {
		t.Errorf("TotalWallUS = %f, want 20000.0", subReceipt.TotalWallUS)
	}

	// prefix_tree_lookup = 2ms = 2000 µs
	if math.Abs(subReceipt.PhasesUS[SubagentPhasePrefixTreeLookup]-2000.0) > 1e-3 {
		t.Errorf("prefix_tree_lookup = %f, want 2000.0", subReceipt.PhasesUS[SubagentPhasePrefixTreeLookup])
	}

	// kv_allocation = 2ms = 2000 µs
	if math.Abs(subReceipt.PhasesUS[SubagentPhaseKVAllocation]-2000.0) > 1e-3 {
		t.Errorf("kv_allocation = %f, want 2000.0", subReceipt.PhasesUS[SubagentPhaseKVAllocation])
	}

	// gpu_kernel = 10ms = 10000 µs
	if math.Abs(subReceipt.GPUKernelWallUS-10000.0) > 1e-3 {
		t.Errorf("GPUKernelWallUS = %f, want 10000.0", subReceipt.GPUKernelWallUS)
	}

	// token_sampling = 2ms = 2000 µs
	if math.Abs(subReceipt.PhasesUS[SubagentPhaseTokenSampling]-2000.0) > 1e-3 {
		t.Errorf("token_sampling = %f, want 2000.0", subReceipt.PhasesUS[SubagentPhaseTokenSampling])
	}

	// host_dispatch = 1ms + 1ms + 1ms + 1ms = 4ms = 4000 µs
	if math.Abs(subReceipt.PhasesUS[SubagentPhaseHostDispatch]-4000.0) > 1e-3 {
		t.Errorf("host_dispatch = %f, want 4000.0", subReceipt.PhasesUS[SubagentPhaseHostDispatch])
	}

	// GPU Kernel % = 50%, Host CPU % = 50%
	if math.Abs(subReceipt.GPUKernelWallPercent-50.0) > 1e-2 {
		t.Errorf("GPUKernelWallPercent = %f, want 50.0", subReceipt.GPUKernelWallPercent)
	}
	if math.Abs(subReceipt.HostCPUOverheadPercent-50.0) > 1e-2 {
		t.Errorf("HostCPUOverheadPercent = %f, want 50.0", subReceipt.HostCPUOverheadPercent)
	}

	// 2. Test round-trip conversion: SubagentTraceReceipt -> PhaseRecorder -> SubagentTraceReceipt
	recRoundTrip, err := subReceipt.ToPhaseRecorder("inkernel", "cuda", "qwen3.8_cuda")
	if err != nil {
		t.Fatalf("ToPhaseRecorder failed: %v", err)
	}

	reconstructed, err := recRoundTrip.SubagentTrace(subReceipt.Turn, subReceipt.SubagentID, 0)
	if err != nil {
		t.Fatalf("reconstructed SubagentTrace failed: %v", err)
	}

	if err := reconstructed.Validate(); err != nil {
		t.Fatalf("reconstructed receipt failed Validate: %v", err)
	}

	if math.Abs(reconstructed.TotalWallUS-subReceipt.TotalWallUS) > 1.0 {
		t.Errorf("reconstructed TotalWallUS = %f, want %f", reconstructed.TotalWallUS, subReceipt.TotalWallUS)
	}
	if math.Abs(reconstructed.GPUKernelWallUS-subReceipt.GPUKernelWallUS) > 1.0 {
		t.Errorf("reconstructed GPUKernelWallUS = %f, want %f", reconstructed.GPUKernelWallUS, subReceipt.GPUKernelWallUS)
	}
	if math.Abs(reconstructed.HostCPUOverheadUS-subReceipt.HostCPUOverheadUS) > 1.0 {
		t.Errorf("reconstructed HostCPUOverheadUS = %f, want %f", reconstructed.HostCPUOverheadUS, subReceipt.HostCPUOverheadUS)
	}
}

func TestSubagentTraceTimerStartStop(t *testing.T) {
	timer := NewSubagentTraceTimer(1, "start-stop-agent")

	timer.TimePhase(SubagentPhaseHostDispatch, func() {
		time.Sleep(2 * time.Millisecond)
	})

	stopLookup := timer.Start(SubagentPhasePrefixTreeLookup)
	time.Sleep(1 * time.Millisecond)
	stopLookup()

	stopKV := timer.StartPhase(SubagentPhaseKVAllocation)
	time.Sleep(1 * time.Millisecond)
	stopKV()

	stopGPU := timer.Start(SubagentPhaseGPUKernel)
	time.Sleep(3 * time.Millisecond)
	stopGPU()

	stopSample := timer.Start(SubagentPhaseTokenSampling)
	time.Sleep(1 * time.Millisecond)
	stopSample()

	receipt, err := timer.Finalize()
	if err != nil {
		t.Fatalf("Finalize failed: %v", err)
	}

	if err := receipt.Validate(); err != nil {
		t.Fatalf("receipt.Validate failed: %v", err)
	}

	// Ensure all phases accumulated positive durations
	for _, p := range SubagentPhases() {
		if receipt.PhasesUS[p] <= 0 {
			t.Errorf("phase %q has non-positive duration %f", p, receipt.PhasesUS[p])
		}
	}

	if receipt.TotalWallUS <= 0 {
		t.Errorf("TotalWallUS is non-positive: %f", receipt.TotalWallUS)
	}
}

func TestSubagentTraceTimerConcurrentRecording(t *testing.T) {
	timer := NewSubagentTraceTimer(5, "concurrent-agent")
	var wg sync.WaitGroup

	for i := 0; i < 50; i++ {
		wg.Add(5)
		go func() {
			defer wg.Done()
			timer.RecordUS(SubagentPhaseHostDispatch, 10.0)
		}()
		go func() {
			defer wg.Done()
			timer.RecordUS(SubagentPhasePrefixTreeLookup, 5.0)
		}()
		go func() {
			defer wg.Done()
			timer.RecordUS(SubagentPhaseKVAllocation, 5.0)
		}()
		go func() {
			defer wg.Done()
			timer.RecordUS(SubagentPhaseGPUKernel, 70.0)
		}()
		go func() {
			defer wg.Done()
			timer.RecordUS(SubagentPhaseTokenSampling, 10.0)
		}()
	}

	wg.Wait()

	receipt, err := timer.Finalize()
	if err != nil {
		t.Fatalf("Finalize failed: %v", err)
	}

	if err := receipt.Validate(); err != nil {
		t.Fatalf("receipt.Validate failed: %v", err)
	}

	const expectedTotal = 50 * (10.0 + 5.0 + 5.0 + 70.0 + 10.0) // 50 * 100 = 5000 µs
	if math.Abs(receipt.TotalWallUS-expectedTotal) > 1e-4 {
		t.Errorf("TotalWallUS = %f, want %f", receipt.TotalWallUS, expectedTotal)
	}
	if math.Abs(receipt.GPUKernelWallUS-(50*70.0)) > 1e-4 {
		t.Errorf("GPUKernelWallUS = %f, want %f", receipt.GPUKernelWallUS, 50*70.0)
	}
	if math.Abs(receipt.HostCPUOverheadUS-(50*30.0)) > 1e-4 {
		t.Errorf("HostCPUOverheadUS = %f, want %f", receipt.HostCPUOverheadUS, 50*30.0)
	}
}
