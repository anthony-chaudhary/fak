package macobs

import (
	"testing"
)

func TestDiagnoseSwapCritical(t *testing.T) {
	hw := HardwareTelemetry{
		Available:     true,
		SwapUsedBytes: 2 * 1024 * 1024 * 1024, // 2GB
		PageOuts:      60000,
	}
	srv := MLXServingTelemetry{Available: true}
	head := HeadroomTelemetry{Available: true, MaxSharedAgents: 10}

	report := Diagnose(hw, srv, head, 5)
	if report.Verdict != VerdictSwapCritical {
		t.Errorf("got verdict %s, want SWAP_CRITICAL", report.Verdict)
	}
	if report.PrimaryBottleneck != BottleneckSwap {
		t.Errorf("got bottleneck %s, want SWAP", report.PrimaryBottleneck)
	}
	if report.RecommendedAgents != 1 {
		t.Errorf("got recommended agents %d, want 1", report.RecommendedAgents)
	}
}

func TestDiagnoseThermalThrottled(t *testing.T) {
	hw := HardwareTelemetry{
		Available:       true,
		ThermalState:    ThermalSerious,
		CPUThermalLevel: 2,
	}
	srv := MLXServingTelemetry{Available: true}
	head := HeadroomTelemetry{Available: true, MaxSharedAgents: 20}

	report := Diagnose(hw, srv, head, 10)
	if report.Verdict != VerdictThermalThrottled {
		t.Errorf("got verdict %s, want THERMAL_THROTTLED", report.Verdict)
	}
	if report.PrimaryBottleneck != BottleneckThermal {
		t.Errorf("got bottleneck %s, want THERMAL", report.PrimaryBottleneck)
	}
	if report.RecommendedAgents != 5 {
		t.Errorf("got recommended agents %d, want 5", report.RecommendedAgents)
	}
}

func TestDiagnosePressureDegrade(t *testing.T) {
	hw := HardwareTelemetry{
		Available:              true,
		MemoryStatusLevel:      10, // critical level <= 15
		WiredMemoryLimitBytes:  24 * 1024 * 1024 * 1024,
		InUseSystemMemoryBytes: 23 * 1024 * 1024 * 1024, // ~96%
	}
	srv := MLXServingTelemetry{Available: true}
	head := HeadroomTelemetry{Available: true, MaxSharedAgents: 20, MaxIsolatedAgents: 4}

	report := Diagnose(hw, srv, head, 10)
	if report.Verdict != VerdictPressureDegrade {
		t.Errorf("got verdict %s, want PRESSURE_DEGRADE", report.Verdict)
	}
	if report.PrimaryBottleneck != BottleneckMemoryCapacity {
		t.Errorf("got bottleneck %s, want MEMORY_CAPACITY", report.PrimaryBottleneck)
	}
	if report.RecommendedAgents != 4 {
		t.Errorf("got recommended agents %d, want 4", report.RecommendedAgents)
	}
}

func TestDiagnoseEvictPrefixCache(t *testing.T) {
	hw := HardwareTelemetry{
		Available:             true,
		WiredMemoryLimitBytes: 24 * 1024 * 1024 * 1024,
	}
	srv := MLXServingTelemetry{
		Available:       true,
		KVCacheUsagePct: 94.5,
		ActiveRequests:  6,
	}
	head := HeadroomTelemetry{Available: true, MaxSharedAgents: 20}

	report := Diagnose(hw, srv, head, 6)
	if report.Verdict != VerdictEvictPrefixCache {
		t.Errorf("got verdict %s, want EVICT_PREFIX_CACHE", report.Verdict)
	}
	if report.PrimaryBottleneck != BottleneckCacheSaturation {
		t.Errorf("got bottleneck %s, want CACHE_SATURATION", report.PrimaryBottleneck)
	}
	if report.RecommendedAgents != 6 {
		t.Errorf("got recommended agents %d, want 6", report.RecommendedAgents)
	}
}

func TestDiagnoseReduceConcurrency(t *testing.T) {
	hw := HardwareTelemetry{
		Available:             true,
		WiredMemoryLimitBytes: 24 * 1024 * 1024 * 1024,
	}
	srv := MLXServingTelemetry{Available: true}
	head := HeadroomTelemetry{Available: true, MaxSharedAgents: 8}

	// Requested 16 agents > MaxSharedAgents 8
	report := Diagnose(hw, srv, head, 16)
	if report.Verdict != VerdictReduceConcurrency {
		t.Errorf("got verdict %s, want REDUCE_CONCURRENCY", report.Verdict)
	}
	if report.PrimaryBottleneck != BottleneckMemoryCapacity {
		t.Errorf("got bottleneck %s, want MEMORY_CAPACITY", report.PrimaryBottleneck)
	}
	if report.RecommendedAgents != 8 {
		t.Errorf("got recommended agents %d, want 8", report.RecommendedAgents)
	}
}

func TestDiagnoseHeadroomOK(t *testing.T) {
	hw := HardwareTelemetry{
		Available:             true,
		WiredMemoryLimitBytes: 24 * 1024 * 1024 * 1024,
		DeviceUtilizationPct:  45.0,
	}
	srv := MLXServingTelemetry{
		Available:          true,
		KVCacheUsagePct:    35.0,
		DecodeTokensPerSec: 32.0,
	}
	head := HeadroomTelemetry{Available: true, MaxSharedAgents: 32}

	report := Diagnose(hw, srv, head, 4)
	if report.Verdict != VerdictHeadroomOK {
		t.Errorf("got verdict %s, want HEADROOM_OK", report.Verdict)
	}
	if report.PrimaryBottleneck != BottleneckNone {
		t.Errorf("got bottleneck %s, want NONE", report.PrimaryBottleneck)
	}
	if report.RecommendedAgents != 4 {
		t.Errorf("got recommended agents %d, want 4", report.RecommendedAgents)
	}
}

func TestDiagnoseBottleneckFineGrained(t *testing.T) {
	// Compute prefill bottleneck: 98% GPU utilization
	hw := HardwareTelemetry{
		Available:            true,
		DeviceUtilizationPct: 98.0,
	}
	srv := MLXServingTelemetry{Available: true, DecodeTokensPerSec: 30.0}
	head := HeadroomTelemetry{Available: true, MaxSharedAgents: 20}

	rep := Diagnose(hw, srv, head, 5)
	if rep.PrimaryBottleneck != BottleneckComputePrefill {
		t.Errorf("got bottleneck %s, want COMPUTE_PREFILL", rep.PrimaryBottleneck)
	}

	// Memory bandwidth bottleneck: high GPU util (85%) with low decode throughput (15 tok/s)
	hw.DeviceUtilizationPct = 85.0
	srv.DecodeTokensPerSec = 15.0
	rep = Diagnose(hw, srv, head, 5)
	if rep.PrimaryBottleneck != BottleneckMemoryBandwidth {
		t.Errorf("got bottleneck %s, want MEMORY_BANDWIDTH", rep.PrimaryBottleneck)
	}
}

func TestClosedEnumsValid(t *testing.T) {
	// ActionVerdict
	verdicts := []ActionVerdict{
		VerdictHeadroomOK,
		VerdictReduceConcurrency,
		VerdictEvictPrefixCache,
		VerdictPressureDegrade,
		VerdictThermalThrottled,
		VerdictSwapCritical,
	}
	for _, v := range verdicts {
		if !v.Valid() {
			t.Errorf("verdict %s should be valid", v)
		}
	}
	if ActionVerdict("UNKNOWN_VERDICT").Valid() {
		t.Errorf("unexpected valid verdict for UNKNOWN_VERDICT")
	}

	// BottleneckType
	bottlenecks := []BottleneckType{
		BottleneckNone,
		BottleneckMemoryCapacity,
		BottleneckMemoryBandwidth,
		BottleneckComputePrefill,
		BottleneckComputeDecode,
		BottleneckThermal,
		BottleneckSwap,
		BottleneckCacheSaturation,
	}
	for _, b := range bottlenecks {
		if !b.Valid() {
			t.Errorf("bottleneck %s should be valid", b)
		}
	}
	if BottleneckType("UNKNOWN").Valid() {
		t.Errorf("unexpected valid bottleneck for UNKNOWN")
	}

	// ThermalState
	thermals := []ThermalState{
		ThermalNominal,
		ThermalFair,
		ThermalSerious,
		ThermalCritical,
		ThermalUnknown,
	}
	for _, th := range thermals {
		if !th.Valid() {
			t.Errorf("thermal %s should be valid", th)
		}
	}

	// PowerSource
	powers := []PowerSource{
		PowerAC,
		PowerBattery,
		PowerUnknown,
	}
	for _, p := range powers {
		if !p.Valid() {
			t.Errorf("power %s should be valid", p)
		}
	}
}

func TestDiagnose_SwapThrashingHighPageOuts(t *testing.T) {
	// Small swap bytes (e.g. 100MB) but > 50,000 pageouts triggers VerdictSwapCritical
	hw := HardwareTelemetry{
		Available:     true,
		SwapUsedBytes: 100 * 1024 * 1024,
		PageOuts:      75000,
	}
	srv := MLXServingTelemetry{Available: true}
	head := HeadroomTelemetry{Available: true, MaxSharedAgents: 10}

	report := Diagnose(hw, srv, head, 5)
	if report.Verdict != VerdictSwapCritical {
		t.Errorf("got verdict %s, want SWAP_CRITICAL", report.Verdict)
	}
	if report.PrimaryBottleneck != BottleneckSwap {
		t.Errorf("got bottleneck %s, want SWAP", report.PrimaryBottleneck)
	}
	if report.RecommendedAgents != 1 {
		t.Errorf("got recommended agents %d, want 1", report.RecommendedAgents)
	}
}

func TestDiagnose_ThermalCriticalAndGpuThrottle(t *testing.T) {
	// ThermalCritical and GPU level >= 2
	hw := HardwareTelemetry{
		Available:       true,
		ThermalState:    ThermalCritical,
		GPUThermalLevel: 3,
	}
	srv := MLXServingTelemetry{Available: true}
	head := HeadroomTelemetry{Available: true, MaxSharedAgents: 16}

	report := Diagnose(hw, srv, head, 8)
	if report.Verdict != VerdictThermalThrottled {
		t.Errorf("got verdict %s, want THERMAL_THROTTLED", report.Verdict)
	}
	if report.RecommendedAgents != 4 { // 8 / 2 = 4
		t.Errorf("got recommended agents %d, want 4", report.RecommendedAgents)
	}

	// Odd requested agents should halve safely with minimum 1
	reportOdd := Diagnose(hw, srv, head, 1)
	if reportOdd.RecommendedAgents != 1 {
		t.Errorf("got recommended agents %d, want 1 (minimum floor)", reportOdd.RecommendedAgents)
	}
}

func TestDiagnose_QueueContention(t *testing.T) {
	// Active requests reached MaxSharedAgents AND QueuedRequests > 0
	hw := HardwareTelemetry{
		Available:             true,
		WiredMemoryLimitBytes: 32 * 1024 * 1024 * 1024,
	}
	srv := MLXServingTelemetry{
		Available:      true,
		ActiveRequests: 8,
		QueuedRequests: 4,
	}
	head := HeadroomTelemetry{
		Available:       true,
		MaxSharedAgents: 8,
	}

	report := Diagnose(hw, srv, head, 8)
	if report.Verdict != VerdictReduceConcurrency {
		t.Errorf("got verdict %s, want REDUCE_CONCURRENCY", report.Verdict)
	}
	if report.PrimaryBottleneck != BottleneckComputePrefill {
		t.Errorf("got bottleneck %s, want COMPUTE_PREFILL", report.PrimaryBottleneck)
	}
	if report.RecommendedAgents != 8 {
		t.Errorf("got recommended agents %d, want 8", report.RecommendedAgents)
	}
}

func TestDiagnose_RadicalEdgeCases(t *testing.T) {
	// 1. Zero or negative requested agents should be sanitized to 1
	hw := HardwareTelemetry{Available: true}
	srv := MLXServingTelemetry{Available: true}
	head := HeadroomTelemetry{Available: true, MaxSharedAgents: 10}

	repZero := Diagnose(hw, srv, head, 0)
	if repZero.RecommendedAgents != 1 {
		t.Errorf("got recommended agents %d, want 1 for zero requested", repZero.RecommendedAgents)
	}

	repNeg := Diagnose(hw, srv, head, -5)
	if repNeg.RecommendedAgents != 1 {
		t.Errorf("got recommended agents %d, want 1 for negative requested", repNeg.RecommendedAgents)
	}

	// 2. Massive concurrency request (e.g. 500 agents) exceeds MaxSharedAgents (10)
	repMassive := Diagnose(hw, srv, head, 500)
	if repMassive.Verdict != VerdictReduceConcurrency {
		t.Errorf("got verdict %s, want REDUCE_CONCURRENCY for massive agents", repMassive.Verdict)
	}
	if repMassive.RecommendedAgents != 10 {
		t.Errorf("got recommended agents %d, want 10", repMassive.RecommendedAgents)
	}

	// 3. Telemetry unavailable -> degraded confidence 0.70
	hwUnavail := HardwareTelemetry{Available: false}
	srvUnavail := MLXServingTelemetry{Available: false}
	headUnavail := HeadroomTelemetry{Available: false}

	repUnavail := Diagnose(hwUnavail, srvUnavail, headUnavail, 2)
	if repUnavail.Confidence != 0.70 {
		t.Errorf("got confidence %f, want 0.70 for unavailable telemetry", repUnavail.Confidence)
	}
}
