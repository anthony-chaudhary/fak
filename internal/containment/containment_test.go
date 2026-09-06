package containment

import (
	"testing"
)

const (
	GiB = 1024 * 1024 * 1024
)

func TestEstimateAPUResidentMemory_GTTBypass(t *testing.T) {
	// Qwen 30B class model with 22 GB weights, 32k context, 2 GB scratch
	meta := ModelMetadata{
		ModelName:       "qwen3-30b-q4k",
		ParameterCount:  30_000_000_000,
		WeightBytes:     20 * GiB,
		ContextTokens:   32768,
		KVBytesPerToken: 65536, // ~2 GB KV
		ScratchpadBytes: 2 * GiB,
	}

	expectedMin := meta.EstimatedMemoryBytes() // 20 + 2.147 + 2 = 24.147 GiB

	// Degraded cgroup telemetry: container cgroup v2 reports only 2 GiB because
	// AMDGPU GTT/TTM allocations bypassed cgroup memory.current counters.
	cgroupBytes := int64(2 * GiB)

	// On APU, hybrid max-pooling must lift the footprint to the unforgeable metadata floor
	attributed := EstimateAPUResidentMemory(cgroupBytes, meta, true)
	if attributed != expectedMin {
		t.Fatalf("attributed = %d (%.2f GiB), want %d (%.2f GiB)",
			attributed, float64(attributed)/float64(GiB), expectedMin, float64(expectedMin)/float64(GiB))
	}
}

func TestEstimateAPUResidentMemory_AccurateCgroup(t *testing.T) {
	meta := ModelMetadata{
		ModelName:       "qwen3-7b-q4k",
		ParameterCount:  7_000_000_000,
		WeightBytes:     5 * GiB,
		ContextTokens:   8192,
		KVBytesPerToken: 32768,
		ScratchpadBytes: 512 * 1024 * 1024,
	}

	// Cgroup reflects actual working set + cached data (8 GiB > 5.76 GiB estimated)
	cgroupBytes := int64(8 * GiB)

	attributed := EstimateAPUResidentMemory(cgroupBytes, meta, true)
	if attributed != cgroupBytes {
		t.Fatalf("attributed = %d, want cgroupBytes %d", attributed, cgroupBytes)
	}
}

func TestEstimateAPUResidentMemory_NonAPU(t *testing.T) {
	meta := ModelMetadata{
		ModelName:       "qwen3-30b-q4k",
		WeightBytes:     20 * GiB,
		ContextTokens:   32768,
		KVBytesPerToken: 65536,
		ScratchpadBytes: 2 * GiB,
	}
	cgroupBytes := int64(2 * GiB)

	// On discrete platforms (not APU), cgroup telemetry is trusted directly
	attributed := EstimateAPUResidentMemory(cgroupBytes, meta, false)
	if attributed != cgroupBytes {
		t.Fatalf("attributed = %d, want non-APU cgroupBytes %d", attributed, cgroupBytes)
	}
}

func TestEvaluateAPUAllocation_Record(t *testing.T) {
	meta := ModelMetadata{
		ModelName:       "deepseek-lite",
		WeightBytes:     10 * GiB,
		ContextTokens:   16384,
		KVBytesPerToken: 32768,
		ScratchpadBytes: 1 * GiB,
	}
	cgroupBytes := int64(3 * GiB)

	record := EvaluateAPUAllocation("instance-worker-1", cgroupBytes, meta, true)
	if !record.GTTBypassDetected {
		t.Errorf("GTTBypassDetected = false, want true")
	}
	expectedEstimate := meta.EstimatedMemoryBytes()
	if record.AttributedResidentBytes != expectedEstimate {
		t.Errorf("AttributedResidentBytes = %d, want %d", record.AttributedResidentBytes, expectedEstimate)
	}
	if record.APUUnderreportedBytes != (expectedEstimate - cgroupBytes) {
		t.Errorf("APUUnderreportedBytes = %d, want %d", record.APUUnderreportedBytes, expectedEstimate-cgroupBytes)
	}
}

func TestArbitrateAPUHeadroom(t *testing.T) {
	// Strix Halo 128GB: ~120 GiB available UMA aperture
	totalAperture := int64(120 * GiB)

	meta1 := ModelMetadata{WeightBytes: 40 * GiB}
	meta2 := ModelMetadata{WeightBytes: 50 * GiB}
	meta3 := ModelMetadata{WeightBytes: 40 * GiB}

	// Instance 1 (40 GiB) and Instance 2 (50 GiB)
	inst1 := EvaluateAPUAllocation("inst-1", 4*GiB, meta1, true)  // GTT bypass: reports 4, actually 40
	inst2 := EvaluateAPUAllocation("inst-2", 50*GiB, meta2, true) // accurate: 50

	consumed, free, hasCap := ArbitrateAPUHeadroom(totalAperture, []APUAllocationRecord{inst1, inst2})
	if !hasCap {
		t.Errorf("hasCap = false, want true")
	}
	if consumed != 90*GiB {
		t.Errorf("consumed = %d, want 90 GiB (%d)", consumed, 90*GiB)
	}
	if free != 30*GiB {
		t.Errorf("free = %d, want 30 GiB (%d)", free, 30*GiB)
	}

	// Attempt to add Instance 3 (40 GiB): 90 + 40 = 130 > 120 (must refuse over-admission)
	inst3 := EvaluateAPUAllocation("inst-3", 5*GiB, meta3, true)
	consumedAll, freeAll, hasCapAll := ArbitrateAPUHeadroom(totalAperture, []APUAllocationRecord{inst1, inst2, inst3})
	if hasCapAll {
		t.Errorf("hasCapAll = true, want false (over-admission prevented)")
	}
	if consumedAll != 130*GiB {
		t.Errorf("consumedAll = %d, want 130 GiB", consumedAll)
	}
	if freeAll != 0 {
		t.Errorf("freeAll = %d, want 0", freeAll)
	}
}
