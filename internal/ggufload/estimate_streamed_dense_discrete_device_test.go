package ggufload

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// The aegis CUDA --cpu-offload-experts serve of DeepSeek-V4.1-Flash Q2_K (FAK_STREAM_Q4K=1, 16 GB
// RTX 5080) logged its device plan against the tagged allocation ledger after the first prefill:
// the plan priced 396,901,056 B of device weights while the ledger held 3,397,380,096 B live. The
// bounded streamed-dense fold had moved every eligible dense k-quant (2,833,336,320 B packed) into
// the host transit row only, yet CUDA keeps each uploaded copy resident in VRAM.
const (
	v41AegisPlannedDeviceWeights = int64(396_901_056)
	v41AegisLedgerDeviceWeights  = int64(3_397_380_096)
	v41DenseEligiblePacked       = int64(2_833_336_320)
	v41StreamResident            = int64(20_701_096_755) // 19.279 GiB, the receipt's resident bound
	v41DenseBound                = int64(2_190_433_157)  // 2.04 GiB, below the eligible side
)

func deviceWeightsBytes(plan compute.MemoryPlan) int64 {
	var n int64
	for _, d := range plan {
		if d.ScopeOrDefault() == compute.MemoryScopeDevice && (d.Class == compute.MemoryWeights || d.Class == compute.MemoryOffload) {
			n += d.Bytes
		}
	}
	return n
}

func TestV41StreamedBoundedDenseDiscreteChargesResidentDenseOnDevice(t *testing.T) {
	ws, err := NewWeightSource(loadV41HeaderFixture(t), nil, 0)
	if err != nil {
		t.Fatalf("NewWeightSource: %v", err)
	}
	shared, err := ws.EstimateCPUOffloadExpertsStreamedBoundedDenseMemoryPlan(1, v41StreamResident, v41DenseBound)
	if err != nil {
		t.Fatalf("shared-pool estimate: %v", err)
	}
	if got := deviceWeightsBytes(shared); got != v41AegisPlannedDeviceWeights {
		t.Fatalf("shared-pool device weights = %d, want the aegis plan's %d (fixture no longer matches the served artifact)", got, v41AegisPlannedDeviceWeights)
	}

	discrete, err := ws.EstimateCPUOffloadExpertsStreamedBoundedDenseDiscreteMemoryPlan(1, v41StreamResident, v41DenseBound)
	if err != nil {
		t.Fatalf("discrete estimate: %v", err)
	}
	if got := memoryPlanBytesByDetail(discrete)["gguf-device-dense-resident"]; got != v41DenseEligiblePacked {
		t.Fatalf("gguf-device-dense-resident = %d, want the eligible dense side %d", got, v41DenseEligiblePacked)
	}
	got := deviceWeightsBytes(discrete)
	if want := v41AegisPlannedDeviceWeights + v41DenseEligiblePacked; got != want {
		t.Fatalf("discrete device weights = %d, want %d", got, want)
	}
	// The residual is the F32 widening of the router and the 31 converted Q2_K tensors, net of the
	// token_embd the CUDA arm never uploads: under 5% of the ledger, against the 88% gap before.
	if miss := v41AegisLedgerDeviceWeights - got; miss < 0 || miss*20 > v41AegisLedgerDeviceWeights {
		t.Fatalf("discrete device weights %d miss the ledger's %d by %d, want within 5%%", got, v41AegisLedgerDeviceWeights, miss)
	}
	if shared.HostTotal() != discrete.HostTotal() {
		t.Fatalf("host total moved: shared %d, discrete %d; the host transit bound must be unchanged", shared.HostTotal(), discrete.HostTotal())
	}
}

func TestStreamedBoundedDenseDiscreteVariantAddsOnlyTheDeviceRow(t *testing.T) {
	ws := streamedBoundedDenseTestWeightSource(t)
	const streamResident, denseResident = int64(64 << 10), int64(8 << 10)
	shared, err := ws.EstimateCPUOffloadExpertsStreamedBoundedDenseMemoryPlan(1, streamResident, denseResident)
	if err != nil {
		t.Fatal(err)
	}
	discrete, err := ws.EstimateCPUOffloadExpertsStreamedBoundedDenseDiscreteMemoryPlan(1, streamResident, denseResident)
	if err != nil {
		t.Fatal(err)
	}
	if len(discrete) != len(shared)+1 {
		t.Fatalf("discrete plan has %d rows, want the shared %d plus one device row", len(discrete), len(shared))
	}
	for i := range shared {
		if discrete[i] != shared[i] {
			t.Fatalf("row %d: %+v != shared %+v", i, discrete[i], shared[i])
		}
	}
	last := discrete[len(discrete)-1]
	if last.Detail != "gguf-device-dense-resident" || last.Scope != compute.MemoryScopeDevice || last.Bytes != streamedDenseEligibleBytes {
		t.Fatalf("device row = %+v, want gguf-device-dense-resident %d on the device", last, streamedDenseEligibleBytes)
	}
	if _, err := ws.EstimateCPUOffloadExpertsStreamedBoundedDenseDiscreteMemoryPlan(1, streamResident, -1); err == nil {
		t.Fatal("negative dense bound was accepted")
	}
}
