package compute

import (
	"errors"
	"testing"
)

// TestAdmissionDequantScratch pins the V4 admission pricing (nvidia-cuda-perf-gap
// ATTRIBUTION section 4 V4 / section 8): the P>=128 dequant + fp16 GEMM scratch that
// cudaDequantScratch allocates per large GEMM must be priced into the request plan so
// RefuseMemoryPlanIfTooBig refuses before launch instead of the request dying mid-prefill
// on a typed cudaMalloc miss (witnessed 2026-10-28: 170 MiB FFN-gate [17408 x 5120] fp16
// split, cudaMalloc(178257920); raider-fak-arm-oom-20261028). The assertions close on the
// typed contract — byte values and the *FitError verdict — never on error prose.
func TestAdmissionDequantScratch(t *testing.T) {
	// Witnessed geometry: Qwen3.8-27B FFN gate [17408 x 5120].
	const gateOut, gateIn = 17408, 5120
	wantMain := int64(gateOut*gateIn) * int64(F32.Bytes())  // f32 dequant copy
	wantSplit := int64(gateOut*gateIn) * int64(F16.Bytes()) // witnessed cudaMalloc(178257920)
	wantTotal := wantMain + wantSplit

	t.Run("scratch bytes mirror cudaDequantScratch", func(t *testing.T) {
		if DequantGEMMScratchRows != 128 {
			t.Fatalf("DequantGEMMScratchRows = %d, want the P>=128 arm boundary 128", DequantGEMMScratchRows)
		}
		if wantSplit != 178257920 {
			t.Fatalf("fixture drift: fp16 split = %d, want the witnessed 178257920", wantSplit)
		}
		if got := DequantScratchBytes(gateOut, gateIn); got != wantTotal {
			t.Fatalf("DequantScratchBytes(%d,%d) = %d, want %d (f32 copy %d + fp16 split %d)",
				gateOut, gateIn, got, wantTotal, wantMain, wantSplit)
		}
	})

	cfg := TransformerScratchConfig{
		HiddenSize:       5120,
		IntermediateSize: 17408,
		NumHeads:         24,
		NumKVHeads:       4,
		HeadDim:          256,
	}

	t.Run("estimator prices the widest decoder GEMM", func(t *testing.T) {
		plan := DequantScratchDemand(cfg)
		if len(plan) != 1 {
			t.Fatalf("DequantScratchDemand = %v, want exactly one demand", plan)
		}
		d := plan[0]
		if d.Class != MemoryScratchpad || !d.DeviceScoped() {
			t.Fatalf("demand = %+v, want a device-scope scratchpad demand", d)
		}
		if d.Bytes != wantTotal {
			t.Fatalf("demand bytes = %d, want %d (hidden x intermediate dominates q/kv widths)", d.Bytes, wantTotal)
		}
	})

	t.Run("incomplete geometry fails open", func(t *testing.T) {
		for _, bad := range []TransformerScratchConfig{
			{},
			{HiddenSize: 5120},
			{HiddenSize: 5120, IntermediateSize: 17408},
			{HiddenSize: 5120, IntermediateSize: 17408, NumHeads: 24},
			{HiddenSize: 5120, IntermediateSize: 17408, NumHeads: 24, NumKVHeads: 4},
		} {
			if plan := DequantScratchDemand(bad); plan != nil {
				t.Fatalf("incomplete geometry %+v priced %v, want nil", bad, plan)
			}
		}
	})

	// A 24 GB card with known free = total and zero headroom: exact arithmetic.
	dev := capDevice{total: 24 << 30, free: 24 << 30, known: true}
	const (
		weights = int64(19 << 30)
		kvBig   = int64(5 << 30)
		kvSmall = int64(4 << 30)
	)
	base := MemoryPlan{
		{Class: MemoryWeights, Bytes: weights, Detail: "resident-weights", DType: "mixed"},
		{Class: MemoryKVCache, Bytes: kvBig, Detail: "hal-kv-store", DType: "f32"},
	}

	t.Run("unpriced plan was admitted", func(t *testing.T) {
		if err := RefuseMemoryPlanIfTooBig(dev, base, 0); err != nil {
			t.Fatalf("base plan without the dequant scratch must still admit: %v", err)
		}
	})

	priced := append(append(MemoryPlan(nil), base...), DequantScratchDemand(cfg)...)

	t.Run("priced plan refuses with the typed verdict", func(t *testing.T) {
		err := RefuseMemoryPlanIfTooBig(dev, priced, 0)
		var fe *FitError
		if !errors.As(err, &fe) {
			t.Fatalf("priced plan refusal = %T (%v), want *FitError", err, err)
		}
		if fe.Verdict != FitTooBig {
			t.Fatalf("verdict = %v, want FitTooBig", fe.Verdict)
		}
		wantWant := weights + kvBig + wantTotal
		if fe.Want != wantWant || fe.Avail != dev.total {
			t.Fatalf("refusal sizing = want %d avail %d, want %d / %d", fe.Want, fe.Avail, wantWant, dev.total)
		}
		found := false
		for _, d := range fe.Demands {
			if d.Class == MemoryScratchpad && d.Bytes == wantTotal {
				found = true
			}
		}
		if !found {
			t.Fatalf("refusal demands %v do not carry the priced dequant scratch %d", fe.Demands, wantTotal)
		}
	})

	t.Run("smaller plan still admits", func(t *testing.T) {
		smaller := append(MemoryPlan{
			{Class: MemoryWeights, Bytes: weights, Detail: "resident-weights", DType: "mixed"},
			{Class: MemoryKVCache, Bytes: kvSmall, Detail: "hal-kv-store", DType: "f32"},
		}, DequantScratchDemand(cfg)...)
		if err := RefuseMemoryPlanIfTooBig(dev, smaller, 0); err != nil {
			t.Fatalf("smaller plan with the dequant scratch must admit: %v", err)
		}
	})
}
