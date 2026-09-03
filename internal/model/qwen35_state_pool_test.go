package model

import (
	"testing"
)

func TestQwen35PreallocatedStateBankWitness(t *testing.T) {
	// First witness requirements (#9956):
	// 1. Preallocate request-indexed GDN state bank at warmup
	// 2. Two active sessions keep stable unit identities across alternating decode steps
	// 3. Release one session unit and verify only that unit is reused
	// 4. Exactly zero state-buffer allocations after bank warmup

	cfg := Config{
		ModelType:           "qwen3_5_text",
		NumLayers:           2,
		LayerTypes:          []string{"linear_attention", "linear_attention"},
		LinearNumKeyHeads:   2,
		LinearKeyHeadDim:    8,
		LinearNumValueHeads: 4,
		LinearValueHeadDim:  8,
		LinearConvKernelDim: 4,
	}

	maxUnits := 2
	bank, err := NewQwen35PreallocatedStateBank(cfg, maxUnits)
	if err != nil {
		t.Fatalf("NewQwen35PreallocatedStateBank failed: %v", err)
	}

	initialReceipt := bank.Receipt()
	if initialReceipt.WarmupAllocations != 8 {
		t.Fatalf("expected 8 warmup allocations, got %d", initialReceipt.WarmupAllocations)
	}
	if initialReceipt.RuntimeAllocations != 0 {
		t.Fatalf("expected 0 runtime allocations, got %d", initialReceipt.RuntimeAllocations)
	}

	// 2. Acquire two active sessions
	u0, layers0, err := bank.Acquire()
	if err != nil || u0 != 0 {
		t.Fatalf("acquire session 0: got unit %d, err %v", u0, err)
	}
	u1, layers1, err := bank.Acquire()
	if err != nil || u1 != 1 {
		t.Fatalf("acquire session 1: got unit %d, err %v", u1, err)
	}

	for step := 0; step < 5; step++ {
		layers0[0].ConvBuffer[0] = float32(100 + step)
		layers0[0].RecurrentBuffer[0] = float32(200 + step)

		layers1[0].ConvBuffer[0] = float32(300 + step)
		layers1[0].RecurrentBuffer[0] = float32(400 + step)

		if layers0[0].ConvBuffer[0] != float32(100+step) {
			t.Fatalf("step %d: session 0 conv buffer corrupted: %v", step, layers0[0].ConvBuffer[0])
		}
		if layers1[0].ConvBuffer[0] != float32(300+step) {
			t.Fatalf("step %d: session 1 conv buffer corrupted: %v", step, layers1[0].ConvBuffer[0])
		}
	}

	// 3. Release session 0 (unit 0)
	if err := bank.Release(u0); err != nil {
		t.Fatalf("release u0 failed: %v", err)
	}

	if layers1[0].ConvBuffer[0] != 304 {
		t.Fatalf("session 1 state altered after session 0 release: %v", layers1[0].ConvBuffer[0])
	}

	uReused, layersReused, err := bank.Acquire()
	if err != nil || uReused != u0 {
		t.Fatalf("expected reuse of unit %d, got unit %d, err %v", u0, uReused, err)
	}

	if layersReused[0].ConvBuffer[0] != 0 {
		t.Fatalf("reused unit 0 buffer not zeroed: %v", layersReused[0].ConvBuffer[0])
	}

	// 4. Verify ZERO runtime allocations
	finalReceipt := bank.Receipt()
	if finalReceipt.RuntimeAllocations != 0 {
		t.Fatalf("runtime allocations after warmup = %d, want 0", finalReceipt.RuntimeAllocations)
	}
}
