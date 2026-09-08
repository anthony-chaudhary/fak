//go:build darwin && arm64 && cgo

package metalgemm

import (
	"testing"
)

func TestMixedQuantMoEFusedMLPParity(t *testing.T) {
	if !Available() {
		t.Skip("no Metal device available")
	}
	defer ResetQ4K()

	const (
		H  = 256
		Im = 256
		k  = 4 // 4 top-k routed experts
	)

	gws := make([]*Q4KWeight, k)
	uws := make([]*Q4KWeight, k)
	dws := make([]*Q6KWeight, k)

	for i := 0; i < k; i++ {
		gws[i] = UploadQ4K(q4kTestRaw(Im, H, uint64(0x1000+i)), Im, H)
		uws[i] = UploadQ4K(q4kTestRaw(Im, H, uint64(0x2000+i)), Im, H)
		dws[i] = UploadQ6K(make([]byte, H*(Im/256)*210), H, Im)
		if gws[i] == nil || uws[i] == nil || dws[i] == nil {
			t.Fatalf("failed to upload expert %d weights", i)
		}
	}

	x := make([]float32, H)
	for i := range x {
		x[i] = float32(i%17-8) * 0.03125
	}

	// 1. Evaluate single command buffer batch across all k experts
	Ycat := make([]float32, k*H)
	observation := NewExecutionObservation(ExecutionQ4KFusedMLPQ6DownBatch)
	ok := FusedMLPQ6DownBatchWithEvents(gws, uws, dws, x, Ycat, observation)
	if !ok {
		t.Fatal("FusedMLPQ6DownBatchWithEvents failed")
	}

	// 2. Evaluate serial reference per-expert
	for i := 0; i < k; i++ {
		yRef := make([]float32, H)
		okRef := FusedMLPQ6Down(gws[i], uws[i], dws[i], x, yRef)
		if !okRef {
			t.Fatalf("serial FusedMLPQ6Down failed for expert %d", i)
		}
		// 3. Verify numerical output matches serial reference within 1e-4 tolerance
		for j := 0; j < H; j++ {
			got := Ycat[i*H+j]
			want := yRef[j]
			diff := got - want
			if diff < 0 {
				diff = -diff
			}
			if diff > 1e-4 {
				t.Fatalf("expert %d element %d mismatch: got %f, want %f (diff %f)", i, j, got, want, diff)
			}
		}
	}
}
