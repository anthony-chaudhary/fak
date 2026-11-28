package compute

import (
	"math"
	"testing"
)

func TestSelectiveStateUpdateDecode(t *testing.T) {
	t.Parallel()
	if MambaBlockSizeForDState(16) != 16 || MambaBlockSizeForDState(64) != 64 || MambaBlockSizeForDState(256) != 256 {
		t.Fatal("unexpected MambaBlockSizeForDState mapping")
	}
	const batch, slots, dim, dstate = 2, 3, 2, 2
	state := make([]float32, slots*dim*dstate)
	for i := range state {
		state[i] = 1.0
	}
	origSlot1 := append([]float32(nil), state[1*dim*dstate:2*dim*dstate]...)

	// Batch 0 maps to slot 2; batch 1 is a pad row (-1) and must be skipped.
	indices := []int{2, -1}
	x := []float32{1, 2, 99, 99}
	dt := []float32{0.5, 0.5, 0.5, 0.5}
	a := []float32{-1, -1, -1, -1}
	b := []float32{1, 0.5, 0, 0}
	c := []float32{1, 1, 0, 0}
	d := []float32{0.25, 0.5}
	z := []float32{1.0, 0.0}

	out, err := SelectiveStateUpdateDecode(state, indices, x, dt, a, b, c, d, z, batch, dim, dstate)
	if err != nil {
		t.Fatal(err)
	}
	// Pad row output must be zero and slot 1 untouched.
	if out[2] != 0 || out[3] != 0 {
		t.Fatalf("pad row output = %v, want 0", out[2:])
	}
	for i, v := range origSlot1 {
		if state[1*dim*dstate+i] != v {
			t.Fatalf("unused slot 1 mutated at %d", i)
		}
	}
	// Batch 0 dim 1 has z=0 -> silu(0)=0 -> out[1]==0.
	if out[1] != 0 {
		t.Fatalf("z=0 gated output = %v, want 0", out[1])
	}
	// Batch 0 dim 0: dA=exp(-0.5), dB=[0.5,0.25], st=[exp(-0.5)+0.5, exp(-0.5)+0.25], y=(sum(st)+0.25)*silu(1).
	dA := math.Exp(-0.5)
	want0 := float32(((dA + 0.5) + (dA + 0.25) + 0.25) * (1.0 / (1.0 + math.Exp(-1.0))))
	if math.Abs(float64(out[0]-want0)) > 1e-5 {
		t.Fatalf("out[0] = %v, want %v", out[0], want0)
	}
}
