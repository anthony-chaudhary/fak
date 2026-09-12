package compute

import (
	"math"
	"testing"
)

func TestTiledConvSiluFloat32Parity(t *testing.T) {
	const wantBits uint32 = 0x3fe17bea
	for _, tc := range []struct {
		name   string
		kernel int
	}{
		{name: "K1 general", kernel: 1},
		{name: "K3 general", kernel: 3},
		{name: "K4 fast", kernel: 4},
	} {
		t.Run(tc.name, func(t *testing.T) {
			kernel := tc.kernel
			weights := make([]float32, kernel)
			weights[kernel-1] = 1

			got, nextState, err := TiledDepthwiseConv1DChannelMajor(
				[]float32{2}, weights, 1, 1, kernel, nil,
			)
			if err != nil {
				t.Fatalf("K=%d: %v", kernel, err)
			}
			if len(got) != 1 {
				t.Fatalf("K=%d: output shape = %d, want 1", kernel, len(got))
			}
			if bits := math.Float32bits(got[0]); bits != wantBits {
				t.Errorf("K=%d: SiLU(2) bits = %08x, want %08x", kernel, bits, wantBits)
			}

			wantStateLen := kernel - 1
			if len(nextState) != wantStateLen {
				t.Fatalf("K=%d: next state shape = %d, want %d", kernel, len(nextState), wantStateLen)
			}
			if wantStateLen > 0 && nextState[wantStateLen-1] != 2 {
				t.Errorf("K=%d: next state tail = %v, want input 2", kernel, nextState)
			}
		})
	}
}
