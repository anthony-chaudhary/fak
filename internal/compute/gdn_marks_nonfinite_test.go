package compute

import (
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestGDNSlotInfinitePeakCustomLadder(t *testing.T) {
	kernels := []GDNSlotKernel{{Name: "custom", Bits: 32, PeakCap: float32(math.Inf(1))}}
	for _, sign := range []int{-1, 1} {
		peak := float32(math.Inf(sign))
		if got, err := SelectGDNSlotKernel(peak, kernels); !errors.Is(err, ErrGDNSlotOverflow) || got != (GDNSlotKernel{}) {
			t.Fatalf("infinite peak %v qualified: kernel=%+v err=%v", peak, got, err)
		}
		if got, err := GDNSelectSlotForState([]float32{0, peak}, kernels); !errors.Is(err, ErrGDNSlotOverflow) || got != (GDNSlotKernel{}) {
			t.Fatalf("infinite state %v qualified: kernel=%+v err=%v", peak, got, err)
		}
	}
	// Preserve the existing custom-cap contract for finite inputs, including
	// exact float32 range boundaries. Only non-finite state is refused here.
	for _, peak := range []float32{0, math.MaxFloat32, -math.MaxFloat32} {
		if got, err := SelectGDNSlotKernel(peak, kernels); err != nil || got != kernels[0] {
			t.Fatalf("finite peak %v changed behavior: kernel=%+v err=%v", peak, got, err)
		}
	}
	if _, err := SelectGDNSlotKernel(float32(math.NaN()), kernels); !errors.Is(err, ErrGDNSlotNaN) {
		t.Fatalf("NaN refusal changed: %v", err)
	}
	if _, err := SelectGDNSlotKernel(float32(math.Inf(1)), nil); !errors.Is(err, ErrGDNSlotEmptyLadder) {
		t.Fatalf("empty ladder precedence changed: %v", err)
	}
}
