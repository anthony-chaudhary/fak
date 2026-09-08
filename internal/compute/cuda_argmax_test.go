//go:build cuda && cgo

package compute

import "testing"

func TestCUDAArgmaxExtremeFiniteValues(t *testing.T) {
	cb := cudaOrSkip(t)
	logits := mkResident(cb, []int{4}, []float32{-3e30, -2e30, -2.5e30, -4e30})
	t.Cleanup(func() { cb.Free(logits) })
	if got := cb.Argmax(logits); got != 1 {
		t.Fatalf("Argmax(all finite below -1e30) = %d, want 1", got)
	}
}
