package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// This reproducer uses only pre-existing APIs so the identical test executes
// against the pre-fix source. The CPU recorder proves dispatch, not silicon.
// fak-test:runtime medium est=1s lane=default
func TestV41AttentionInputNormDeviceReachability(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("attention-input norm software witness requires cpu-ref")
	}
	m := v41FullStepPatchedNorms(t)
	b := &v41QueryNormTestBackend{v41DenseTestBackend: newV41DenseTestBackend(), outputs: map[compute.Buffer]bool{}}
	s := v41DenseTestSession(t, m, b)
	b.queryWeight = func() compute.Tensor { return s.halW[layerName(0, "attn_norm.weight")] }
	s.Prefill([]int{1, 2})
	s.Step(3)
	if b.norms != 3 {
		t.Fatalf("attention-input device normalization calls=%d, want 3 (two prefill, one decode)", b.norms)
	}
}
