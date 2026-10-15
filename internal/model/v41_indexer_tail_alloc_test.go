package model

import (
	"fmt"
	"math"
	"testing"
)

// fak-test:runtime fast est=10ms lane=default
func TestV41IndexTailScratch(t *testing.T) {
	const heads, dim, rd = 32, 128, 64
	seed := make([]float32, heads*dim)
	for i := range seed {
		seed[i] = math.Float32frombits(0x3f800000 + uint32(i)*193)
	}
	seed[dim-rd] = math.Float32frombits(0x80000000)
	legacy := func(values []float32) error {
		for h := 0; h < heads; h++ {
			tail := values[(h+1)*dim-rd : (h+1)*dim]
			rounded, err := v41IndexTailScratchLegacyCopy(3, "indexer.wq_b.weight", "rotary", tail)
			if err != nil {
				return err
			}
			copy(tail, rounded)
		}
		return nil
	}
	want, got := make([]float32, len(seed)), make([]float32, len(seed))
	for _, bad := range []uint32{0, 0x7fc00123, 0x7f800000, 0x7f7fffff} {
		copy(want, seed)
		if bad != 0 {
			want[5*dim+dim-rd+7] = math.Float32frombits(bad)
		}
		copy(got, want)
		oldErr := legacy(want)
		err := v41IndexTailsBF16(3, got, heads, dim, rd, nil)
		if (err == nil) != (oldErr == nil) || err != nil && err.Error() != oldErr.Error() {
			t.Fatalf("error mismatch for %08x: got %v, want %v", bad, err, oldErr)
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("copyback mismatch for %08x at %d: got %08x, want %08x", bad, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
			}
		}
	}
	var err error
	before := testing.AllocsPerRun(50, func() { copy(want, seed); err = legacy(want) })
	if err != nil {
		t.Fatal(err)
	}
	after := testing.AllocsPerRun(50, func() { copy(got, seed); err = v41IndexTailsBF16(3, got, heads, dim, rd, nil) })
	if err != nil {
		t.Fatal(err)
	}
	if before-after < heads-1 {
		t.Fatalf("scratch allocations: before=%g after=%g, want at least %d saved", before, after, heads-1)
	}
	t.Logf("tail scratch allocations per call: %g -> %g; exact output and error-state parity", before, after)
}

// Keep the allocating baseline independent of the new scratch helper.
func v41IndexTailScratchLegacyCopy(l int, leaf, phase string, values []float32) ([]float32, error) {
	out := make([]float32, len(values))
	for i, value := range values {
		if finite32(value) {
			out[i] = v41RoundBF16(value)
			if finite32(out[i]) {
				continue
			}
		}
		return nil, &V41ProjectionOperationError{Layer: l, Leaf: leaf, Stage: string(v41StageIndexer), Cause: v41StageErr(v41StageIndexer, l, fmt.Errorf("%w: %s BF16 %s value[%d] is non-finite or overflows", ErrV41ForwardStage, leaf, phase, i))}
	}
	return out, nil
}

// fak-test:runtime fast est=10ms lane=default
func TestV41IndexTailScratchHostRoPE(t *testing.T) {
	const heads, dim, rd = 32, 128, 64
	m := &Model{Cfg: Config{IndexHeadDim: dim, QKRopeHeadDim: rd, RopeTheta: 10000}}
	seed := make([]float32, heads*dim)
	for i := range seed {
		seed[i] = math.Float32frombits(0x3f800000 + uint32(i)*193)
	}
	want, got := make([]float32, len(seed)), make([]float32, len(seed))
	for _, bad := range []uint32{0, 0x7fc00123, 0x7f800000, 0x7f7fffff} {
		copy(want, seed)
		if bad != 0 {
			want[5*dim+dim-rd+7] = math.Float32frombits(bad)
		}
		copy(got, want)
		cos, sin := v41RopeTableForLayer(m.Cfg, 3, 17)
		var oldErr error
		for h := 0; h < heads; h++ {
			head := want[h*dim : (h+1)*dim]
			applyRopeTailInterleaved(head, cos, sin, rd)
			var rounded []float32
			rounded, oldErr = v41IndexTailScratchLegacyCopy(3, "indexer.wq_b.weight", "rotary", head[dim-rd:])
			if oldErr != nil {
				break
			}
			copy(head[dim-rd:], rounded)
		}
		err := m.v41IndexRoPEPublished(3, 17, got, heads, nil, nil)
		if (err == nil) != (oldErr == nil) || err != nil && err.Error() != oldErr.Error() {
			t.Fatalf("RoPE error mismatch for %08x: got %v want %v", bad, err, oldErr)
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("RoPE partial copyback mismatch for %08x at %d", bad, i)
			}
		}
	}
}
