package ggufload

import (
	"testing"
)

// TestSplitGLMMoeDsaExpertsQ4KRaw pins the raw-Q4_K expert split: each per-expert byte slice is the
// contiguous block of the [E,out,in] blob, block-aligned (out*in multiple of 256), so no dequant is
// needed. It must equal the same per-expert region the f32 splitter carves, byte for byte.
func TestSplitGLMMoeDsaExpertsQ4KRaw(t *testing.T) {
	const e, out, in = 3, 32, 256 // out*in = 8192 = 32 super-blocks; block-aligned
	per := out * in
	perBytes := (per / q4kSuperBlockWeights) * q4kSuperBlockBytes // 32 * 144 = 4608
	raw := make([]byte, e*perBytes)
	for i := range raw {
		raw[i] = byte((i*7 + 3) & 0xff) // distinct, recoverable pattern
	}
	got, aligned, err := splitGLMMoeDsaExpertsQ4KRaw(5, "gate_proj", []int{e, out, in}, raw)
	if err != nil {
		t.Fatalf("split: %v", err)
	}
	if !aligned {
		t.Fatal("expected block-aligned split, got aligned=false")
	}
	if len(got) != e {
		t.Fatalf("got %d experts, want %d", len(got), e)
	}
	for x := 0; x < e; x++ {
		ex := got[x]
		wantName := "model.layers.5.mlp.experts." + itoa(x) + ".gate_proj.weight"
		if ex.Name != wantName {
			t.Errorf("expert %d name = %q, want %q", x, ex.Name, wantName)
		}
		if ex.Shape[0] != out || ex.Shape[1] != in {
			t.Errorf("expert %d shape = %v, want [%d %d]", x, ex.Shape, out, in)
		}
		if len(ex.Raw) != perBytes {
			t.Fatalf("expert %d raw len = %d, want %d", x, len(ex.Raw), perBytes)
		}
		for i := 0; i < perBytes; i++ {
			if ex.Raw[i] != raw[x*perBytes+i] {
				t.Fatalf("expert %d byte %d = %d, want %d (raw slice misaligned)", x, i, ex.Raw[i], raw[x*perBytes+i])
			}
		}
	}
}

// TestSplitGLMMoeDsaExpertsQ4KRawUnaligned confirms a non-block-aligned out*in returns aligned=false
// (no error), so the loader falls back to the f32 dequant-split.
func TestSplitGLMMoeDsaExpertsQ4KRawUnaligned(t *testing.T) {
	// out*in = 32*100 = 3200, not a multiple of 256.
	_, aligned, err := splitGLMMoeDsaExpertsQ4KRaw(0, "up_proj", []int{2, 32, 100}, make([]byte, 1))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if aligned {
		t.Fatal("expected aligned=false for a non-block-aligned tensor")
	}
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b []byte
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}
