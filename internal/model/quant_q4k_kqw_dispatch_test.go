package model

import (
	"encoding/binary"
	"testing"
)

// TestQ4KKernelRoutesResidentQ6KToKQuant pins the dispatch routing in sessionQ4KKernel.mul:
// a weight that is resident in the k-quant store (kqw) as Q6_K — the q4_k_m expert down_proj,
// which loads Q6_K, NOT Q4_K — must be served by the resident k-quant GEMV (kQuantMatRows),
// NOT the Q8 dequant-and-requantize fallback. The proof is structural: q8w has no entry for
// the name, so if dispatch still fell through to qMatRows(M.q8(name), ...) it would PANIC
// ("q8 tensor not built"). Reaching a result that is bit-identical to a direct kQuantMatRows
// is therefore proof the new kqw branch fired. Without the fix this test panics.
func TestQ4KKernelRoutesResidentQ6KToKQuant(t *testing.T) {
	const (
		out = 5   // odd, to exercise the parallel row-split tail
		in  = 512 // 2 Q6_K super-blocks per row
	)
	nblk := in / qkK
	bb := kindQ6K.blockBytes()
	raw := make([]byte, out*nblk*bb)
	lcgBytes(raw, 0x0fedcba987654321)
	// Set a finite per-super-block scale (d = 1.0) so the decoded weights stay finite.
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			blk := raw[(o*nblk+b)*bb:]
			binary.LittleEndian.PutUint16(blk[q6kBlockBytes-2:], f16One) // d = 1.0 at block end
		}
	}

	m := &Model{
		Cfg:  Config{HiddenSize: in},
		q4kw: map[string]*q4kTensor{}, // intentionally EMPTY for the name
		q8w:  map[string]*q8Tensor{},  // intentionally EMPTY → Q8 fallback would panic
		kqw:  map[string]*kQuantTensor{},
	}
	const name = "model.layers.0.mlp.experts.3.down_proj.weight"
	m.kqw[name] = quantizeKQuantFromRaw(raw, out, in, kindQ6K)

	s := &Session{M: m}
	k := sessionQ4KKernel{s}
	x := make([]float32, in)
	for i := range x {
		x[i] = float32((i*7)%23) - 11
	}

	// Route through the dispatch under test. With the kqw branch present this returns the
	// resident k-quant GEMV; without it this line panics on the missing Q8 tensor.
	got := k.mul(name, k.prep(x), out, in)
	if len(got) != out {
		t.Fatalf("dispatch len=%d want %d", len(got), out)
	}

	// Direct resident k-quant GEMV over the SAME tensor: dispatch must be bit-identical to it.
	want := kQuantMatRows(m.kqw[name], x)
	if len(want) != out {
		t.Fatalf("ref len=%d want %d", len(want), out)
	}
	for o := 0; o < out; o++ {
		if got[o] != want[o] {
			t.Fatalf("kqw dispatch not bit-identical to kQuantMatRows at row %d: got=%v want=%v", o, got, want)
		}
	}
}
