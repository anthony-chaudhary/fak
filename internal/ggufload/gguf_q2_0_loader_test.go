package ggufload

// gguf_q2_0_loader_test.go — loader-level witnesses for the #4870 Q2_0 wiring.
// The unit test in internal/model/quant_q2_resident_test.go constructs the
// q2Tensor directly, bypassing BOTH loader hunks; these tests drive the real
// loader seams instead:
//
//  1. residentExpertBlockGeometry admits TensorQ2_0 (gguf_parload.go) — the
//     gate that lets a batched-expert Q2_0 blob take the raw-resident split.
//  2. applyQ4KTensorWork routes a resident Q2_0 pendingTensor to
//     builder.AddResidentQ2 (quant_q4k_loader.go), landing it in the model's
//     ternary store, observable via (*model.Model).Q2Count().
//
// Path taken: the loader-apply path (applyQ4KTensorWork called directly, the
// exact collector-side apply step), NOT the builder fallback — the function is
// callable from a test with a nil profiler (LoadProfiler methods are nil-safe)
// and an empty KV-b merge buffer (the resident branch never touches it).
//
// Note on the tensor name: pendingTensor.name on the real load path is always
// the CANONICAL name (CanonicalTensorNameArch / the expert splitter emit e.g.
// model.layers.L.mlp.down_proj.weight), and AddResidentQ2's eligibility gate
// (residentQuantTarget → isQuantWeight) silently skips non-canonical names, so
// the test uses the canonical form a real load would carry.

import (
	"encoding/binary"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestResidentExpertBlockGeometryAdmitsQ2_0 witnesses hunk 1: the resident
// block-geometry gate returns the ternary group-128 layout (34 B per block)
// for TensorQ2_0 instead of falling back to the f32 dequant→Q8 path.
func TestResidentExpertBlockGeometryAdmitsQ2_0(t *testing.T) {
	blockWeights, blockBytes, ok := residentExpertBlockGeometry(TensorQ2_0)
	if !ok {
		t.Fatal("residentExpertBlockGeometry(TensorQ2_0) ok = false, want true (Q2_0 must be residentable)")
	}
	if blockWeights != 128 {
		t.Errorf("blockWeights = %d, want 128 (ternary group-128 blocks)", blockWeights)
	}
	if blockBytes != blockQ2_0Bytes {
		t.Errorf("blockBytes = %d, want blockQ2_0Bytes (%d)", blockBytes, blockQ2_0Bytes)
	}
	if blockQ2_0Bytes != 34 {
		t.Errorf("blockQ2_0Bytes = %d, want 34 (f16 scale + 128 2-bit codes)", blockQ2_0Bytes)
	}
}

// q2_0TestBlob builds a minimal valid Q2_0 raw payload for [out, 128]: one
// 34-byte block per row — a little-endian f16 scale of 1.0 (0x3C00) followed
// by 32 bytes of packed 2-bit codes (0xe4 = codes 00,01,10,11 low-to-high,
// the golden-reference block pattern).
func q2_0TestBlob(out int) []byte {
	blob := make([]byte, out*blockQ2_0Bytes)
	for row := 0; row < out; row++ {
		blk := blob[row*blockQ2_0Bytes:]
		binary.LittleEndian.PutUint16(blk[:2], 0x3C00) // f16 1.0
		for i := 0; i < 32; i++ {
			blk[2+i] = 0xe4
		}
	}
	return blob
}

// TestApplyQ4KTensorWorkLandsResidentQ2 witnesses hunk 2 end-to-end: a
// resident Q2_0 pendingTensor pushed through the REAL collector apply step
// (applyQ4KTensorWork) reaches builder.AddResidentQ2 and lands in the built
// model's ternary store, counted by Q2Count(). A plain f32 quant weight rides
// the same apply path so Build() has the q8w tensor it requires.
func TestApplyQ4KTensorWorkLandsResidentQ2(t *testing.T) {
	const out, in = 3, 128
	blob := q2_0TestBlob(out)
	if len(blob) != out*1*blockQ2_0Bytes {
		t.Fatalf("fixture blob = %d bytes, want %d (out×1 block×34 B)", len(blob), out*blockQ2_0Bytes)
	}

	cfg := model.Config{ModelType: "qwen35", HiddenSize: in, IntermediateSize: in}
	builder := model.NewQuantBuilder(cfg, false)
	kvbHalf := map[int]glmKVBHalf{}

	// The resident Q2_0 tensor, exactly as the load collector receives it.
	q2Work := tensorWork{
		tickBytes: int64(len(blob)),
		pending: []pendingTensor{{
			resident:     true,
			residentType: TensorQ2_0,
			name:         "model.layers.0.mlp.down_proj.weight",
			shape:        []int{out, in},
			raw:          blob,
		}},
	}
	if err := applyQ4KTensorWork(q2Work, nil, cfg, builder, kvbHalf, false); err != nil {
		t.Fatalf("applyQ4KTensorWork(resident Q2_0): %v", err)
	}

	// One f32 quant weight through the same apply path so Build() succeeds
	// (it refuses a model with an empty q8w store).
	f32Work := tensorWork{
		pending: []pendingTensor{{
			name:  "model.layers.0.self_attn.o_proj.weight",
			shape: []int{in, in},
			f32:   make([]float32, in*in),
		}},
	}
	if err := applyQ4KTensorWork(f32Work, nil, cfg, builder, kvbHalf, false); err != nil {
		t.Fatalf("applyQ4KTensorWork(f32): %v", err)
	}

	m, err := builder.Build()
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	if got := m.Q2Count(); got != 1 {
		t.Fatalf("Q2Count() = %d, want 1 (resident Q2_0 tensor missing from the ternary store)", got)
	}
}
