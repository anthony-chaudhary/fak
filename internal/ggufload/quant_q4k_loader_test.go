package ggufload

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestLoadModelQ4KRoutesByIdentityNorm is the small-scale (27B-free) integration test for
// the resident-Q4_K loader. It writes a real 1-layer GGUF with four Q4_K matmul tensors,
// loads it via LoadModelQ4K, and asserts the loader's routing:
//
//   - identity-normalized weights (self_attn.v_proj, mlp.down_proj, lm_head) → resident q4kw
//     (raw Q4_K bytes, no round-trip).
//   - the normalize-sensitive self_attn.q_proj (rotary) → the proven dequant→normalize→Q8
//     path (q8w), NOT q4kw — this is the critical correctness gate: storing q_proj raw would
//     feed wrongly-laid-out weights to the forward.
//
// It also checks ResidentReport reflects the 3/1 split. This is the cheapest witness that
// the loader's NEW routing logic (eligibility + type gate + CanonicalTensorName mapping +
// the dequant+normalize fallback) all fire together correctly on a real GGUF file.
func TestLoadModelQ4KRoutesByIdentityNorm(t *testing.T) {
	const dim = 256 // reduction dim a multiple of qkK → 1 super-block/row
	// Each Q4_K tensor [dim,dim]: 256 super-blocks × 144 B = 36864 B. All-zero blocks are
	// valid Q4_K (d=min=0 → dequant to all zeros); the test checks ROUTING, not values.
	const blkBytes = 36864

	var b bytes.Buffer
	writeMinimalHeader(&b, 4, 8)
	writeKVString(&b, "general.architecture", "llama")
	writeKVUint32(&b, "general.alignment", 32)
	writeKVUint32(&b, "llama.embedding_length", dim)
	writeKVUint32(&b, "llama.block_count", 1)
	writeKVUint32(&b, "llama.attention.head_count", 1)
	writeKVUint32(&b, "llama.attention.key_length", dim)
	writeKVUint32(&b, "llama.feed_forward_length", dim)
	writeKVFloat32(&b, "llama.attention.layer_norm_rms_epsilon", 1e-6)

	writeTensorInfoForTest(&b, "blk.0.attn_v.weight", []uint64{dim, dim}, TensorQ4_K, 0)
	writeTensorInfoForTest(&b, "blk.0.attn_q.weight", []uint64{dim, dim}, TensorQ4_K, blkBytes)
	writeTensorInfoForTest(&b, "blk.0.ffn_down.weight", []uint64{dim, dim}, TensorQ4_K, 2*blkBytes)
	writeTensorInfoForTest(&b, "output.weight", []uint64{dim, dim}, TensorQ4_K, 3*blkBytes)
	padToAlignment(&b, 32)
	for i := 0; i < 4; i++ {
		b.Write(bytes.Repeat([]byte{0}, blkBytes))
	}

	path := filepath.Join(t.TempDir(), "tiny.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	m, err := LoadModelQ4K(path)
	if err != nil {
		t.Fatalf("LoadModelQ4K: %v", err)
	}

	// Identity-normalized weights → q4kw (raw), NOT q8w.
	for _, name := range []string{
		"model.layers.0.self_attn.v_proj.weight",
		"model.layers.0.mlp.down_proj.weight",
		"lm_head.weight",
	} {
		if !m.HasQ4K(name) {
			t.Errorf("expected %s in q4kw (identity-norm → raw)", name)
		}
		if m.HasQ8(name) {
			t.Errorf("%s must NOT be in q8w (it is identity-normalized)", name)
		}
	}
	// Normalize-sensitive q_proj → q8w (dequant→normalize→Q8), NOT q4kw.
	qp := "model.layers.0.self_attn.q_proj.weight"
	if !m.HasQ8(qp) {
		t.Errorf("expected %s in q8w (rotary → normalize-sensitive)", qp)
	}
	if m.HasQ4K(qp) {
		t.Errorf("%s must NOT be in q4kw (rotary weights held raw would corrupt the forward)", qp)
	}
	r := m.ResidentReport()
	if r.Q4KTensors != 3 || r.Q8Tensors != 1 {
		t.Errorf("resident split: q4k=%d q8=%d, want 3/1", r.Q4KTensors, r.Q8Tensors)
	}
}

func TestLoadModelQ4KProfileTicksProgress(t *testing.T) {
	const dim = 256
	const blkBytes = 36864

	var b bytes.Buffer
	writeMinimalHeader(&b, 4, 8)
	writeKVString(&b, "general.architecture", "llama")
	writeKVUint32(&b, "general.alignment", 32)
	writeKVUint32(&b, "llama.embedding_length", dim)
	writeKVUint32(&b, "llama.block_count", 1)
	writeKVUint32(&b, "llama.attention.head_count", 1)
	writeKVUint32(&b, "llama.attention.key_length", dim)
	writeKVUint32(&b, "llama.feed_forward_length", dim)
	writeKVFloat32(&b, "llama.attention.layer_norm_rms_epsilon", 1e-6)

	writeTensorInfoForTest(&b, "blk.0.attn_v.weight", []uint64{dim, dim}, TensorQ4_K, 0)
	writeTensorInfoForTest(&b, "blk.0.attn_q.weight", []uint64{dim, dim}, TensorQ4_K, blkBytes)
	writeTensorInfoForTest(&b, "blk.0.ffn_down.weight", []uint64{dim, dim}, TensorQ4_K, 2*blkBytes)
	writeTensorInfoForTest(&b, "output.weight", []uint64{dim, dim}, TensorQ4_K, 3*blkBytes)
	padToAlignment(&b, 32)
	for i := 0; i < 4; i++ {
		b.Write(bytes.Repeat([]byte{0}, blkBytes))
	}

	path := filepath.Join(t.TempDir(), "tiny.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
	prof := NewLoadProfiler()
	var progress bytes.Buffer
	prof.Progress = &progress
	if _, err := LoadModelQ4KProfile(path, prof); err != nil {
		t.Fatalf("LoadModelQ4KProfile: %v", err)
	}
	if prof.Total != 4 || prof.ggufSeen != 4 {
		t.Fatalf("progress tensors total/seen = %d/%d, want 4/4", prof.Total, prof.ggufSeen)
	}
	if prof.cumBytes != 4*blkBytes {
		t.Fatalf("progress bytes = %d, want %d", prof.cumBytes, 4*blkBytes)
	}
	if out := progress.String(); !strings.Contains(out, "100% (4/4 tensors") {
		t.Fatalf("progress output missing final 100%% line:\n%s", out)
	}
}
