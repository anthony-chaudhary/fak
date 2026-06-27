package ggufload

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

// TestParallelQuantLoadDeterministic is the byte-identity gate for the parallel GGUF load
// pipeline (gguf_parload.go): loading the SAME complete glm_moe_dsa GGUF at several worker
// counts must produce a functionally identical model — same resident split AND bit-identical
// forward logits — as the serial (1-worker) load. The pipeline applies builder mutations
// serially in original tensor order, so any worker count is equivalent; this test proves it
// across the expert-split, KV-b-merge, and canonical-dequant branches at once.
func TestParallelQuantLoadDeterministic(t *testing.T) {
	// Block-32-aligned dims (every matmul contraction dim ÷32) so the Q8 builder the
	// resident-Q4_K loader uses for non-resident matmul weights quantizes whole — matching
	// TestGLMMoeDsaGGUFQuantLoadForwards.
	const (
		H, V                = 32, 8
		qLora, kvLora       = 32, 32
		qkNope, qkRope, vHd = 16, 16, 16
		nH                  = 2
		idxHeads, idxDim    = 2, 16
		E, I, sharedI       = 3, 32, 32
	)
	path := filepath.Join(t.TempDir(), "glm_det.gguf")
	if err := os.WriteFile(path, glmMoeDsaFullGGUF(H, V, qLora, kvLora, qkNope, qkRope, vHd, nH, idxHeads, idxDim, E, I, sharedI), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	ids := []int{0, 1, 2, 3, 4} // seq >= index_topk(4) so the DSA indexer has enough keys

	load := func(workers string) ([][]float32, int, int) {
		t.Setenv("FAK_GGUF_LOAD_WORKERS", workers)
		m, err := LoadModelQ4K(path)
		if err != nil {
			t.Fatalf("LoadModelQ4K (workers=%s): %v", workers, err)
		}
		act := m.Forward(ids)
		if act == nil || len(act.Logits) != len(ids) {
			t.Fatalf("workers=%s: Forward returned %d logit rows, want %d", workers, lenLogits(act), len(ids))
		}
		r := m.ResidentReport()
		return act.Logits, r.Q4KTensors, r.Q8Tensors
	}

	wantLogits, wantQ4K, wantQ8 := load("1") // serial reference
	for _, w := range []string{"2", "4", "8", "16"} {
		gotLogits, gotQ4K, gotQ8 := load(w)
		if gotQ4K != wantQ4K || gotQ8 != wantQ8 {
			t.Fatalf("workers=%s resident split = q4k%d/q8%d, want q4k%d/q8%d (parallel load diverged)", w, gotQ4K, gotQ8, wantQ4K, wantQ8)
		}
		if len(gotLogits) != len(wantLogits) {
			t.Fatalf("workers=%s row count %d, want %d", w, len(gotLogits), len(wantLogits))
		}
		for pos := range wantLogits {
			if len(gotLogits[pos]) != len(wantLogits[pos]) {
				t.Fatalf("workers=%s row %d width %d, want %d", w, pos, len(gotLogits[pos]), len(wantLogits[pos]))
			}
			for i := range wantLogits[pos] {
				if gotLogits[pos][i] != wantLogits[pos][i] {
					t.Fatalf("workers=%s logit[%d][%d] = %v, serial = %v (parallel load is NOT byte-identical)",
						w, pos, i, gotLogits[pos][i], wantLogits[pos][i])
				}
			}
		}
	}
}

// TestParallelQuantLoadPathAccounting checks the per-quant-type resident-vs-dequant breakdown
// the parallel collector records (the S4 visibility): a 4-tensor all-Q4_K llama GGUF where 3
// weights are identity-normalized (resident raw) and one (rotary q_proj) takes the dequant→Q8
// path must tally as {Q4_K, dense}: resident=3, dequant=1.
func TestParallelQuantLoadPathAccounting(t *testing.T) {
	const dim = 256
	const blkBytes = 36864 // [dim,dim] Q4_K = 256 super-blocks × 144 B

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
	path := filepath.Join(t.TempDir(), "q4k.gguf")
	if err := os.WriteFile(path, b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}

	// Force >1 worker so the parallel collector path records the accounting.
	t.Setenv("FAK_GGUF_LOAD_WORKERS", "4")
	prof := NewLoadProfiler()
	if _, err := LoadModelQ4KProfile(path, prof); err != nil {
		t.Fatalf("LoadModelQ4KProfile: %v", err)
	}
	rows := prof.loadPathRows()
	var q4kDense *LoadPathStat
	for i := range rows {
		if rows[i].QuantType == "Q4_K" && !rows[i].Expert {
			q4kDense = &rows[i]
		}
	}
	if q4kDense == nil {
		t.Fatalf("load-path breakdown missing a {Q4_K, dense} row; got %+v", rows)
	}
	if q4kDense.ResidentTensors != 3 {
		t.Errorf("Q4_K dense resident tensors = %d, want 3 (v/down/output held raw)", q4kDense.ResidentTensors)
	}
	if q4kDense.DequantTensors != 1 {
		t.Errorf("Q4_K dense dequant tensors = %d, want 1 (rotary q_proj → Q8)", q4kDense.DequantTensors)
	}

	// The summary writer must render the rows (operator-facing visibility).
	var summary bytes.Buffer
	prof.EmitLoadPathSummary(&summary)
	if out := summary.String(); !bytes.Contains([]byte(out), []byte("Q4_K")) || !bytes.Contains([]byte(out), []byte("resident=3")) {
		t.Fatalf("load-path summary missing the Q4_K resident=3 row:\n%s", out)
	}
}
