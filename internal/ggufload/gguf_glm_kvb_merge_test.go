package ggufload

import (
	"math"
	"testing"
)

// TestMergeGLMMoeDsaKVB pins the MLA KV-b 2->1 merge layout: attn_k_b is per-head TRANSPOSED
// ([kvLora,qkNope]->[qkNope,kvLora]) while attn_v_b is a straight copy ([vHead,kvLora]), and the
// combined kv_b_proj is [k_nope rows, then v rows] per head. The oracle is the projection the
// native forward computes — kv_b_proj must, when matmul'd against a latent, reproduce the SAME
// per-head k_nope and v projections as the original split tensors. Verified against llama.cpp's
// convert split (k_b transposed, v_b not) + internal/model/glm_dsa.go's per-head consumption.
func TestMergeGLMMoeDsaKVB(t *testing.T) {
	const nH, kvLora, qkNope, vHead = 2, 3, 2, 4

	// Build attn_k_b in model order [nH, kvLora, qkNope] (k_b is stored TRANSPOSED at convert).
	// Build attn_v_b in model order [nH, vHead, kvLora]. Fill with distinct, recoverable values.
	kData := make([]float32, nH*kvLora*qkNope)
	for h := 0; h < nH; h++ {
		for i := 0; i < kvLora; i++ { // input (kv_lora) axis
			for o := 0; o < qkNope; o++ { // output (qk_nope) axis
				kData[h*kvLora*qkNope+i*qkNope+o] = float32(1000*h + 100*o + i) // encode (h,o,i)
			}
		}
	}
	vData := make([]float32, nH*vHead*kvLora)
	for h := 0; h < nH; h++ {
		for o := 0; o < vHead; o++ { // output (v_head) axis
			for i := 0; i < kvLora; i++ { // input (kv_lora) axis
				vData[h*vHead*kvLora+o*kvLora+i] = float32(9000 + 1000*h + 100*o + i)
			}
		}
	}

	merged, err := mergeGLMMoeDsaKVB(0, []int{nH, kvLora, qkNope}, kData, []int{nH, vHead, kvLora}, vData)
	if err != nil {
		t.Fatalf("merge: %v", err)
	}
	wantRows := nH * (qkNope + vHead)
	if merged.Shape[0] != wantRows || merged.Shape[1] != kvLora {
		t.Fatalf("merged shape = %v, want [%d %d]", merged.Shape, wantRows, kvLora)
	}
	if merged.Name != "model.layers.0.self_attn.kv_b_proj.weight" {
		t.Fatalf("merged name = %q", merged.Name)
	}

	// Oracle: the forward (glm_dsa.go) reads kv_b_proj as row-major [nH*(qkNope+vHead), kvLora],
	// per head h slices rows [h*(qkNope+vHead) : +(qkNope+vHead)], takes the first qkNope as k_nope
	// and the rest as v. Row r is dotted against the kvLora latent. So row (h-block, k-row o) MUST
	// equal attn_k_b[h]'s OUTPUT row o (the transpose: kv_b_proj[o][i] == attn_k_b[h][i][o]); and
	// row (h-block, v-row o) MUST equal attn_v_b[h]'s row o verbatim.
	kv := merged.Data
	per := qkNope + vHead
	for h := 0; h < nH; h++ {
		for o := 0; o < qkNope; o++ {
			row := kv[(h*per+o)*kvLora : (h*per+o+1)*kvLora]
			for i := 0; i < kvLora; i++ {
				want := kData[h*kvLora*qkNope+i*qkNope+o] // TRANSPOSE source index
				if math.Abs(float64(row[i]-want)) > 0 {
					t.Fatalf("k row h=%d o=%d i=%d: got %v want %v (transpose wrong)", h, o, i, row[i], want)
				}
			}
		}
		for o := 0; o < vHead; o++ {
			row := kv[(h*per+qkNope+o)*kvLora : (h*per+qkNope+o+1)*kvLora]
			for i := 0; i < kvLora; i++ {
				want := vData[h*vHead*kvLora+o*kvLora+i] // straight copy
				if math.Abs(float64(row[i]-want)) > 0 {
					t.Fatalf("v row h=%d o=%d i=%d: got %v want %v", h, o, i, row[i], want)
				}
			}
		}
	}
}

// TestMergeGLMMoeDsaKVBShapeMismatch confirms the merge fails loud on a bad shape (head/kv_lora
// mismatch or wrong rank) rather than silently producing a corrupt tensor.
func TestMergeGLMMoeDsaKVBShapeMismatch(t *testing.T) {
	good3 := func(d ...int) ([]int, []float32) {
		n := 1
		for _, x := range d {
			n *= x
		}
		return d, make([]float32, n)
	}
	ks, kd := good3(2, 3, 2)
	vs, vd := good3(3, 4, 3) // head count 3 != 2
	if _, err := mergeGLMMoeDsaKVB(0, ks, kd, vs, vd); err == nil {
		t.Fatal("expected head-mismatch error, got nil")
	}
	ks2, kd2 := good3(2, 3, 2)
	if _, err := mergeGLMMoeDsaKVB(0, ks2, kd2, []int{2, 4}, make([]float32, 8)); err == nil {
		t.Fatal("expected rank error on 2-D attn_v_b, got nil")
	}
}

// TestBufferGLMKVBHalfPairing confirms a half is buffered until its partner arrives (order-
// independent) and that an unpaired half is reported.
func TestBufferGLMKVBHalfPairing(t *testing.T) {
	buf := map[int]glmKVBHalf{}
	ks := []int{2, 3, 2}
	vs := []int{2, 4, 3}
	kd := make([]float32, 2*3*2)
	vd := make([]float32, 2*4*3)
	// v first, then k (out of order) — must merge on the second.
	if _, ready, err := bufferGLMKVBHalf(buf, 5, "v", vs, vd); err != nil || ready {
		t.Fatalf("v-first: ready=%v err=%v, want not-ready", ready, err)
	}
	if err := glmKVBUnpaired(buf); err == nil {
		t.Fatal("expected unpaired error with only v buffered")
	}
	merged, ready, err := bufferGLMKVBHalf(buf, 5, "k", ks, kd)
	if err != nil || !ready {
		t.Fatalf("k-second: ready=%v err=%v, want ready", ready, err)
	}
	if merged.Shape[0] != 2*(2+4) {
		t.Fatalf("merged rows = %d", merged.Shape[0])
	}
	if err := glmKVBUnpaired(buf); err != nil {
		t.Fatalf("buffer should be empty after pairing: %v", err)
	}
	// duplicate half must error
	if _, _, err := bufferGLMKVBHalf(buf, 7, "k", ks, kd); err != nil {
		t.Fatalf("first k: %v", err)
	}
	if _, _, err := bufferGLMKVBHalf(buf, 7, "k", ks, kd); err == nil {
		t.Fatal("expected duplicate-k error")
	}
}
