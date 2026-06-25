package ggufload

import (
	"bytes"
	"math"
	"os"
	"path/filepath"
	"strings"
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

// TestGLMMoeDsaSplitKVBF32Tensors proves the real F32 loader loop intercepts the split GGUF
// names before CanonicalTensorNameArch and materializes the single canonical kv_b_proj tensor.
func TestGLMMoeDsaSplitKVBF32Tensors(t *testing.T) {
	const nH, kvLora, qkNope, vHead = 2, 3, 2, 2
	kData := []float32{
		10, 11, 20, 21, 30, 31,
		110, 111, 120, 121, 130, 131,
	}
	vData := []float32{
		40, 41, 42, 50, 51, 52,
		140, 141, 142, 150, 151, 152,
	}
	want := []float32{
		10, 20, 30,
		11, 21, 31,
		40, 41, 42,
		50, 51, 52,
		110, 120, 130,
		111, 121, 131,
		140, 141, 142,
		150, 151, 152,
	}

	path := filepath.Join(t.TempDir(), "glm_split_kvb.gguf")
	if err := os.WriteFile(path, glmMoeDsaSplitKVBGGUF(nH, kvLora, qkNope, vHead, kData, vData), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	cfg, tensors, err := ws.F32Tensors()
	if err != nil {
		t.Fatalf("F32Tensors: %v", err)
	}
	if cfg.ModelType != "glm_moe_dsa" {
		t.Fatalf("ModelType=%q, want glm_moe_dsa", cfg.ModelType)
	}
	if len(tensors) != 1 {
		t.Fatalf("loaded %d tensors, want only the merged kv_b_proj", len(tensors))
	}
	assertModelTensorForTest(t, map[string]modelTensorForTest{
		tensors[0].Name: {shape: tensors[0].Shape, data: tensors[0].Data},
	}, "model.layers.0.self_attn.kv_b_proj.weight", []int{nH * (qkNope + vHead), kvLora}, want)
}

func TestGLMMoeDsaSplitKVBRejectsUnpairedHalf(t *testing.T) {
	path := filepath.Join(t.TempDir(), "glm_unpaired_kvb.gguf")
	if err := os.WriteFile(path, glmMoeDsaUnpairedKVBGGUF(), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	_, _, err = ws.F32Tensors()
	if err == nil || !strings.Contains(err.Error(), "missing attn_v_b") {
		t.Fatalf("F32Tensors unpaired error = %v, want missing attn_v_b", err)
	}
}

func glmMoeDsaSplitKVBGGUF(nH, kvLora, qkNope, vHead int, kData, vData []float32) []byte {
	tensors := []kvbTestTensor{
		{"blk.0.attn_k_b.weight", []uint64{uint64(qkNope), uint64(kvLora), uint64(nH)}, kData},
		{"blk.0.attn_v_b.weight", []uint64{uint64(kvLora), uint64(vHead), uint64(nH)}, vData},
	}
	return glmMoeDsaKVBOnlyGGUF(nH, kvLora, qkNope, vHead, tensors)
}

func glmMoeDsaUnpairedKVBGGUF() []byte {
	return glmMoeDsaKVBOnlyGGUF(1, 2, 2, 1, []kvbTestTensor{
		{"blk.0.attn_k_b.weight", []uint64{2, 2, 1}, []float32{1, 2, 3, 4}},
	})
}

type kvbTestTensor struct {
	name string
	dims []uint64
	data []float32
}

func glmMoeDsaKVBOnlyGGUF(nH, kvLora, qkNope, vHead int, tensors []kvbTestTensor) []byte {
	align := func(x int) int { return (x + 31) / 32 * 32 }

	var kv bytes.Buffer
	nKV := 0
	ks := func(k, v string) { writeKVString(&kv, k, v); nKV++ }
	ku := func(k string, v uint32) { writeKVUint32(&kv, k, v); nKV++ }
	kf := func(k string, v float32) { writeKVFloat32(&kv, k, v); nKV++ }
	ks("general.architecture", "glm_moe_dsa")
	ku("general.alignment", 32)
	ku("glm_moe_dsa.embedding_length", 4)
	ku("glm_moe_dsa.block_count", 1)
	ku("glm_moe_dsa.attention.head_count", uint32(nH))
	ku("glm_moe_dsa.attention.head_count_kv", uint32(nH))
	ku("glm_moe_dsa.feed_forward_length", 8)
	kf("glm_moe_dsa.attention.layer_norm_rms_epsilon", 1e-5)
	kf("glm_moe_dsa.rope.freq_base", 10000)
	ku("glm_moe_dsa.expert_count", 2)
	ku("glm_moe_dsa.expert_used_count", 1)
	ku("glm_moe_dsa.expert_feed_forward_length", 8)
	ku("glm_moe_dsa.attention.q_lora_rank", 4)
	ku("glm_moe_dsa.attention.kv_lora_rank", uint32(kvLora))
	ku("glm_moe_dsa.attention.qk_nope_head_dim", uint32(qkNope))
	ku("glm_moe_dsa.attention.qk_rope_head_dim", 1)
	ku("glm_moe_dsa.attention.v_head_dim", uint32(vHead))

	var ti bytes.Buffer
	off := 0
	for _, t := range tensors {
		writeTensorInfoForTest(&ti, t.name, t.dims, TensorF32, uint64(off))
		off = align(off + len(t.data)*4)
	}
	var b bytes.Buffer
	writeMinimalHeader(&b, uint64(len(tensors)), uint64(nKV))
	b.Write(kv.Bytes())
	b.Write(ti.Bytes())
	padToAlignment(&b, 32)
	dataStart := b.Len()
	off = 0
	for _, t := range tensors {
		padToLen(&b, dataStart+off)
		for _, v := range t.data {
			writeF32ForTest(&b, v)
		}
		off = align(off + len(t.data)*4)
	}
	return b.Bytes()
}
