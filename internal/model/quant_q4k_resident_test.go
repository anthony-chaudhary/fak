package model

import (
	"math/rand"
	"testing"
)

// TestResidentQ4KEligibleQwen35Hybrid pins the normalization-correctness routing of the
// resident Q4_K loader: only IDENTITY-normalized matmul weights may be held raw (the GGUF
// bytes are already HF layout for them); every normalize-sensitive weight MUST stay on the
// proven dequant→normalize→Q8 path or the forward gets wrongly-laid-out weights and the
// first token is garbage. This is the cheap (ms) witness for the critical audit gap; the
// full end-to-end witness is cmd/q4kdiag's 248068 first-token check on the 27B.
func TestResidentQ4KEligibleQwen35Hybrid(t *testing.T) {
	cfg := Config{
		HiddenSize: 5120, NumLayers: 4, NumHeads: 24, NumKVHeads: 4, HeadDim: 256,
		LayerTypes:        []string{"linear_attention", "linear_attention", "linear_attention", "full_attention"},
		LinearNumKeyHeads: 16, LinearNumValueHeads: 48, LinearKeyHeadDim: 128, LinearValueHeadDim: 128,
	}
	if !cfg.IsQwen35Hybrid() {
		t.Fatal("fixture must be qwen35 hybrid")
	}
	cases := []struct {
		name string
		want bool
	}{
		// identity-normalized matmul weights → eligible (raw q4k)
		{"model.layers.0.mlp.gate_proj.weight", true},
		{"model.layers.0.mlp.up_proj.weight", true},
		{"model.layers.0.mlp.down_proj.weight", true},
		{"model.layers.3.self_attn.v_proj.weight", true},
		{"model.layers.3.self_attn.o_proj.weight", true},
		{"lm_head.weight", true},

		// normalize-sensitive → NOT eligible (must go Q8): linear-attention family is
		// value-head-reordered (nK=16≠nV=48); full-attn q/k are rotary/gated-unpermuted.
		{"model.layers.0.linear_attn.in_proj_qkv.weight", false},
		{"model.layers.0.linear_attn.in_proj_z.weight", false},
		{"model.layers.0.linear_attn.in_proj_a.weight", false},
		{"model.layers.0.linear_attn.in_proj_b.weight", false},
		{"model.layers.0.linear_attn.out_proj.weight", false},
		{"model.layers.3.self_attn.q_proj.weight", false},
		{"model.layers.3.self_attn.k_proj.weight", false},

		// non-matmul tensors → not eligible (small f32 path)
		{"model.layers.0.input_layernorm.weight", false},
		{"model.embed_tokens.weight", false},
	}
	for _, c := range cases {
		if got := ResidentQ4KEligible(cfg, c.name); got != c.want {
			t.Errorf("ResidentQ4KEligible(%q)=%v want %v", c.name, got, c.want)
		}
	}
}

// TestResidentQ4KEligibleNonHybrid covers the Llama/Qwen2 rotary path: only q/k (which get
// unpermuteRotary) are excluded; the rest of the matmul set is identity and eligible.
func TestResidentQ4KEligibleNonHybrid(t *testing.T) {
	cfg := Config{HiddenSize: 2048, NumLayers: 2, NumHeads: 16, NumKVHeads: 8, HeadDim: 128}
	cases := []struct {
		name string
		want bool
	}{
		{"model.layers.0.self_attn.q_proj.weight", false},
		{"model.layers.0.self_attn.k_proj.weight", false},
		{"model.layers.0.self_attn.v_proj.weight", true},
		{"model.layers.0.self_attn.o_proj.weight", true},
		{"model.layers.0.mlp.down_proj.weight", true},
		{"lm_head.weight", true},
	}
	for _, c := range cases {
		if got := ResidentQ4KEligible(cfg, c.name); got != c.want {
			t.Errorf("ResidentQ4KEligible(%q)=%v want %v", c.name, got, c.want)
		}
	}
}

// TestSessionQ4KKernelMixedDispatch proves the Q4K decode kernel's per-name dispatch
// WITHOUT a 27B: a Model with one matmul weight in q4kw (raw Q4_K) and one in q8w must
// route the first through q4kMatRows and the second through the Q8 GEMV. This is the
// integration seam the resident hybrid model depends on at decode time.
func TestSessionQ4KKernelMixedDispatch(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	// q4kw["a"]: out=2, in=256 (one super-block). Fill with valid (finite-f16) bytes via
	// the same block constraint the q4k arithmetic tests use.
	raw := make([]byte, 2*1*q4kBlockBytes)
	for o := 0; o < 2; o++ {
		blk := raw[o*q4kBlockBytes : (o+1)*q4kBlockBytes]
		// constrained random block (finite d/min) — reuse the peer's helper signature inline
		// (kept local so this test does not depend on quant_q4k_test.go's helpers).
		for i := range blk {
			blk[i] = byte(rng.Intn(256))
		}
		for s := 0; s < 2; s++ {
			hi := blk[s*2+1]
			exp := int((hi >> 2) & 0x1f)
			if exp == 0 || exp == 0x1f {
				exp = 12
			}
			blk[s*2+1] = (hi & 0x83) | byte(exp<<2)
		}
	}
	m := &Model{
		Cfg:      Config{HiddenSize: 256},
		manifest: map[string]tensorMeta{},
		q8w:      map[string]*q8Tensor{},
		q4kw:     map[string]*q4kTensor{},
	}
	m.q4kw["a"] = quantizeQ4KFromRaw(raw, 2, 256)
	// q8w["b"]: out=4, in=32 (one block), all-zero codes/scale (valid Q8_0).
	m.q8w["b"] = &q8Tensor{out: 4, in: 32, nblk: 1, q: make([]int8, 4*32), d: []float32{1, 1, 1, 1}}

	s := &Session{M: m}
	k := sessionQ4KKernel{s}
	x := make([]float32, 256) // reduced over in=256 for "a"

	// "a" → q4kMatRows (the raw resident path), must equal the direct GEMV.
	yA := k.mul("a", k.prep(x), 2, 256)
	yAdirect := q4kMatRows(m.q4kw["a"], x)
	if len(yA) != 2 || yA[0] != yAdirect[0] || yA[1] != yAdirect[1] {
		t.Errorf("q4kw dispatch mismatch: kernel=%v direct=%v", yA, yAdirect)
	}

	// "b" → Q8 GEMV (the normalize-sensitive / Q6_K minority path). in=32.
	xb := make([]float32, 32)
	yB := k.mul("b", k.prep(xb), 4, 32)
	yBdirect := qMatRows(m.q8w["b"], quantizeVecQ8(xb))
	if len(yB) != 4 {
		t.Fatalf("q8 dispatch: len=%d want 4", len(yB))
	}
	for i := range yB {
		if yB[i] != yBdirect[i] {
			t.Errorf("q8 dispatch mismatch at %d: kernel=%v direct=%v", i, yB[i], yBdirect[i])
		}
	}
}
