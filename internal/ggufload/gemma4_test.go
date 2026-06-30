package ggufload

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestGemma4GGUFConfigDerivesHeterogeneousGeometryAndRunsForward witnesses Gemma 4 GGUF
// enablement: the loader derives the per-layer local/global geometry (head_dim, kv-head
// count, RoPE base, window), the Gemma sandwich-norm + GeGLU + embed-scale + final-logit
// soft-cap axes, and the arch-aware canonical tensor mapping (ffn_norm -> pre-feedforward
// norm, post_ffw_norm -> post-feedforward norm, layer_output_scale, rope_freqs); the model
// then runs a finite cacheless forward through BOTH a sliding layer and a global,
// V-less (V = K projection) layer.
func TestGemma4GGUFConfigDerivesHeterogeneousGeometryAndRunsForward(t *testing.T) {
	path := filepath.Join(t.TempDir(), "gemma4.gguf")
	if err := os.WriteFile(path, tinyGemma4GGUF(t), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	c, err := f.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}

	if c.ModelType != "gemma4" || !c.ActGeluTanh || c.BlockTopology.String() != "SandwichNorm" {
		t.Fatalf("gemma4 base axes not derived: type=%q geglu=%v topo=%v", c.ModelType, c.ActGeluTanh, c.BlockTopology)
	}
	if c.NormGain1p {
		t.Fatalf("gemma4 GGUF norms are baked (+1) and use plain RMSNorm: NormGain1p must stay false")
	}
	if c.LogitSoftcap != 30 {
		t.Fatalf("final_logit_softcapping = %v, want 30", c.LogitSoftcap)
	}
	if !c.QKNorm {
		t.Fatalf("gemma4 carries q/k norm tensors; QKNorm must be derived true")
	}
	if len(c.SuppressTokens) != 2 || c.SuppressTokens[0] != 3 || c.SuppressTokens[1] != 4 {
		t.Fatalf("SuppressTokens = %v, want [3 4]", c.SuppressTokens)
	}
	wantEmbed := math.Sqrt(float64(c.HiddenSize))
	if math.Abs(c.EmbedScale-wantEmbed) > 1e-9 {
		t.Fatalf("EmbedScale = %v, want sqrt(hidden)=%v", c.EmbedScale, wantEmbed)
	}
	// Per-layer geometry: layers 0..2 sliding (head_dim 16, kv 2, window 2, theta 1e4),
	// layer 3 global (head_dim 32, kv 1, no window, theta 1e6).
	wantType := []string{"sliding_attention", "sliding_attention", "sliding_attention", "full_attention"}
	wantHD := []int{16, 16, 16, 32}
	wantKV := []int{2, 2, 2, 1}
	wantWin := []int{2, 2, 2, -1}
	wantTheta := []float64{10000, 10000, 10000, 1e6}
	for l := 0; l < 4; l++ {
		if c.LayerTypes[l] != wantType[l] {
			t.Fatalf("layer %d type = %q, want %q", l, c.LayerTypes[l], wantType[l])
		}
		if c.HeadDimPerLayer[l] != wantHD[l] {
			t.Fatalf("layer %d head_dim = %d, want %d", l, c.HeadDimPerLayer[l], wantHD[l])
		}
		if c.NumKVHeadsPerLayer[l] != wantKV[l] {
			t.Fatalf("layer %d kv_heads = %d, want %d", l, c.NumKVHeadsPerLayer[l], wantKV[l])
		}
		if c.Window[l] != wantWin[l] {
			t.Fatalf("layer %d window = %d, want %d", l, c.Window[l], wantWin[l])
		}
		if c.RopeThetaPerLayer[l] != wantTheta[l] {
			t.Fatalf("layer %d rope theta = %v, want %v", l, c.RopeThetaPerLayer[l], wantTheta[l])
		}
	}

	// Canonical tensor mapping: gemma sandwich-norm names + the new gemma4 tensors.
	mustMap := map[string]string{
		"blk.0.ffn_norm.weight":            "model.layers.0.pre_feedforward_layernorm.weight",
		"blk.0.post_ffw_norm.weight":       "model.layers.0.post_feedforward_layernorm.weight",
		"blk.0.post_attention_norm.weight": "model.layers.0.post_attention_layernorm.weight",
		"blk.3.layer_output_scale.weight":  "model.layers.3.layer_output_scale.weight",
		"rope_freqs.weight":                "model.rope_freqs.weight",
	}
	for src, want := range mustMap {
		got, ok := CanonicalTensorNameArch(src, "gemma4")
		if !ok || got != want {
			t.Fatalf("CanonicalTensorNameArch(%q,gemma4) = (%q,%v), want %q", src, got, ok, want)
		}
	}

	// Full f32 forward through both regimes; output must be finite.
	m, err := LoadModel(path)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	act := m.Forward([]int{0, 1, 2})
	if act.Seq != 3 || len(act.Logits) != 3 || len(act.Logits[2]) != 5 {
		t.Fatalf("bad forward shape: seq=%d logits=%dx?", act.Seq, len(act.Logits))
	}
	suppressed := map[int]bool{3: true, 4: true}
	for i, v := range act.Logits[2] {
		if suppressed[i] {
			// Suppress tokens are forced to -inf at the final-logit stage.
			if !math.IsInf(float64(v), -1) {
				t.Fatalf("suppressed logit[%d]=%v, want -inf", i, v)
			}
			continue
		}
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("logit[%d] not finite: %v", i, v)
		}
		// Final-logit soft-cap pins every non-suppressed logit into (-30,30).
		if v <= -30 || v >= 30 {
			t.Fatalf("logit[%d]=%v escaped the final soft-cap (-30,30)", i, v)
		}
	}

	// Q4_K resident-quant load path must also map and run.
	qm, err := LoadModelQuant(path)
	if err != nil {
		t.Fatalf("LoadModelQuant: %v", err)
	}
	qact := qm.Forward([]int{0, 1, 2})
	if len(qact.Logits) != 3 || len(qact.Logits[2]) != 5 {
		t.Fatalf("bad quant forward shape")
	}
	for i, v := range qact.Logits[2] {
		if suppressed[i] {
			continue
		}
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("quant logit[%d] not finite: %v", i, v)
		}
	}
}

func TestGemma3PostFFWNormCanonicalMapping(t *testing.T) {
	mustMap := map[string]string{
		"blk.0.ffn_norm.weight":      "model.layers.0.pre_feedforward_layernorm.weight",
		"blk.0.post_ffw_norm.weight": "model.layers.0.post_feedforward_layernorm.weight",
	}
	for src, want := range mustMap {
		got, ok := CanonicalTensorNameArch(src, "gemma3")
		if !ok || got != want {
			t.Fatalf("CanonicalTensorNameArch(%q,gemma3) = (%q,%v), want %q", src, got, ok, want)
		}
	}
}

var _ = model.SandwichNorm // model is used via f.Config() return type assertions above

// tinyGemma4GGUF builds a minimal but architecturally faithful gemma4 GGUF: 4 layers
// (3 sliding + 1 global), heterogeneous per-layer head_dim/kv-heads, q/k norms, a global
// layer with NO v_proj (V = K), per-layer output-scale tensors, sandwich-norm tensors,
// and a shared rope_freqs vector.
func tinyGemma4GGUF(t *testing.T) []byte {
	t.Helper()
	// Dims are multiples of 32 so the Q8/Q4_K resident-quant load path (reduction dim
	// must be a multiple of 32) also exercises the gemma4 geometry, matching the real
	// checkpoint whose hidden size is 3840.
	const (
		H       = 32 // n_embd
		I       = 32
		V       = 5
		heads   = 2
		hdSWA   = 16 // local head_dim
		hdFull  = 32 // global head_dim
		kvSWA   = 2
		kvFull  = 1
		nLayers = 4
	)
	var tensors []tinyGGUFTensor
	add := func(name string, dims []uint64, data []float32) {
		n := 1
		for _, d := range dims {
			n *= int(d)
		}
		if data == nil {
			data = scaledSequenceF32ForTest(int(len(name)), n)
		}
		tensors = append(tensors, tinyGGUFTensor{name: name, dims: dims, data: data})
	}
	add("token_embd.weight", []uint64{H, V}, nil)
	add("output_norm.weight", []uint64{H}, onesF32ForTest(H))
	add("rope_freqs.weight", []uint64{hdFull / 2}, onesF32ForTest(hdFull/2))

	for l := 0; l < nLayers; l++ {
		p := "blk." + strconv.Itoa(l) + "."
		sliding := l < 3
		hd, kv := hdFull, kvFull
		if sliding {
			hd, kv = hdSWA, kvSWA
		}
		add(p+"attn_norm.weight", []uint64{H}, onesF32ForTest(H))
		add(p+"attn_q.weight", []uint64{H, uint64(heads * hd)}, nil)
		add(p+"attn_k.weight", []uint64{H, uint64(kv * hd)}, nil)
		if sliding {
			add(p+"attn_v.weight", []uint64{H, uint64(kv * hd)}, nil) // global layer omits v_proj
		}
		add(p+"attn_output.weight", []uint64{uint64(heads * hd), H}, nil)
		add(p+"attn_q_norm.weight", []uint64{uint64(hd)}, onesF32ForTest(hd))
		add(p+"attn_k_norm.weight", []uint64{uint64(hd)}, onesF32ForTest(hd))
		add(p+"post_attention_norm.weight", []uint64{H}, onesF32ForTest(H))
		add(p+"ffn_norm.weight", []uint64{H}, onesF32ForTest(H))
		add(p+"ffn_gate.weight", []uint64{H, I}, nil)
		add(p+"ffn_up.weight", []uint64{H, I}, nil)
		add(p+"ffn_down.weight", []uint64{I, H}, nil)
		add(p+"post_ffw_norm.weight", []uint64{H}, onesF32ForTest(H))
		add(p+"layer_output_scale.weight", []uint64{1}, []float32{1.0})
	}

	offsets := make([]uint64, len(tensors))
	var off uint64
	for i, tt := range tensors {
		offsets[i] = off
		off = alignOffset(off+uint64(len(tt.data))*4, 32)
	}

	var b bytes.Buffer
	writeMinimalHeader(&b, uint64(len(tensors)), 22)
	writeKVString(&b, "general.architecture", "gemma4")
	writeKVUint32(&b, "general.alignment", 32)
	writeKVUint64(&b, "gemma4.context_length", 64)
	writeKVUint64(&b, "gemma4.embedding_length", H)
	writeKVUint64(&b, "gemma4.block_count", nLayers)
	writeKVUint64(&b, "gemma4.feed_forward_length", I)
	writeKVUint64(&b, "gemma4.attention.head_count", heads)
	writeKVUint64(&b, "gemma4.attention.key_length", hdFull)
	writeKVUint64(&b, "gemma4.attention.value_length", hdFull)
	writeKVUint64(&b, "gemma4.attention.key_length_swa", hdSWA)
	writeKVUint64(&b, "gemma4.attention.value_length_swa", hdSWA)
	writeKVUint64(&b, "gemma4.attention.sliding_window", 2)
	writeKVFloat32(&b, "gemma4.attention.layer_norm_rms_epsilon", 1e-6)
	writeKVFloat32(&b, "gemma4.rope.freq_base", 1e6)
	writeKVFloat32(&b, "gemma4.rope.freq_base_swa", 10000)
	writeKVUint64(&b, "gemma4.rope.dimension_count", hdFull)
	writeKVUint64(&b, "gemma4.rope.dimension_count_swa", hdSWA)
	writeKVFloat32(&b, "gemma4.final_logit_softcapping", 30)
	writeKVIntArrayForTest(&b, "gemma4.attention.head_count_kv", []int32{kvSWA, kvSWA, kvSWA, kvFull})
	writeKVBoolArrayForTest(&b, "gemma4.attention.sliding_window_pattern", []bool{true, true, true, false})
	writeKVIntArrayForTest(&b, "tokenizer.ggml.suppress_tokens", []int32{3, 4})
	writeKVStringArray(&b, "tokenizer.ggml.tokens", []string{"a", "b", "c", "d", "e"})

	for i, tt := range tensors {
		writeTensorInfoForTest(&b, tt.name, tt.dims, TensorF32, offsets[i])
	}
	padToAlignment(&b, 32)
	dataStart := b.Len()
	for i, tt := range tensors {
		padToLen(&b, dataStart+int(offsets[i]))
		for _, v := range tt.data {
			writeF32ForTest(&b, v)
		}
	}
	return b.Bytes()
}

func writeKVIntArrayForTest(b *bytes.Buffer, key string, values []int32) {
	writeStringForTest(b, key)
	_ = binary.Write(b, binary.LittleEndian, uint32(TypeArray))
	_ = binary.Write(b, binary.LittleEndian, uint32(TypeInt32))
	_ = binary.Write(b, binary.LittleEndian, uint64(len(values)))
	for _, v := range values {
		_ = binary.Write(b, binary.LittleEndian, v)
	}
}

func writeKVBoolArrayForTest(b *bytes.Buffer, key string, values []bool) {
	writeStringForTest(b, key)
	_ = binary.Write(b, binary.LittleEndian, uint32(TypeArray))
	_ = binary.Write(b, binary.LittleEndian, uint32(TypeBool))
	_ = binary.Write(b, binary.LittleEndian, uint64(len(values)))
	for _, v := range values {
		if v {
			b.WriteByte(1)
		} else {
			b.WriteByte(0)
		}
	}
}
