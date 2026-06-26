package model

import (
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestGPTQDequantAndMatRows4BitAnd8Bit(t *testing.T) {
	for _, bits := range []int{4, 8} {
		t.Run("bits"+itoa(bits), func(t *testing.T) {
			out, in, groupSize := 8, 16, 8
			if bits == 8 {
				out, in, groupSize = 4, 8, 4
			}
			codes, zeros, scales := gptqFixtureCodes(bits, out, in, groupSize)
			qt, err := newGPTQTensor(
				packGPTQWeight(codes, bits),
				packGPTQZeros(zeros, bits),
				scales,
				nil,
				out, in, bits, groupSize,
			)
			if err != nil {
				t.Fatalf("newGPTQTensor: %v", err)
			}
			row := make([]float32, in)
			for o := 0; o < out; o++ {
				gptqDequantRow(row, qt, o)
				for i := 0; i < in; i++ {
					g := i / groupSize
					want := (float32(codes[i][o]) - float32(zeros[g][o])) * scales[g*out+o]
					if !closeF32(row[i], want, 1e-6) {
						t.Fatalf("bits=%d row=%d col=%d got %g want %g", bits, o, i, row[i], want)
					}
				}
			}
			x := make([]float32, in)
			for i := range x {
				x[i] = float32(i%5-2) * 0.25
			}
			got := gptqMatRows(qt, x)
			for o := 0; o < out; o++ {
				gptqDequantRow(row, qt, o)
				var want float32
				for i := 0; i < in; i++ {
					want += row[i] * x[i]
				}
				if !closeF32(got[o], want, 1e-5) {
					t.Fatalf("bits=%d matrow[%d] got %g want %g", bits, o, got[o], want)
				}
			}
		})
	}
}

func TestGPTQGIdxSelectsScaleGroup(t *testing.T) {
	const out, in, bits = 8, 8, 4
	codes := make([][]uint32, in)
	for i := range codes {
		codes[i] = make([]uint32, out)
		for o := 0; o < out; o++ {
			codes[i][o] = 9
		}
	}
	zeros := make([][]uint32, 2)
	for g := range zeros {
		zeros[g] = make([]uint32, out)
		for o := 0; o < out; o++ {
			zeros[g][o] = 8
		}
	}
	scales := make([]float32, 2*out)
	for o := 0; o < out; o++ {
		scales[o] = 1
		scales[out+o] = 3
	}
	gidx := []int{1, 0, 1, 0, 1, 0, 1, 0}
	qt, err := newGPTQTensor(packGPTQWeight(codes, bits), packGPTQZeros(zeros, bits), scales, gidx, out, in, bits, 0)
	if err != nil {
		t.Fatalf("newGPTQTensor: %v", err)
	}
	row := make([]float32, in)
	gptqDequantRow(row, qt, 0)
	for i, got := range row {
		want := float32(1)
		if gidx[i] == 1 {
			want = 3
		}
		if got != want {
			t.Fatalf("row[%d]=%g want %g (g_idx group %d)", i, got, want, gidx[i])
		}
	}
}

func TestLoadGPTQSafetensorsRoundTripAndResidentDispatch(t *testing.T) {
	const (
		bits      = 4
		out       = 8
		in        = 16
		groupSize = 8
	)
	codes, zeros, scales := gptqFixtureCodes(bits, out, in, groupSize)
	tensors := map[string]tinySTTensor{
		"proj.qweight":              {dtype: "I32", shape: []int{in / (32 / bits), out}, data: u32TestBytes(packGPTQWeight(codes, bits))},
		"proj.qzeros":               {dtype: "I32", shape: []int{in / groupSize, out / (32 / bits)}, data: u32TestBytes(packGPTQZeros(zeros, bits))},
		"proj.scales":               {dtype: "F32", shape: []int{in / groupSize, out}, data: f32TestBytes(scales)},
		"model.embed_tokens.weight": {dtype: "F32", shape: []int{2, in}, data: f32TestBytes(sequenceFloats(2*in, 0.125))},
	}
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), tinySafetensorsBytes(t, tensors), 0o644); err != nil {
		t.Fatalf("write model.safetensors: %v", err)
	}
	cfg := `{
		"model_type":"llama",
		"hidden_size":16,
		"num_hidden_layers":1,
		"num_attention_heads":2,
		"num_key_value_heads":2,
		"head_dim":8,
		"intermediate_size":32,
		"vocab_size":2,
		"rms_norm_eps":1e-5,
		"quantization_config":{"quant_method":"gptq","bits":4,"group_size":8}
	}`
	if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0o644); err != nil {
		t.Fatalf("write config.json: %v", err)
	}

	m, err := LoadGPTQ(dir)
	if err != nil {
		t.Fatalf("LoadGPTQ: %v", err)
	}
	if m.GPTQCount() != 1 {
		t.Fatalf("GPTQCount=%d want 1", m.GPTQCount())
	}
	gotOut, gotIn, gotBits, gotGroup := m.GPTQShape("proj.weight")
	if gotOut != out || gotIn != in || gotBits != bits || gotGroup != groupSize {
		t.Fatalf("GPTQShape=(%d,%d,%d,%d), want (%d,%d,%d,%d)", gotOut, gotIn, gotBits, gotGroup, out, in, bits, groupSize)
	}
	if !m.has("model.embed_tokens.weight") {
		t.Fatal("LoadGPTQ dropped normal f32 tensor")
	}
	if m.has("proj.scales") || m.has("proj.qzeros") {
		t.Fatal("LoadGPTQ leaked GPTQ auxiliary tensors into the f32 manifest")
	}

	x := sequenceFloats(in, -0.25)
	got := m.residentMatRows("proj.weight", x, out, in)
	want := gptqMatRows(m.gptq("proj.weight"), x)
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("residentMatRows[%d]=%g want %g", i, got[i], want[i])
		}
	}
}

func TestGPTQSessionUsesResidentHead(t *testing.T) {
	cfg := Config{
		HiddenSize:       32,
		NumLayers:        1,
		NumHeads:         4,
		NumKVHeads:       4,
		HeadDim:          8,
		IntermediateSize: 64,
		VocabSize:        8,
		RMSNormEps:       1e-5,
		RopeTheta:        10000,
	}
	m := NewSynthetic(cfg)
	codes, zeros, scales := gptqFixtureCodes(4, cfg.VocabSize, cfg.HiddenSize, 16)
	qt, err := newGPTQTensor(packGPTQWeight(codes, 4), packGPTQZeros(zeros, 4), scales, nil, cfg.VocabSize, cfg.HiddenSize, 4, 16)
	if err != nil {
		t.Fatalf("newGPTQTensor: %v", err)
	}
	m.gptqw = map[string]*gptqTensor{"lm_head.weight": qt}

	s := m.NewSession()
	s.GPTQ = true
	logits := s.Prefill([]int{1, 2, 3})
	if len(logits) != cfg.VocabSize {
		t.Fatalf("Prefill logits len=%d want %d", len(logits), cfg.VocabSize)
	}
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("Prefill logits[%d] is non-finite: %v", i, v)
		}
	}
	next := argmaxF32(logits)
	step := s.Step(next)
	if len(step) != cfg.VocabSize {
		t.Fatalf("Step logits len=%d want %d", len(step), cfg.VocabSize)
	}
	if s.Cache.Len() != 4 {
		t.Fatalf("cache len=%d want 4", s.Cache.Len())
	}
}

func TestGPTQSessionArgmaxExactAgainstDequantizedF32(t *testing.T) {
	cfg := Config{
		HiddenSize:        32,
		NumLayers:         1,
		NumHeads:          4,
		NumKVHeads:        4,
		HeadDim:           8,
		IntermediateSize:  32,
		VocabSize:         16,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: false,
		ModelType:         "llama",
	}
	refTensors, residentTensors, gptqw := tinyExactGPTQOracleTensors(t, cfg)
	ref, err := NewFromF32Tensors(cfg, refTensors)
	if err != nil {
		t.Fatalf("build f32 oracle: %v", err)
	}
	got, err := NewFromF32Tensors(cfg, residentTensors)
	if err != nil {
		t.Fatalf("build resident model: %v", err)
	}
	got.gptqw = gptqw

	refS := ref.NewSession()
	gotS := got.NewSession()
	gotS.GPTQ = true
	refLogits := refS.Prefill([]int{1, 3, 5})
	gotLogits := gotS.Prefill([]int{1, 3, 5})
	for step := 0; step < 4; step++ {
		assertGPTQOracleLogits(t, "step "+itoa(step), refLogits, gotLogits)
		next := argmaxF32(refLogits)
		refLogits = refS.Step(next)
		gotLogits = gotS.Step(next)
	}
	if gotS.Cache.Len() != refS.Cache.Len() {
		t.Fatalf("GPTQ cache len=%d want f32 len=%d", gotS.Cache.Len(), refS.Cache.Len())
	}
}

func tinyExactGPTQOracleTensors(t *testing.T, cfg Config) ([]NamedTensorF32, []NamedTensorF32, map[string]*gptqTensor) {
	t.Helper()
	const bits, groupSize = 4, 16
	H, I, V := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize
	nH, nKV, hd := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	var ref []NamedTensorF32
	var resident []NamedTensorF32
	gptqw := map[string]*gptqTensor{}

	addShared := func(name string, shape []int, data []float32) {
		cpRef := append([]float32(nil), data...)
		cpRes := append([]float32(nil), data...)
		ref = append(ref, NamedTensorF32{Name: name, Shape: append([]int(nil), shape...), Data: cpRef})
		resident = append(resident, NamedTensorF32{Name: name, Shape: append([]int(nil), shape...), Data: cpRes})
	}
	addGPTQ := func(name string, out, in int) {
		qt, f32 := exactGPTQTensorForTest(t, bits, out, in, groupSize)
		gptqw[name] = qt
		ref = append(ref, NamedTensorF32{Name: name, Shape: []int{out, in}, Data: f32})
	}

	addShared("model.embed_tokens.weight", []int{V, H}, sequenceFloats(V*H, 0.03125))
	for l := 0; l < cfg.NumLayers; l++ {
		p := func(s string) string { return layerName(l, s) }
		addShared(p("input_layernorm.weight"), []int{H}, gptqRepeatF32(H, 1))
		addGPTQ(p("self_attn.q_proj.weight"), nH*hd, H)
		addGPTQ(p("self_attn.k_proj.weight"), nKV*hd, H)
		addGPTQ(p("self_attn.v_proj.weight"), nKV*hd, H)
		addGPTQ(p("self_attn.o_proj.weight"), H, nH*hd)
		addShared(p("post_attention_layernorm.weight"), []int{H}, gptqRepeatF32(H, 1))
		addGPTQ(p("mlp.gate_proj.weight"), I, H)
		addGPTQ(p("mlp.up_proj.weight"), I, H)
		addGPTQ(p("mlp.down_proj.weight"), H, I)
	}
	addShared("model.norm.weight", []int{H}, gptqRepeatF32(H, 1))
	addGPTQ("lm_head.weight", V, H)
	return ref, resident, gptqw
}

func exactGPTQTensorForTest(t *testing.T, bits, out, in, groupSize int) (*gptqTensor, []float32) {
	t.Helper()
	codes, zeros, scales := gptqFixtureCodes(bits, out, in, groupSize)
	qt, err := newGPTQTensor(packGPTQWeight(codes, bits), packGPTQZeros(zeros, bits), scales, nil, out, in, bits, groupSize)
	if err != nil {
		t.Fatalf("newGPTQTensor(%d,%d): %v", out, in, err)
	}
	f32 := make([]float32, out*in)
	row := make([]float32, in)
	for o := 0; o < out; o++ {
		gptqDequantRow(row, qt, o)
		copy(f32[o*in:(o+1)*in], row)
	}
	return qt, f32
}

func assertGPTQOracleLogits(t *testing.T, label string, want, got []float32) {
	t.Helper()
	if argmaxF32(got) != argmaxF32(want) {
		t.Fatalf("%s argmax got %d want %d", label, argmaxF32(got), argmaxF32(want))
	}
	if cos := cosineSimilarity(want, got); cos < 0.99999 {
		t.Fatalf("%s cosine=%0.8f < 0.99999", label, cos)
	}
	var maxDiff float32
	for i := range want {
		d := want[i] - got[i]
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
	}
	if maxDiff > 1e-4 {
		t.Fatalf("%s max|diff|=%g > 1e-4", label, maxDiff)
	}
}

func gptqRepeatF32(n int, v float32) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = v
	}
	return x
}

func gptqFixtureCodes(bits, out, in, groupSize int) (codes [][]uint32, zeros [][]uint32, scales []float32) {
	mask := uint32((1 << uint(bits)) - 1)
	nGroups := in / groupSize
	codes = make([][]uint32, in)
	for i := 0; i < in; i++ {
		codes[i] = make([]uint32, out)
		for o := 0; o < out; o++ {
			codes[i][o] = uint32((i*3 + o*5 + 7) % int(mask+1))
		}
	}
	zeros = make([][]uint32, nGroups)
	scales = make([]float32, nGroups*out)
	for g := 0; g < nGroups; g++ {
		zeros[g] = make([]uint32, out)
		for o := 0; o < out; o++ {
			zeros[g][o] = 1 + uint32((g+o)%int(mask))
			scales[g*out+o] = float32(0.03125 * float64(1+g+o%3))
		}
	}
	return codes, zeros, scales
}

func packGPTQWeight(codes [][]uint32, bits int) []uint32 {
	in, out := len(codes), len(codes[0])
	pack := 32 / bits
	qw := make([]uint32, (in/pack)*out)
	mask := uint32((1 << uint(bits)) - 1)
	for i := 0; i < in; i++ {
		for o := 0; o < out; o++ {
			qw[(i/pack)*out+o] |= (codes[i][o] & mask) << (uint(i%pack) * uint(bits))
		}
	}
	return qw
}

func packGPTQZeros(zeros [][]uint32, bits int) []uint32 {
	nGroups, out := len(zeros), len(zeros[0])
	pack := 32 / bits
	qz := make([]uint32, nGroups*(out/pack))
	mask := uint32((1 << uint(bits)) - 1)
	for g := 0; g < nGroups; g++ {
		for o := 0; o < out; o++ {
			stored := (zeros[g][o] - 1) & mask
			qz[g*(out/pack)+o/pack] |= stored << (uint(o%pack) * uint(bits))
		}
	}
	return qz
}

func closeF32(a, b, tol float32) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= tol
}

func i64TestBytes(vals []int64) []byte {
	b := make([]byte, len(vals)*8)
	for i, v := range vals {
		binary.LittleEndian.PutUint64(b[i*8:], uint64(v))
	}
	return b
}
