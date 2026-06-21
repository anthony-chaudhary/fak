package model

import (
	"math"
	"path/filepath"
	"testing"
)

// TestLoadSafetensorsQuantMatchesRegular pins the memory-lean loader to the regular path: a
// quantize-at-load Model must produce Q8 weights bit-identical to LoadSafetensors + Quantize
// (same quantizeQ8 over the same bf16 decode), must DROP the f32 of the big matmul weights
// (the memory win), and must decode bit-for-bit identically. Skips without the cached
// SmolLM2 safetensors (same fixture the rest of safetensors_test uses).
func TestLoadSafetensorsQuantMatchesRegular(t *testing.T) {
	var cfg Config
	if err := readJSON(filepath.Join(cacheDir, "config.json"), &cfg); err != nil {
		t.Skip("no cache config; run export_oracle.py")
	}
	path := findSafetensors(t) // skips if the HF snapshot is not cached

	reg, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	reg.Quantize()

	lean, err := LoadSafetensorsQuant(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuant: %v", err)
	}

	// 1. every quantized weight is bit-identical (codes + scales).
	if len(lean.q8w) != len(reg.q8w) {
		t.Fatalf("q8w count: lean %d != reg %d", len(lean.q8w), len(reg.q8w))
	}
	for name, a := range reg.q8w {
		b, ok := lean.q8w[name]
		if !ok {
			t.Fatalf("lean missing q8 tensor %s", name)
		}
		if a.out != b.out || a.in != b.in || a.nblk != b.nblk {
			t.Fatalf("%s shape: reg(%d,%d,%d) != lean(%d,%d,%d)", name, a.out, a.in, a.nblk, b.out, b.in, b.nblk)
		}
		for i := range a.q {
			if a.q[i] != b.q[i] {
				t.Fatalf("%s code[%d]: reg %d != lean %d", name, i, a.q[i], b.q[i])
			}
		}
		for i := range a.d {
			if math.Float32bits(a.d[i]) != math.Float32bits(b.d[i]) {
				t.Fatalf("%s scale[%d]: reg %v != lean %v", name, i, a.d[i], b.d[i])
			}
		}
	}

	// 2. the memory win: lean must NOT retain f32 for the big matmul weights, but MUST keep
	//    the small f32 tensors the Q8 forward path reads directly (embed, norms, biases).
	for name := range reg.q8w {
		if isQuantWeight(name) && lean.has(name) {
			t.Errorf("lean retains f32 for %s — no memory win", name)
		}
	}
	for _, small := range []string{"model.embed_tokens.weight", "model.norm.weight"} {
		if !lean.has(small) {
			t.Errorf("lean dropped %s — the Q8 path needs it f32", small)
		}
	}

	// 3. decode is bit-for-bit identical (same Q8 weights + same f32 embed/norm/bias).
	ids := []int{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	sReg := reg.NewSession()
	sReg.Quant = true
	sLean := lean.NewSession()
	sLean.Quant = true
	lr := sReg.Prefill(ids)
	ll := sLean.Prefill(ids)
	for i := range lr {
		if math.Float32bits(lr[i]) != math.Float32bits(ll[i]) {
			t.Fatalf("prefill logit[%d]: reg %v != lean %v (not bit-identical)", i, lr[i], ll[i])
		}
	}
	id := argmax(lr)
	for step := 0; step < 8; step++ {
		a := sReg.Step(id)
		b := sLean.Step(id)
		if argmax(a) != argmax(b) {
			t.Fatalf("decode step %d: reg argmax %d != lean argmax %d", step, argmax(a), argmax(b))
		}
		id = argmax(a)
	}
	// 4. the dir dispatcher routes a no-index snapshot to the single-file path, byte-identically.
	dirM, err := LoadSafetensorsQuantDir(filepath.Dir(path), cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuantDir: %v", err)
	}
	if len(dirM.q8w) != len(lean.q8w) {
		t.Fatalf("dir vs file q8w count: %d != %d", len(dirM.q8w), len(lean.q8w))
	}
	for name, a := range lean.q8w {
		b, ok := dirM.q8w[name]
		if !ok {
			t.Fatalf("dir load missing q8 tensor %s", name)
		}
		for i := range a.q {
			if a.q[i] != b.q[i] {
				t.Fatalf("dir vs file %s code[%d] differ", name, i)
			}
		}
	}

	t.Logf("lean loader: %d Q8 tensors bit-identical, f32 big-weights dropped, decode bit-identical, dir-dispatch ok", len(lean.q8w))
}

func TestLoadSafetensorsQuantCanonicalMoEWeights(t *testing.T) {
	cfg := Config{
		HiddenSize:        32,
		NumLayers:         1,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           8,
		IntermediateSize:  64,
		VocabSize:         64,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		NumExperts:        2,
		NumExpertsPerTok:  1,
	}
	tensors := map[string]tinySTTensor{
		"model.embed_tokens.weight": {
			dtype: "F32",
			shape: []int{cfg.VocabSize, cfg.HiddenSize},
			data:  f32TestBytes(sequenceFloats(cfg.VocabSize*cfg.HiddenSize, 0.03)),
		},
		"model.norm.weight": {
			dtype: "F32",
			shape: []int{cfg.HiddenSize},
			data:  f32TestBytes(repeatFloat32(cfg.HiddenSize, 1)),
		},
		layerName(0, "input_layernorm.weight"): {
			dtype: "F32",
			shape: []int{cfg.HiddenSize},
			data:  f32TestBytes(repeatFloat32(cfg.HiddenSize, 1)),
		},
		layerName(0, "post_attention_layernorm.weight"): {
			dtype: "F32",
			shape: []int{cfg.HiddenSize},
			data:  f32TestBytes(repeatFloat32(cfg.HiddenSize, 1)),
		},
		layerName(0, "self_attn.q_proj.weight"): {
			dtype: "F32",
			shape: []int{cfg.NumHeads * cfg.HeadDim, cfg.HiddenSize},
			data:  f32TestBytes(sequenceFloats(cfg.NumHeads*cfg.HeadDim*cfg.HiddenSize, 0.01)),
		},
		layerName(0, "self_attn.k_proj.weight"): {
			dtype: "F32",
			shape: []int{cfg.NumKVHeads * cfg.HeadDim, cfg.HiddenSize},
			data:  f32TestBytes(sequenceFloats(cfg.NumKVHeads*cfg.HeadDim*cfg.HiddenSize, 0.02)),
		},
		layerName(0, "self_attn.v_proj.weight"): {
			dtype: "F32",
			shape: []int{cfg.NumKVHeads * cfg.HeadDim, cfg.HiddenSize},
			data:  f32TestBytes(sequenceFloats(cfg.NumKVHeads*cfg.HeadDim*cfg.HiddenSize, 0.04)),
		},
		layerName(0, "self_attn.o_proj.weight"): {
			dtype: "F32",
			shape: []int{cfg.HiddenSize, cfg.NumHeads * cfg.HeadDim},
			data:  f32TestBytes(sequenceFloats(cfg.HiddenSize*cfg.NumHeads*cfg.HeadDim, 0.05)),
		},
		routerName(0): {
			dtype: "F32",
			shape: []int{cfg.NumExperts, cfg.HiddenSize},
			data:  f32TestBytes(sequenceFloats(cfg.NumExperts*cfg.HiddenSize, 0.06)),
		},
	}
	for e := 0; e < cfg.NumExperts; e++ {
		tensors[expertName(0, e, "gate_proj.weight")] = tinySTTensor{
			dtype: "F32",
			shape: []int{cfg.IntermediateSize, cfg.HiddenSize},
			data:  f32TestBytes(sequenceFloats(cfg.IntermediateSize*cfg.HiddenSize, float32(e)+0.07)),
		}
		tensors[expertName(0, e, "up_proj.weight")] = tinySTTensor{
			dtype: "F32",
			shape: []int{cfg.IntermediateSize, cfg.HiddenSize},
			data:  f32TestBytes(sequenceFloats(cfg.IntermediateSize*cfg.HiddenSize, float32(e)+0.08)),
		}
		tensors[expertName(0, e, "down_proj.weight")] = tinySTTensor{
			dtype: "F32",
			shape: []int{cfg.HiddenSize, cfg.IntermediateSize},
			data:  f32TestBytes(sequenceFloats(cfg.HiddenSize*cfg.IntermediateSize, float32(e)+0.09)),
		}
	}

	m, err := LoadSafetensorsQuant(writeTinySafetensors(t, tensors), cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuant: %v", err)
	}
	for _, name := range []string{
		routerName(0),
		expertName(0, 0, "gate_proj.weight"),
		expertName(0, 0, "up_proj.weight"),
		expertName(0, 0, "down_proj.weight"),
	} {
		if m.q8w[name] == nil {
			t.Fatalf("quant loader did not build Q8 tensor %s", name)
		}
		if m.has(name) {
			t.Fatalf("quant loader retained f32 copy of %s", name)
		}
	}

	s := m.NewSession()
	s.Quant = true
	logits := s.Prefill([]int{3, 17, 5, 23})
	if len(logits) != cfg.VocabSize {
		t.Fatalf("logits len = %d, want %d", len(logits), cfg.VocabSize)
	}
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("Q8 MoE quant-load logit[%d] not finite: %v", i, v)
		}
	}
}

func TestLoadSafetensorsQuantFusedProjectionWeights(t *testing.T) {
	cfg := Config{
		HiddenSize:        32,
		NumLayers:         1,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           8,
		IntermediateSize:  64,
		VocabSize:         64,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		ModelType:         "phi3",
	}
	qRows := cfg.NumHeads * cfg.HeadDim
	kvRows := cfg.NumKVHeads * cfg.HeadDim
	tensors := map[string]tinySTTensor{
		"model.embed_tokens.weight": tinyF32Tensor([]int{cfg.VocabSize, cfg.HiddenSize},
			sequenceFloats(cfg.VocabSize*cfg.HiddenSize, 0.03)),
		"model.norm.weight": tinyF32Tensor([]int{cfg.HiddenSize}, repeatFloat32(cfg.HiddenSize, 1)),
		layerName(0, "input_layernorm.weight"): tinyF32Tensor([]int{cfg.HiddenSize},
			repeatFloat32(cfg.HiddenSize, 1)),
		layerName(0, "post_attention_layernorm.weight"): tinyF32Tensor([]int{cfg.HiddenSize},
			repeatFloat32(cfg.HiddenSize, 1)),
		layerName(0, suffixQKVProj): tinyF32Tensor([]int{qRows + 2*kvRows, cfg.HiddenSize},
			sequenceFloats((qRows+2*kvRows)*cfg.HiddenSize, 0.01)),
		layerName(0, "self_attn.o_proj.weight"): tinyF32Tensor([]int{cfg.HiddenSize, qRows},
			sequenceFloats(cfg.HiddenSize*qRows, 0.05)),
		layerName(0, suffixGateUpProj): tinyF32Tensor([]int{2 * cfg.IntermediateSize, cfg.HiddenSize},
			sequenceFloats(2*cfg.IntermediateSize*cfg.HiddenSize, 0.07)),
		layerName(0, "mlp.down_proj.weight"): tinyF32Tensor([]int{cfg.HiddenSize, cfg.IntermediateSize},
			sequenceFloats(cfg.HiddenSize*cfg.IntermediateSize, 0.09)),
	}

	m, err := LoadSafetensorsQuant(writeTinySafetensors(t, tensors), cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuant: %v", err)
	}
	for _, tt := range []struct {
		name string
		out  int
		in   int
	}{
		{layerName(0, "self_attn.q_proj.weight"), qRows, cfg.HiddenSize},
		{layerName(0, "self_attn.k_proj.weight"), kvRows, cfg.HiddenSize},
		{layerName(0, "self_attn.v_proj.weight"), kvRows, cfg.HiddenSize},
		{layerName(0, "mlp.gate_proj.weight"), cfg.IntermediateSize, cfg.HiddenSize},
		{layerName(0, "mlp.up_proj.weight"), cfg.IntermediateSize, cfg.HiddenSize},
	} {
		assertQ8Shape(t, m, tt.name, tt.out, tt.in)
	}
	for _, name := range []string{
		layerName(0, suffixQKVProj),
		layerName(0, suffixGateUpProj),
	} {
		if m.has(name) {
			t.Fatalf("quant loader retained fused f32 tensor %s", name)
		}
		if m.q8w[name] != nil {
			t.Fatalf("quant loader built fused-name Q8 tensor %s", name)
		}
	}
	assertQuantPrefillFinite(t, m, cfg)
}

func TestLoadSafetensorsQuantGPTOSSPackedMoEWeights(t *testing.T) {
	cfg := Config{
		HiddenSize:        32,
		NumLayers:         1,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           8,
		IntermediateSize:  64,
		VocabSize:         64,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		ModelType:         "gpt_oss",
		NumExperts:        2,
		NumExpertsPerTok:  1,
	}
	tensors := tinyQuantMoEBaseTensors(cfg)
	sourceNames := []string{
		layerName(0, "mlp.router.weight"),
		layerName(0, "mlp.router.bias"),
		layerName(0, "mlp.experts.gate_up_proj.weight"),
		layerName(0, "mlp.experts.gate_up_proj.bias"),
		layerName(0, "mlp.experts.down_proj"),
		layerName(0, "mlp.experts.down_proj_bias"),
	}
	tensors[sourceNames[0]] = tinyF32Tensor([]int{cfg.NumExperts, cfg.HiddenSize},
		sequenceFloats(cfg.NumExperts*cfg.HiddenSize, 0.06))
	tensors[sourceNames[1]] = tinyF32Tensor([]int{cfg.NumExperts},
		sequenceFloats(cfg.NumExperts, 0.006))
	tensors[sourceNames[2]] = tinyF32Tensor([]int{cfg.NumExperts, cfg.HiddenSize, 2 * cfg.IntermediateSize},
		sequenceFloats(cfg.NumExperts*cfg.HiddenSize*2*cfg.IntermediateSize, 0.007))
	tensors[sourceNames[3]] = tinyF32Tensor([]int{cfg.NumExperts, 2 * cfg.IntermediateSize},
		sequenceFloats(cfg.NumExperts*2*cfg.IntermediateSize, 0.008))
	tensors[sourceNames[4]] = tinyF32Tensor([]int{cfg.NumExperts, cfg.IntermediateSize, cfg.HiddenSize},
		sequenceFloats(cfg.NumExperts*cfg.IntermediateSize*cfg.HiddenSize, 0.009))
	tensors[sourceNames[5]] = tinyF32Tensor([]int{cfg.NumExperts, cfg.HiddenSize},
		sequenceFloats(cfg.NumExperts*cfg.HiddenSize, 0.01))

	m, err := LoadSafetensorsQuant(writeTinySafetensors(t, tensors), cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuant: %v", err)
	}
	assertQ8Shape(t, m, routerName(0), cfg.NumExperts, cfg.HiddenSize)
	assertTensorShape(t, m, routerBiasName(0), []int{cfg.NumExperts})
	for e := 0; e < cfg.NumExperts; e++ {
		assertQ8Shape(t, m, expertName(0, e, "gate_proj.weight"), cfg.IntermediateSize, cfg.HiddenSize)
		assertQ8Shape(t, m, expertName(0, e, "up_proj.weight"), cfg.IntermediateSize, cfg.HiddenSize)
		assertQ8Shape(t, m, expertName(0, e, "down_proj.weight"), cfg.HiddenSize, cfg.IntermediateSize)
		assertTensorShape(t, m, expertName(0, e, "gate_proj.bias"), []int{cfg.IntermediateSize})
		assertTensorShape(t, m, expertName(0, e, "up_proj.bias"), []int{cfg.IntermediateSize})
		assertTensorShape(t, m, expertName(0, e, "down_proj.bias"), []int{cfg.HiddenSize})
	}
	for _, name := range sourceNames {
		if m.has(name) {
			t.Fatalf("quant loader retained source f32 tensor %s", name)
		}
		if m.q8w[name] != nil {
			t.Fatalf("quant loader built source-name Q8 tensor %s", name)
		}
	}
	assertQuantPrefillFinite(t, m, cfg)
}

func TestLoadSafetensorsQuantGPTOSSMXFP4Experts(t *testing.T) {
	cfg := Config{
		HiddenSize:        32,
		NumLayers:         1,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           8,
		IntermediateSize:  32,
		VocabSize:         64,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		ModelType:         "gpt_oss",
		NumExperts:        1,
		NumExpertsPerTok:  1,
	}
	tensors := tinyQuantMoEBaseTensors(cfg)
	tensors[layerName(0, "mlp.router.weight")] = tinyF32Tensor([]int{cfg.NumExperts, cfg.HiddenSize},
		sequenceFloats(cfg.NumExperts*cfg.HiddenSize, 0.06))
	sourceNames := []string{
		layerName(0, "mlp.experts.gate_up_proj_blocks"),
		layerName(0, "mlp.experts.gate_up_proj_scales"),
		layerName(0, "mlp.experts.down_proj_blocks"),
		layerName(0, "mlp.experts.down_proj_scales"),
	}
	tensors[sourceNames[0]] = tinySTTensor{
		dtype: "U8",
		shape: []int{cfg.NumExperts, 2 * cfg.IntermediateSize, 1, cfg.HiddenSize / 2},
		data:  repeatByte(cfg.NumExperts*2*cfg.IntermediateSize*cfg.HiddenSize/2, 0x21),
	}
	tensors[sourceNames[1]] = tinySTTensor{
		dtype: "U8",
		shape: []int{cfg.NumExperts, 2 * cfg.IntermediateSize, 1},
		data:  repeatByte(cfg.NumExperts*2*cfg.IntermediateSize, 127),
	}
	tensors[sourceNames[2]] = tinySTTensor{
		dtype: "U8",
		shape: []int{cfg.NumExperts, cfg.HiddenSize, 1, cfg.IntermediateSize / 2},
		data:  repeatByte(cfg.NumExperts*cfg.HiddenSize*cfg.IntermediateSize/2, 0x43),
	}
	tensors[sourceNames[3]] = tinySTTensor{
		dtype: "U8",
		shape: []int{cfg.NumExperts, cfg.HiddenSize, 1},
		data:  repeatByte(cfg.NumExperts*cfg.HiddenSize, 127),
	}

	m, err := LoadSafetensorsQuant(writeTinySafetensors(t, tensors), cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuant: %v", err)
	}
	assertQ8Shape(t, m, routerName(0), cfg.NumExperts, cfg.HiddenSize)
	assertQ8Shape(t, m, expertName(0, 0, "gate_proj.weight"), cfg.IntermediateSize, cfg.HiddenSize)
	assertQ8Shape(t, m, expertName(0, 0, "up_proj.weight"), cfg.IntermediateSize, cfg.HiddenSize)
	assertQ8Shape(t, m, expertName(0, 0, "down_proj.weight"), cfg.HiddenSize, cfg.IntermediateSize)
	for _, name := range append(sourceNames,
		layerName(0, "mlp.experts.gate_up_proj"),
		layerName(0, "mlp.experts.down_proj"),
	) {
		if m.has(name) {
			t.Fatalf("quant loader retained source f32 tensor %s", name)
		}
		if m.q8w[name] != nil {
			t.Fatalf("quant loader built source-name Q8 tensor %s", name)
		}
	}
	assertQuantPrefillFinite(t, m, cfg)
}

func TestLoadSafetensorsQuantMixtralBlockSparseMoEWeights(t *testing.T) {
	cfg := Config{
		HiddenSize:        32,
		NumLayers:         1,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           8,
		IntermediateSize:  64,
		VocabSize:         64,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		ModelType:         "mixtral",
		NumExperts:        2,
		NumExpertsPerTok:  1,
	}
	tensors := tinyQuantMoEBaseTensors(cfg)
	sourceNames := []string{layerName(0, "block_sparse_moe.gate.weight")}
	tensors[sourceNames[0]] = tinyF32Tensor([]int{cfg.NumExperts, cfg.HiddenSize},
		sequenceFloats(cfg.NumExperts*cfg.HiddenSize, 0.06))
	for e := 0; e < cfg.NumExperts; e++ {
		prefix := layerName(0, "block_sparse_moe.experts."+itoa(e)+".")
		for _, tt := range []struct {
			source string
			shape  []int
			seed   float32
		}{
			{prefix + "w1.weight", []int{cfg.IntermediateSize, cfg.HiddenSize}, float32(e) + 0.07},
			{prefix + "w2.weight", []int{cfg.HiddenSize, cfg.IntermediateSize}, float32(e) + 0.08},
			{prefix + "w3.weight", []int{cfg.IntermediateSize, cfg.HiddenSize}, float32(e) + 0.09},
		} {
			sourceNames = append(sourceNames, tt.source)
			tensors[tt.source] = tinyF32Tensor(tt.shape, sequenceFloats(tensorElemCount(tt.shape), tt.seed))
		}
	}

	m, err := LoadSafetensorsQuant(writeTinySafetensors(t, tensors), cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuant: %v", err)
	}
	assertQ8Shape(t, m, routerName(0), cfg.NumExperts, cfg.HiddenSize)
	for e := 0; e < cfg.NumExperts; e++ {
		assertQ8Shape(t, m, expertName(0, e, "gate_proj.weight"), cfg.IntermediateSize, cfg.HiddenSize)
		assertQ8Shape(t, m, expertName(0, e, "up_proj.weight"), cfg.IntermediateSize, cfg.HiddenSize)
		assertQ8Shape(t, m, expertName(0, e, "down_proj.weight"), cfg.HiddenSize, cfg.IntermediateSize)
	}
	for _, name := range sourceNames {
		if m.has(name) {
			t.Fatalf("quant loader retained source f32 tensor %s", name)
		}
		if m.q8w[name] != nil {
			t.Fatalf("quant loader built source-name Q8 tensor %s", name)
		}
	}
	assertQuantPrefillFinite(t, m, cfg)
}

func tinyQuantMoEBaseTensors(cfg Config) map[string]tinySTTensor {
	H := cfg.HiddenSize
	nH, nKV, hd := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	return map[string]tinySTTensor{
		"model.embed_tokens.weight": tinyF32Tensor([]int{cfg.VocabSize, H},
			sequenceFloats(cfg.VocabSize*H, 0.03)),
		"model.norm.weight": tinyF32Tensor([]int{H}, repeatFloat32(H, 1)),
		layerName(0, "input_layernorm.weight"): tinyF32Tensor([]int{H},
			repeatFloat32(H, 1)),
		layerName(0, "post_attention_layernorm.weight"): tinyF32Tensor([]int{H},
			repeatFloat32(H, 1)),
		layerName(0, "self_attn.q_proj.weight"): tinyF32Tensor([]int{nH * hd, H},
			sequenceFloats(nH*hd*H, 0.01)),
		layerName(0, "self_attn.k_proj.weight"): tinyF32Tensor([]int{nKV * hd, H},
			sequenceFloats(nKV*hd*H, 0.02)),
		layerName(0, "self_attn.v_proj.weight"): tinyF32Tensor([]int{nKV * hd, H},
			sequenceFloats(nKV*hd*H, 0.04)),
		layerName(0, "self_attn.o_proj.weight"): tinyF32Tensor([]int{H, nH * hd},
			sequenceFloats(H*nH*hd, 0.05)),
	}
}

func tinyF32Tensor(shape []int, vals []float32) tinySTTensor {
	return tinySTTensor{dtype: "F32", shape: shape, data: f32TestBytes(vals)}
}

func assertQ8Shape(t *testing.T, m *Model, name string, out, in int) {
	t.Helper()
	qt := m.q8w[name]
	if qt == nil {
		t.Fatalf("missing Q8 tensor %s", name)
	}
	if qt.out != out || qt.in != in {
		t.Fatalf("%s Q8 shape = [%d,%d], want [%d,%d]", name, qt.out, qt.in, out, in)
	}
	if m.has(name) {
		t.Fatalf("quant loader retained f32 copy of %s", name)
	}
}

func assertQuantPrefillFinite(t *testing.T, m *Model, cfg Config) {
	t.Helper()
	s := m.NewSession()
	s.Quant = true
	logits := s.Prefill([]int{3, 17, 5, 23})
	if len(logits) != cfg.VocabSize {
		t.Fatalf("logits len = %d, want %d", len(logits), cfg.VocabSize)
	}
	for i, v := range logits {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("Q8 quant-load logit[%d] not finite: %v", i, v)
		}
	}
}

func repeatFloat32(n int, v float32) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = v
	}
	return out
}

func repeatByte(n int, v byte) []byte {
	out := make([]byte, n)
	for i := range out {
		out[i] = v
	}
	return out
}
