package ggufload

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"testing"
)

type ggufOraclePrompt struct {
	Index     int   `json:"index"`
	Ids       []int `json:"ids"`
	GreedyIds []int `json:"greedy_ids"`
}

type ggufOracleDoc struct {
	Model   string             `json:"model"`
	Prompts []ggufOraclePrompt `json:"prompts"`
}

func TestReadParsesMetadataTensorDirectoryAndConfig(t *testing.T) {
	var b bytes.Buffer
	writeString := func(s string) {
		_ = binary.Write(&b, binary.LittleEndian, uint64(len(s)))
		b.WriteString(s)
	}
	writeKV := func(k string, typ ValueType, writeValue func()) {
		writeString(k)
		_ = binary.Write(&b, binary.LittleEndian, uint32(typ))
		writeValue()
	}
	writeU64 := func(v uint64) { _ = binary.Write(&b, binary.LittleEndian, v) }
	writeU32 := func(v uint32) { _ = binary.Write(&b, binary.LittleEndian, v) }
	writeF32 := func(v float32) { writeU32(math.Float32bits(v)) }
	writeStrVal := func(s string) func() { return func() { writeString(s) } }
	writeU64Val := func(v uint64) func() { return func() { writeU64(v) } }
	writeU32Val := func(v uint32) func() { return func() { writeU32(v) } }
	writeF32Val := func(v float32) func() { return func() { writeF32(v) } }

	b.WriteString(Magic)
	writeU32(Version)
	writeU64(3)  // tensors
	writeU64(14) // metadata KVs

	writeKV("general.architecture", TypeString, writeStrVal("qwen2"))
	writeKV("general.alignment", TypeUint32, writeU32Val(64))
	writeKV("qwen2.context_length", TypeUint64, writeU64Val(32768))
	writeKV("qwen2.embedding_length", TypeUint64, writeU64Val(32))
	writeKV("qwen2.block_count", TypeUint64, writeU64Val(2))
	writeKV("qwen2.feed_forward_length", TypeUint64, writeU64Val(64))
	writeKV("qwen2.rope.dimension_count", TypeUint64, writeU64Val(8))
	writeKV("qwen2.attention.head_count", TypeUint64, writeU64Val(4))
	writeKV("qwen2.attention.head_count_kv", TypeUint64, writeU64Val(2))
	writeKV("qwen2.attention.layer_norm_rms_epsilon", TypeFloat32, writeF32Val(1e-5))
	writeKV("qwen2.rope.freq_base", TypeFloat32, writeF32Val(1000000))
	writeKV("tokenizer.ggml.eos_token_id", TypeUint32, writeU32Val(2))
	writeKV("tokenizer.ggml.tokens", TypeArray, func() {
		writeU32(uint32(TypeString))
		writeU64(3)
		writeString("<unk>")
		writeString("hello")
		writeString("world")
	})
	writeKV("tokenizer.ggml.token_type", TypeArray, func() {
		writeU32(uint32(TypeInt32))
		writeU64(3)
		writeU32(uint32(1))
		writeU32(uint32(1))
		writeU32(uint32(1))
	})

	writeTensor := func(name string, dims []uint64, typ TensorType, off uint64) {
		writeString(name)
		writeU32(uint32(len(dims)))
		for _, d := range dims {
			writeU64(d)
		}
		writeU32(uint32(typ))
		writeU64(off)
	}
	writeTensor("token_embd.weight", []uint64{32, 3}, TensorF32, 0)
	writeTensor("output.weight", []uint64{32, 3}, TensorQ4_K, 64)
	writeTensor("blk.0.attn_q.bias", []uint64{32}, TensorF32, 128)

	got, err := Read(bytes.NewReader(b.Bytes()))
	if err != nil {
		t.Fatalf("Read: %v", err)
	}
	if got.Version != Version {
		t.Fatalf("version=%d", got.Version)
	}
	if got.Alignment != 64 {
		t.Fatalf("alignment=%d", got.Alignment)
	}
	if got.TensorDataOffset%64 != 0 {
		t.Fatalf("tensor data offset %d is not 64-byte aligned", got.TensorDataOffset)
	}
	if len(got.Tensors) != 3 {
		t.Fatalf("tensor count=%d", len(got.Tensors))
	}
	if got.Tensors[1].Name != "output.weight" || got.Tensors[1].Type != TensorQ4_K || got.Tensors[1].FileOffset != got.TensorDataOffset+64 {
		t.Fatalf("bad tensor directory entry: %#v", got.Tensors[1])
	}
	toks, ok := got.StringArray("tokenizer.ggml.tokens")
	if !ok || !reflect.DeepEqual(toks, []string{"<unk>", "hello", "world"}) {
		t.Fatalf("tokens=%#v ok=%v", toks, ok)
	}

	cfg, err := got.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	if cfg.HiddenSize != 32 || cfg.NumLayers != 2 || cfg.NumHeads != 4 || cfg.NumKVHeads != 2 {
		t.Fatalf("bad model dimensions: %#v", cfg)
	}
	if cfg.HeadDim != 8 || cfg.IntermediateSize != 64 || cfg.VocabSize != 3 {
		t.Fatalf("bad derived config: %#v", cfg)
	}
	if math.Float64bits(cfg.RMSNormEps) != math.Float64bits(float64(float32(1e-5))) || cfg.RopeTheta != 1000000 {
		t.Fatalf("bad floats: %#v", cfg)
	}
	if cfg.TieWordEmbeddings || !cfg.AttentionBias || cfg.ModelType != "qwen2" || cfg.EOSTokenID != 2 {
		t.Fatalf("bad flags: %#v", cfg)
	}
}

func TestReadRejectsBadAlignmentAndBool(t *testing.T) {
	t.Run("bad alignment", func(t *testing.T) {
		var b bytes.Buffer
		writeMinimalHeader(&b, 0, 1)
		writeKVUint32(&b, "general.alignment", 7)
		if _, err := Read(bytes.NewReader(b.Bytes())); err == nil {
			t.Fatal("Read accepted non-multiple-of-8 alignment")
		}
	})

	t.Run("bad bool", func(t *testing.T) {
		var b bytes.Buffer
		writeMinimalHeader(&b, 0, 1)
		writeStringForTest(&b, "test.flag")
		_ = binary.Write(&b, binary.LittleEndian, uint32(TypeBool))
		b.WriteByte(2)
		if _, err := Read(bytes.NewReader(b.Bytes())); err == nil {
			t.Fatal("Read accepted invalid bool byte")
		}
	})
}

func TestWeightSourceReadsAndDequantizesSimpleTensors(t *testing.T) {
	path := filepath.Join(t.TempDir(), "tiny.gguf")
	if err := os.WriteFile(path, tinyTensorGGUF(t), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	assertTensor := func(name string, want []float32) {
		t.Helper()
		got, info, err := ws.TensorF32(name)
		if err != nil {
			t.Fatalf("TensorF32(%s): %v", name, err)
		}
		if len(got) != len(want) {
			t.Fatalf("%s len=%d, want %d (info=%#v)", name, len(got), len(want), info)
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("%s[%d]=%v bits=%#x, want %v bits=%#x", name, i, got[i], math.Float32bits(got[i]), want[i], math.Float32bits(want[i]))
			}
		}
	}

	assertTensor("f32.weight", []float32{1.25, -2.5})
	assertTensor("f16.weight", []float32{1, -2, 0.5})
	assertTensor("bf16.weight", []float32{3.5, -4})
	q8Want := make([]float32, qk8_0)
	for i := range q8Want {
		q8Want[i] = float32(i - 16)
	}
	assertTensor("q8.weight", q8Want)
	_, q4Want := q4KFixtureCodes()
	assertTensor("q4k.weight", q4Want)
	_, q6Want := q6KFixtureBlock()
	assertTensor("q6k.weight", q6Want)
	_, q5_0Want := q5FixtureBlock(false)
	assertTensor("q5_0.weight", q5_0Want)
	_, q5_1Want := q5FixtureBlock(true)
	assertTensor("q5_1.weight", q5_1Want)
	_, q5KWant := q5KFixtureBlock()
	assertTensor("q5k.weight", q5KWant)
	_, q2KWant := q2KFixtureBlock()
	assertTensor("q2k.weight", q2KWant)
	_, q3KWant := q3KFixtureBlock()
	assertTensor("q3k.weight", q3KWant)
	if _, ok := ws.Tensor("missing.weight"); ok {
		t.Fatal("missing tensor reported present")
	}
}

func TestWeightSourceBuildsCanonicalModelTensors(t *testing.T) {
	path := filepath.Join(t.TempDir(), "canonical.gguf")
	if err := os.WriteFile(path, tinyCanonicalModelGGUF(t), 0o644); err != nil {
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
	if cfg.HiddenSize != 2 || cfg.VocabSize != 3 || cfg.ModelType != "qwen2" || cfg.EOSTokenID != 2 {
		t.Fatalf("bad config: %#v", cfg)
	}
	byName := map[string]modelTensorForTest{}
	for _, tt := range tensors {
		byName[tt.Name] = modelTensorForTest{shape: tt.Shape, data: tt.Data}
	}
	assertModelTensorForTest(t, byName, "model.embed_tokens.weight", []int{3, 2}, []float32{1, 2, 3, 4, 5, 6})
	assertModelTensorForTest(t, byName, "model.norm.weight", []int{2}, []float32{7, 8})
	assertModelTensorForTest(t, byName, "lm_head.weight", []int{3, 2}, []float32{9, 10, 11, 12, 13, 14})

	if _, err := ws.Model(); err != nil {
		t.Fatalf("Model: %v", err)
	}
	if _, err := LoadModel(path); err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
}

func TestWeightSourceUnpermutesRotaryQKWeights(t *testing.T) {
	path := filepath.Join(t.TempDir(), "qk.gguf")
	qHF := sequenceF32ForTest(0, 16)
	kHF := sequenceF32ForTest(100, 16)
	if err := os.WriteFile(path, tinyRotaryQKGGUF(t, qHF, kHF), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	_, tensors, err := ws.F32Tensors()
	if err != nil {
		t.Fatalf("F32Tensors: %v", err)
	}
	byName := map[string]modelTensorForTest{}
	for _, tt := range tensors {
		byName[tt.Name] = modelTensorForTest{shape: tt.Shape, data: tt.Data}
	}
	assertModelTensorForTest(t, byName, "model.layers.0.self_attn.q_proj.weight", []int{4, 4}, qHF)
	assertModelTensorForTest(t, byName, "model.layers.0.self_attn.k_proj.weight", []int{4, 4}, kHF)
}

// TestWeightSourceKeepsNEOXRotaryQKWeights guards the inverse of the llama case: a
// NEOX-rope GGUF (qwen2) stores q/k already in HF order, so the loader must NOT run the
// llama-family rotary unpermute over them. Regression test for the real-model bug where
// every Qwen GGUF decoded to gibberish because its q/k were being unpermuted.
func TestWeightSourceKeepsNEOXRotaryQKWeights(t *testing.T) {
	path := filepath.Join(t.TempDir(), "qk_qwen2.gguf")
	qHF := sequenceF32ForTest(0, 16)
	kHF := sequenceF32ForTest(100, 16)
	if err := os.WriteFile(path, tinyRotaryQKGGUFQwen2(t, qHF, kHF), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	_, tensors, err := ws.F32Tensors()
	if err != nil {
		t.Fatalf("F32Tensors: %v", err)
	}
	byName := map[string]modelTensorForTest{}
	for _, tt := range tensors {
		byName[tt.Name] = modelTensorForTest{shape: tt.Shape, data: tt.Data}
	}
	// q/k must come back byte-for-byte as written (no unpermute).
	assertModelTensorForTest(t, byName, "model.layers.0.self_attn.q_proj.weight", []int{4, 4}, qHF)
	assertModelTensorForTest(t, byName, "model.layers.0.self_attn.k_proj.weight", []int{4, 4}, kHF)
}

func TestQwen35GGUFConfigCanonicalizesHybridTensorsAndRunsForward(t *testing.T) {
	path := filepath.Join(t.TempDir(), "qwen35.gguf")
	qHF := sequenceF32ForTest(200, 64*32) // [2*heads*head_dim, hidden]
	kHF := sequenceF32ForTest(300, 32*32)
	qkvHF := scaledSequenceF32ForTest(500, 64*32)
	zHF := scaledSequenceF32ForTest(600, 32*32)
	aHF := scaledSequenceF32ForTest(700, 4*32)
	bHF := scaledSequenceF32ForTest(800, 4*32)
	convHF := scaledSequenceF32ForTest(900, 64*4)
	outHF := scaledSequenceF32ForTest(1000, 32*32)
	aLogHF := []float32{0, 0.25, 0.5, 0.75}
	dtHF := []float32{0.1, 0.2, 0.3, 0.4}
	if err := os.WriteFile(path, tinyQwen35HybridGGUF(t, qHF, kHF), 0o644); err != nil {
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
	if cfg.ModelType != "qwen35" || !cfg.IsQwen35Hybrid() || !cfg.AttnOutputGate || !cfg.QKNorm || !cfg.NormGain1p {
		t.Fatalf("qwen35 flags not derived: %#v", cfg)
	}
	if cfg.HeadDim != 32 || cfg.PartialRotaryFactor != 0.5 || cfg.RopeTheta != 10000000 {
		t.Fatalf("qwen35 rope/head dims wrong: head=%d partial=%v theta=%v", cfg.HeadDim, cfg.PartialRotaryFactor, cfg.RopeTheta)
	}
	if cfg.LinearConvKernelDim != 4 || cfg.LinearKeyHeadDim != 8 || cfg.LinearValueHeadDim != 8 ||
		cfg.LinearNumKeyHeads != 2 || cfg.LinearNumValueHeads != 4 {
		t.Fatalf("qwen35 linear dims wrong: %#v", cfg)
	}
	if len(cfg.LayerTypes) != 4 || cfg.LayerTypes[0] != "linear_attention" || cfg.LayerTypes[3] != "full_attention" {
		t.Fatalf("layer types = %v", cfg.LayerTypes)
	}

	byName := map[string]modelTensorForTest{}
	for _, tt := range tensors {
		byName[tt.Name] = modelTensorForTest{shape: tt.Shape, data: tt.Data}
	}
	assertModelTensorForTest(t, byName, "model.norm.weight", []int{32}, make([]float32, 32))
	assertModelTensorForTest(t, byName, "model.layers.0.input_layernorm.weight", []int{32}, make([]float32, 32))
	assertModelTensorForTest(t, byName, "model.layers.0.post_attention_layernorm.weight", []int{32}, make([]float32, 32))
	assertModelTensorForTest(t, byName, "model.layers.0.self_attn.qkv_proj.weight", []int{64, 32}, qkvHF)
	assertModelTensorForTest(t, byName, "model.layers.0.self_attn.q_gate_proj.weight", []int{32, 32}, zHF)
	assertModelTensorForTest(t, byName, "model.layers.0.linear_attn.in_proj_a.weight", []int{4, 32}, aHF)
	assertModelTensorForTest(t, byName, "model.layers.0.linear_attn.in_proj_b.weight", []int{4, 32}, bHF)
	assertModelTensorForTest(t, byName, "model.layers.0.linear_attn.conv1d.weight", []int{64, 4}, convHF)
	assertModelTensorForTest(t, byName, "model.layers.0.linear_attn.out_proj.weight", []int{32, 32}, outHF)
	assertModelTensorForTest(t, byName, "model.layers.0.linear_attn.A_log", []int{4}, aLogHF)
	assertModelTensorForTest(t, byName, "model.layers.0.linear_attn.dt_bias", []int{4}, dtHF)
	assertModelTensorForTest(t, byName, "model.layers.0.linear_attn.norm.weight", []int{8}, onesF32ForTest(8))
	assertModelTensorForTest(t, byName, "model.layers.3.self_attn.q_proj.weight", []int{64, 32}, qHF)
	assertModelTensorForTest(t, byName, "model.layers.3.self_attn.k_proj.weight", []int{32, 32}, kHF)
	assertModelTensorForTest(t, byName, "model.layers.3.input_layernorm.weight", []int{32}, make([]float32, 32))
	assertModelTensorForTest(t, byName, "model.layers.3.post_attention_layernorm.weight", []int{32}, make([]float32, 32))
	assertModelTensorForTest(t, byName, "model.layers.3.self_attn.q_norm.weight", []int{32}, make([]float32, 32))
	assertModelTensorForTest(t, byName, "model.layers.3.self_attn.k_norm.weight", []int{32}, make([]float32, 32))

	m, err := LoadModel(path)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	act := m.Forward([]int{0, 1})
	if act.Seq != 2 || len(act.Logits) != 2 || len(act.Logits[1]) != 5 {
		t.Fatalf("bad forward shape: seq=%d logits=%dx?", act.Seq, len(act.Logits))
	}
	for i, v := range act.Logits[1] {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("logit[%d] is not finite: %v", i, v)
		}
	}

	qm, err := LoadModelQuant(path)
	if err != nil {
		t.Fatalf("LoadModelQuant: %v", err)
	}
	qact := qm.Forward([]int{0, 1})
	if qact.Seq != 2 || len(qact.Logits) != 2 || len(qact.Logits[1]) != 5 {
		t.Fatalf("bad quant forward shape: seq=%d logits=%dx?", qact.Seq, len(qact.Logits))
	}
	for i, v := range qact.Logits[1] {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("quant logit[%d] is not finite: %v", i, v)
		}
	}

	qs := qm.NewSession()
	qs.Quant = true
	qlogits := qs.Prefill([]int{0, 1})
	if len(qlogits) != 5 {
		t.Fatalf("bad quant session logits len=%d", len(qlogits))
	}
	qstep := qs.Step(2)
	if len(qstep) != 5 {
		t.Fatalf("bad quant session step logits len=%d", len(qstep))
	}
	for i, v := range qstep {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("quant session step logit[%d] is not finite: %v", i, v)
		}
	}
}

func TestLoadModelQuantProfileReportsLoadPhases(t *testing.T) {
	path := filepath.Join(t.TempDir(), "qwen35.gguf")
	if err := os.WriteFile(path, tinyQwen35HybridGGUF(t, sequenceF32ForTest(200, 64*32), sequenceF32ForTest(300, 32*32)), 0o644); err != nil {
		t.Fatalf("write fixture: %v", err)
	}
	profiler := NewLoadProfiler()
	m, err := LoadModelQuantProfile(path, profiler)
	if err != nil {
		t.Fatalf("LoadModelQuantProfile: %v", err)
	}
	if m.Cfg.ModelType != "qwen35" || !m.Cfg.IsQwen35Hybrid() {
		t.Fatalf("bad model from profiled load: %#v", m.Cfg)
	}
	p := profiler.Snapshot("gguf-lean-q8", path, 1)
	if p.TensorCount == 0 {
		t.Fatalf("profile tensor count = 0: %#v", p)
	}
	for _, phase := range []string{
		"gguf_open_index",
		"gguf_config",
		"gguf_read",
		"gguf_dequant",
		"gguf_normalize",
		"quant_builder_add",
		"quant_builder_finalize",
	} {
		if loadPhaseCalls(p, phase) == 0 {
			t.Fatalf("phase %q not recorded: %#v", phase, p.Phases)
		}
	}
	if len(p.TopTensors) == 0 || p.TopTensors[0].Name == "" || p.TopTensors[0].CanonicalName == "" {
		t.Fatalf("top tensor details missing: %#v", p.TopTensors)
	}
}

func TestOptionalSmolLM2Q4KMGGUFDequantizesAllTensors(t *testing.T) {
	if os.Getenv("FAK_GGUF_REAL_SMOKE") != "1" {
		t.Skip("set FAK_GGUF_REAL_SMOKE=1 to run the local ignored GGUF smoke test")
	}
	path := filepath.Join("..", "..", "experiments", "model-baseline", "gguf", "SmolLM2-135M-Instruct-Q4_K_M.gguf")
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			t.Skipf("local GGUF smoke fixture not found: %s", path)
		}
		t.Fatalf("stat local GGUF smoke fixture: %v", err)
	}
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	defer ws.Close()

	cfg, err := ws.File.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	if cfg.HiddenSize == 0 || cfg.NumLayers == 0 || cfg.VocabSize == 0 {
		t.Fatalf("incomplete config from real GGUF: %#v", cfg)
	}

	counts := make(map[TensorType]int)
	for _, info := range ws.File.Tensors {
		got, _, err := ws.TensorF32(info.Name)
		if err != nil {
			t.Fatalf("TensorF32(%s type %d): %v", info.Name, info.Type, err)
		}
		if len(got) == 0 {
			t.Fatalf("TensorF32(%s) returned no values", info.Name)
		}
		counts[info.Type]++
	}
	if counts[TensorQ4_K] == 0 || counts[TensorQ6_K] == 0 {
		t.Fatalf("real Q4_K_M GGUF did not exercise Q4_K and Q6_K tensors: %#v", counts)
	}
	t.Logf("dequantized %d tensors from real Q4_K_M GGUF by type: %#v", len(ws.File.Tensors), counts)
}

func TestOptionalQwen35GGUFMapsEveryTensorName(t *testing.T) {
	if os.Getenv("FAK_GGUF_REAL_SMOKE") != "1" {
		t.Skip("set FAK_GGUF_REAL_SMOKE=1 to run the local ignored GGUF smoke test")
	}
	home, err := os.UserHomeDir()
	if err != nil {
		t.Fatalf("home dir: %v", err)
	}
	dir := filepath.Join(home, ".cache", "fak-models", "gguf")
	var path string
	for _, name := range []string{"Qwen3.6-27B-Q4_K_M.gguf", "Qwen3.6-27B.q4_k_m.gguf"} {
		candidate := filepath.Join(dir, name)
		if _, err := os.Stat(candidate); err == nil {
			path = candidate
			break
		} else if !os.IsNotExist(err) {
			t.Fatalf("stat local Qwen35 GGUF smoke fixture: %v", err)
		}
	}
	if path == "" {
		t.Skipf("local Qwen35 GGUF smoke fixture not found under %s", dir)
	}
	f, err := Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	cfg, err := f.Config()
	if err != nil {
		t.Fatalf("Config: %v", err)
	}
	if cfg.ModelType != "qwen35" || cfg.NumLayers != 64 || cfg.HeadDim != 256 || cfg.PartialRotaryFactor != 0.25 ||
		cfg.LinearNumKeyHeads != 16 || cfg.LinearNumValueHeads != 48 || cfg.LinearKeyHeadDim != 128 || cfg.LinearConvKernelDim != 4 {
		t.Fatalf("bad Qwen35 config from real GGUF: %#v", cfg)
	}
	missing := []string{}
	for _, info := range f.Tensors {
		if _, ok := CanonicalTensorName(info.Name); !ok {
			missing = append(missing, info.Name)
		}
	}
	if len(missing) > 0 {
		t.Fatalf("missing canonical mappings for real Qwen35 GGUF tensors: %v", missing)
	}
	t.Logf("mapped %d real Qwen35 GGUF tensors without dequantizing payloads", len(f.Tensors))
}

func TestOptionalSmolLM2F16GGUFGreedyMatchesHFOracle(t *testing.T) {
	if os.Getenv("FAK_GGUF_REAL_SMOKE") != "1" {
		t.Skip("set FAK_GGUF_REAL_SMOKE=1 to run the local ignored GGUF smoke test")
	}
	path := filepath.Join("..", "..", "experiments", "model-baseline", "gguf", "SmolLM2-135M-Instruct-f16.gguf")
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			t.Skipf("local GGUF smoke fixture not found: %s", path)
		}
		t.Fatalf("stat local GGUF smoke fixture: %v", err)
	}
	oraclePath := filepath.Join("..", "model", ".cache", "smollm2-135m", "oracle.json")
	rawOracle, err := os.ReadFile(oraclePath)
	if err != nil {
		if os.IsNotExist(err) {
			t.Skipf("local HF oracle not found: %s", oraclePath)
		}
		t.Fatalf("read oracle: %v", err)
	}
	var doc ggufOracleDoc
	if err := json.Unmarshal(rawOracle, &doc); err != nil {
		t.Fatalf("parse oracle: %v", err)
	}
	m, err := LoadModel(path)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	for _, p := range doc.Prompts {
		got := m.NewSession().Generate(p.Ids, len(p.GreedyIds))
		t.Logf("prompt %d gguf greedy=%v", p.Index, got)
		t.Logf("prompt %d hf   greedy=%v", p.Index, p.GreedyIds)
		if len(got) != len(p.GreedyIds) {
			t.Fatalf("prompt %d greedy len=%d, want %d", p.Index, len(got), len(p.GreedyIds))
		}
		for i := range p.GreedyIds {
			if got[i] != p.GreedyIds[i] {
				t.Fatalf("prompt %d greedy token %d: gguf=%d hf=%d", p.Index, i, got[i], p.GreedyIds[i])
			}
		}
	}
}

func loadPhaseCalls(p *LoadProfile, phase string) int {
	for _, st := range p.Phases {
		if st.Phase == phase {
			return st.Calls
		}
	}
	return 0
}

func writeMinimalHeader(b *bytes.Buffer, tensors, kvs uint64) {
	b.WriteString(Magic)
	_ = binary.Write(b, binary.LittleEndian, uint32(Version))
	_ = binary.Write(b, binary.LittleEndian, tensors)
	_ = binary.Write(b, binary.LittleEndian, kvs)
}

func writeKVUint32(b *bytes.Buffer, key string, value uint32) {
	writeStringForTest(b, key)
	_ = binary.Write(b, binary.LittleEndian, uint32(TypeUint32))
	_ = binary.Write(b, binary.LittleEndian, value)
}

func writeKVUint64(b *bytes.Buffer, key string, value uint64) {
	writeStringForTest(b, key)
	_ = binary.Write(b, binary.LittleEndian, uint32(TypeUint64))
	_ = binary.Write(b, binary.LittleEndian, value)
}

func writeKVFloat32(b *bytes.Buffer, key string, value float32) {
	writeStringForTest(b, key)
	_ = binary.Write(b, binary.LittleEndian, uint32(TypeFloat32))
	_ = binary.Write(b, binary.LittleEndian, math.Float32bits(value))
}

func writeKVString(b *bytes.Buffer, key, value string) {
	writeStringForTest(b, key)
	_ = binary.Write(b, binary.LittleEndian, uint32(TypeString))
	writeStringForTest(b, value)
}

func writeKVStringArray(b *bytes.Buffer, key string, values []string) {
	writeStringForTest(b, key)
	_ = binary.Write(b, binary.LittleEndian, uint32(TypeArray))
	_ = binary.Write(b, binary.LittleEndian, uint32(TypeString))
	_ = binary.Write(b, binary.LittleEndian, uint64(len(values)))
	for _, value := range values {
		writeStringForTest(b, value)
	}
}

func writeStringForTest(b *bytes.Buffer, s string) {
	_ = binary.Write(b, binary.LittleEndian, uint64(len(s)))
	b.WriteString(s)
}

type modelTensorForTest struct {
	shape []int
	data  []float32
}

func assertModelTensorForTest(t *testing.T, byName map[string]modelTensorForTest, name string, shape []int, data []float32) {
	t.Helper()
	got, ok := byName[name]
	if !ok {
		t.Fatalf("missing tensor %s", name)
	}
	if !reflect.DeepEqual(got.shape, shape) {
		t.Fatalf("%s shape=%v, want %v", name, got.shape, shape)
	}
	if len(got.data) != len(data) {
		t.Fatalf("%s len=%d, want %d", name, len(got.data), len(data))
	}
	for i := range data {
		if math.Float32bits(got.data[i]) != math.Float32bits(data[i]) {
			t.Fatalf("%s[%d]=%v, want %v", name, i, got.data[i], data[i])
		}
	}
}

func assertModelTensorShapeForTest(t *testing.T, byName map[string]modelTensorForTest, name string, shape []int) {
	t.Helper()
	got, ok := byName[name]
	if !ok {
		t.Fatalf("missing tensor %s", name)
	}
	if !reflect.DeepEqual(got.shape, shape) {
		t.Fatalf("%s shape=%v, want %v", name, got.shape, shape)
	}
}

func tinyCanonicalModelGGUF(t *testing.T) []byte {
	t.Helper()
	var b bytes.Buffer
	writeMinimalHeader(&b, 3, 13)
	writeKVString(&b, "general.architecture", "qwen2")
	writeKVUint32(&b, "general.alignment", 32)
	writeKVUint64(&b, "qwen2.context_length", 16)
	writeKVUint64(&b, "qwen2.embedding_length", 2)
	writeKVUint64(&b, "qwen2.block_count", 1)
	writeKVUint64(&b, "qwen2.feed_forward_length", 4)
	writeKVUint64(&b, "qwen2.rope.dimension_count", 1)
	writeKVUint64(&b, "qwen2.attention.head_count", 2)
	writeKVUint64(&b, "qwen2.attention.head_count_kv", 1)
	writeKVFloat32(&b, "qwen2.attention.layer_norm_rms_epsilon", 1e-5)
	writeKVFloat32(&b, "qwen2.rope.freq_base", 10000)
	writeKVUint32(&b, "tokenizer.ggml.eos_token_id", 2)
	writeKVStringArray(&b, "tokenizer.ggml.tokens", []string{"a", "b", "c"})
	writeTensorInfoForTest(&b, "token_embd.weight", []uint64{2, 3}, TensorF32, 0)
	writeTensorInfoForTest(&b, "output_norm.weight", []uint64{2}, TensorF32, 32)
	writeTensorInfoForTest(&b, "output.weight", []uint64{2, 3}, TensorF32, 64)
	padToAlignment(&b, 32)
	dataStart := b.Len()
	for _, v := range []float32{1, 2, 3, 4, 5, 6} {
		writeF32ForTest(&b, v)
	}
	padToLen(&b, dataStart+32)
	for _, v := range []float32{7, 8} {
		writeF32ForTest(&b, v)
	}
	padToLen(&b, dataStart+64)
	for _, v := range []float32{9, 10, 11, 12, 13, 14} {
		writeF32ForTest(&b, v)
	}
	return b.Bytes()
}

func tinyRotaryQKGGUF(t *testing.T, qHF, kHF []float32) []byte {
	t.Helper()
	var b bytes.Buffer
	writeMinimalHeader(&b, 2, 12)
	writeKVString(&b, "general.architecture", "llama")
	writeKVUint32(&b, "general.alignment", 32)
	writeKVUint64(&b, "llama.context_length", 16)
	writeKVUint64(&b, "llama.embedding_length", 4)
	writeKVUint64(&b, "llama.block_count", 1)
	writeKVUint64(&b, "llama.feed_forward_length", 8)
	writeKVUint64(&b, "llama.rope.dimension_count", 4)
	writeKVUint64(&b, "llama.attention.head_count", 1)
	writeKVUint64(&b, "llama.attention.head_count_kv", 1)
	writeKVFloat32(&b, "llama.attention.layer_norm_rms_epsilon", 1e-5)
	writeKVFloat32(&b, "llama.rope.freq_base", 10000)
	writeKVStringArray(&b, "tokenizer.ggml.tokens", []string{"a", "b"})
	writeTensorInfoForTest(&b, "blk.0.attn_q.weight", []uint64{4, 4}, TensorF32, 0)
	writeTensorInfoForTest(&b, "blk.0.attn_k.weight", []uint64{4, 4}, TensorF32, 64)
	padToAlignment(&b, 32)
	dataStart := b.Len()
	for _, v := range permuteRotaryForTest(qHF, 1, 4, 4) {
		writeF32ForTest(&b, v)
	}
	padToLen(&b, dataStart+64)
	for _, v := range permuteRotaryForTest(kHF, 1, 4, 4) {
		writeF32ForTest(&b, v)
	}
	return b.Bytes()
}

// tinyRotaryQKGGUFQwen2 mirrors tinyRotaryQKGGUF but for a NEOX-rope architecture
// ("qwen2"). convert_hf_to_gguf.py does NOT permute q/k for these models, so the q/k
// tensors are written in plain HF order and the loader must return them UNCHANGED.
func tinyRotaryQKGGUFQwen2(t *testing.T, qHF, kHF []float32) []byte {
	t.Helper()
	var b bytes.Buffer
	writeMinimalHeader(&b, 2, 12)
	writeKVString(&b, "general.architecture", "qwen2")
	writeKVUint32(&b, "general.alignment", 32)
	writeKVUint64(&b, "qwen2.context_length", 16)
	writeKVUint64(&b, "qwen2.embedding_length", 4)
	writeKVUint64(&b, "qwen2.block_count", 1)
	writeKVUint64(&b, "qwen2.feed_forward_length", 8)
	writeKVUint64(&b, "qwen2.rope.dimension_count", 4)
	writeKVUint64(&b, "qwen2.attention.head_count", 1)
	writeKVUint64(&b, "qwen2.attention.head_count_kv", 1)
	writeKVFloat32(&b, "qwen2.attention.layer_norm_rms_epsilon", 1e-5)
	writeKVFloat32(&b, "qwen2.rope.freq_base", 10000)
	writeKVStringArray(&b, "tokenizer.ggml.tokens", []string{"a", "b"})
	writeTensorInfoForTest(&b, "blk.0.attn_q.weight", []uint64{4, 4}, TensorF32, 0)
	writeTensorInfoForTest(&b, "blk.0.attn_k.weight", []uint64{4, 4}, TensorF32, 64)
	padToAlignment(&b, 32)
	dataStart := b.Len()
	for _, v := range qHF { // stored in HF order, NOT permuted
		writeF32ForTest(&b, v)
	}
	padToLen(&b, dataStart+64)
	for _, v := range kHF {
		writeF32ForTest(&b, v)
	}
	return b.Bytes()
}

func sequenceF32ForTest(start, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(start + i)
	}
	return out
}

func scaledSequenceF32ForTest(start, n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32(start+i) * 0.001
	}
	return out
}

func onesF32ForTest(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = 1
	}
	return out
}

func permuteRotaryForTest(src []float32, heads, headDim, in int) []float32 {
	dst := make([]float32, len(src))
	half := headDim / 2
	for h := 0; h < heads; h++ {
		for j := 0; j < half; j++ {
			for p := 0; p < 2; p++ {
				for c := 0; c < in; c++ {
					dst[((h*half+j)*2+p)*in+c] = src[((h*2+p)*half+j)*in+c]
				}
			}
		}
	}
	return dst
}

func permuteQwen35GatedQForTest(src []float32, heads, headDim, in int) []float32 {
	dst := make([]float32, len(src))
	half := headDim / 2
	for h := 0; h < heads; h++ {
		head := h * 2 * headDim
		for j := 0; j < half; j++ {
			for p := 0; p < 2; p++ {
				for c := 0; c < in; c++ {
					dst[(head+j*2+p)*in+c] = src[(head+p*half+j)*in+c]
				}
			}
		}
		copy(dst[(head+headDim)*in:(head+2*headDim)*in], src[(head+headDim)*in:(head+2*headDim)*in])
	}
	return dst
}

func interleaveQwen35QKVRowsForTest(src []float32, keyDim, nK, nV, headDim, rowWidth int) []float32 {
	dst := append([]float32(nil), src...)
	vOff := 2 * keyDim * rowWidth
	copy(dst[vOff:], interleaveQwen35ValueRowsForTest(src[vOff:], nK, nV, headDim, rowWidth))
	return dst
}

func interleaveQwen35ValueRowsForTest(src []float32, nK, nV, headSpan, rowWidth int) []float32 {
	ratio := nV / nK
	dst := make([]float32, len(src))
	rowBlock := headSpan * rowWidth
	for k := 0; k < nK; k++ {
		for r := 0; r < ratio; r++ {
			srcHead := k*ratio + r
			dstHead := r*nK + k
			copy(dst[dstHead*rowBlock:(dstHead+1)*rowBlock], src[srcHead*rowBlock:(srcHead+1)*rowBlock])
		}
	}
	return dst
}

func interleaveQwen35ValueColsForTest(src []float32, rows, nK, nV, headDim int) []float32 {
	ratio := nV / nK
	cols := nV * headDim
	dst := make([]float32, len(src))
	for row := 0; row < rows; row++ {
		rowOff := row * cols
		for k := 0; k < nK; k++ {
			for r := 0; r < ratio; r++ {
				srcHead := k*ratio + r
				dstHead := r*nK + k
				copy(dst[rowOff+dstHead*headDim:rowOff+(dstHead+1)*headDim], src[rowOff+srcHead*headDim:rowOff+(srcHead+1)*headDim])
			}
		}
	}
	return dst
}

type tinyGGUFTensor struct {
	name string
	dims []uint64
	data []float32
}

func tinyQwen35HybridGGUF(t *testing.T, qHF, kHF []float32) []byte {
	t.Helper()
	const (
		H             = 32
		I             = 64
		V             = 5
		heads         = 1
		kvHeads       = 1
		headDim       = 32
		linHeadDim    = 8
		linKeyHeads   = 2
		linValueHeads = 4
	)
	const (
		linKeyDim = linKeyHeads * linHeadDim
		linValDim = linValueHeads * linHeadDim
		convDim   = 2*linKeyDim + linValDim
	)
	var tensors []tinyGGUFTensor
	add := func(name string, dims []uint64, data []float32) {
		n := 1
		for _, d := range dims {
			n *= int(d)
		}
		if data == nil {
			data = tinyQwen35Data(name, n)
		}
		if len(data) != n {
			t.Fatalf("%s has %d values, want %d", name, len(data), n)
		}
		tensors = append(tensors, tinyGGUFTensor{name: name, dims: dims, data: data})
	}
	add("token_embd.weight", []uint64{H, V}, nil)
	add("output_norm.weight", []uint64{H}, onesF32ForTest(H))
	add("output.weight", []uint64{H, V}, nil)
	qkvHF := scaledSequenceF32ForTest(500, convDim*H)
	zHF := scaledSequenceF32ForTest(600, linValDim*H)
	aHF := scaledSequenceF32ForTest(700, linValueHeads*H)
	bHF := scaledSequenceF32ForTest(800, linValueHeads*H)
	convHF := scaledSequenceF32ForTest(900, convDim*4)
	outHF := scaledSequenceF32ForTest(1000, H*linValDim)
	aLogHF := []float32{0, 0.25, 0.5, 0.75}
	dtHF := []float32{0.1, 0.2, 0.3, 0.4}
	for l := 0; l < 3; l++ {
		p := "blk." + strconv.Itoa(l) + "."
		add(p+"attn_gate.weight", []uint64{H, linValDim}, interleaveQwen35ValueRowsForTest(zHF, linKeyHeads, linValueHeads, linHeadDim, H))
		add(p+"attn_norm.weight", []uint64{H}, onesF32ForTest(H))
		add(p+"attn_qkv.weight", []uint64{H, convDim}, interleaveQwen35QKVRowsForTest(qkvHF, linKeyDim, linKeyHeads, linValueHeads, linHeadDim, H))
		add(p+"ffn_down.weight", []uint64{I, H}, nil)
		add(p+"ffn_gate.weight", []uint64{H, I}, nil)
		add(p+"ffn_up.weight", []uint64{H, I}, nil)
		add(p+"post_attention_norm.weight", []uint64{H}, onesF32ForTest(H))
		add(p+"ssm_a", []uint64{linValueHeads}, interleaveQwen35ValueRowsForTest(aLogHF, linKeyHeads, linValueHeads, 1, 1))
		add(p+"ssm_alpha.weight", []uint64{H, linValueHeads}, interleaveQwen35ValueRowsForTest(aHF, linKeyHeads, linValueHeads, 1, H))
		add(p+"ssm_beta.weight", []uint64{H, linValueHeads}, interleaveQwen35ValueRowsForTest(bHF, linKeyHeads, linValueHeads, 1, H))
		add(p+"ssm_conv1d.weight", []uint64{4, convDim}, interleaveQwen35QKVRowsForTest(convHF, linKeyDim, linKeyHeads, linValueHeads, linHeadDim, 4))
		add(p+"ssm_dt.bias", []uint64{linValueHeads}, interleaveQwen35ValueRowsForTest(dtHF, linKeyHeads, linValueHeads, 1, 1))
		add(p+"ssm_norm.weight", []uint64{linHeadDim}, onesF32ForTest(linHeadDim))
		add(p+"ssm_out.weight", []uint64{linValDim, H}, interleaveQwen35ValueColsForTest(outHF, H, linKeyHeads, linValueHeads, linHeadDim))
	}
	{
		p := "blk.3."
		add(p+"attn_k.weight", []uint64{H, kvHeads * headDim}, permuteRotaryForTest(kHF, kvHeads, headDim, H))
		add(p+"attn_k_norm.weight", []uint64{headDim}, onesF32ForTest(headDim))
		add(p+"attn_norm.weight", []uint64{H}, onesF32ForTest(H))
		add(p+"attn_output.weight", []uint64{heads * headDim, H}, nil)
		add(p+"attn_q.weight", []uint64{H, 2 * heads * headDim}, permuteQwen35GatedQForTest(qHF, heads, headDim, H))
		add(p+"attn_q_norm.weight", []uint64{headDim}, onesF32ForTest(headDim))
		add(p+"attn_v.weight", []uint64{H, kvHeads * headDim}, nil)
		add(p+"ffn_down.weight", []uint64{I, H}, nil)
		add(p+"ffn_gate.weight", []uint64{H, I}, nil)
		add(p+"ffn_up.weight", []uint64{H, I}, nil)
		add(p+"post_attention_norm.weight", []uint64{H}, onesF32ForTest(H))
	}

	offsets := make([]uint64, len(tensors))
	var off uint64
	for i, tt := range tensors {
		offsets[i] = off
		off = alignOffset(off+uint64(len(tt.data))*4, 32)
	}

	var b bytes.Buffer
	writeMinimalHeader(&b, uint64(len(tensors)), 21)
	writeKVString(&b, "general.architecture", "qwen35")
	writeKVUint32(&b, "general.alignment", 32)
	writeKVUint64(&b, "qwen35.context_length", 32)
	writeKVUint64(&b, "qwen35.embedding_length", H)
	writeKVUint64(&b, "qwen35.block_count", 4)
	writeKVUint64(&b, "qwen35.feed_forward_length", I)
	writeKVUint64(&b, "qwen35.attention.head_count", heads)
	writeKVUint64(&b, "qwen35.attention.head_count_kv", kvHeads)
	writeKVUint64(&b, "qwen35.attention.key_length", headDim)
	writeKVUint64(&b, "qwen35.attention.value_length", headDim)
	writeKVFloat32(&b, "qwen35.attention.layer_norm_rms_epsilon", 1e-5)
	writeKVFloat32(&b, "qwen35.rope.freq_base", 10000000)
	writeKVUint64(&b, "qwen35.rope.dimension_count", 16)
	writeKVUint64(&b, "qwen35.full_attention_interval", 4)
	writeKVUint64(&b, "qwen35.ssm.conv_kernel", 4)
	writeKVUint64(&b, "qwen35.ssm.state_size", linHeadDim)
	writeKVUint64(&b, "qwen35.ssm.group_count", linKeyHeads)
	writeKVUint64(&b, "qwen35.ssm.inner_size", linValDim)
	writeKVUint64(&b, "qwen35.ssm.time_step_rank", linValueHeads)
	writeKVUint32(&b, "tokenizer.ggml.eos_token_id", 2)
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

func tinyQwen35Data(name string, n int) []float32 {
	out := make([]float32, n)
	var seed uint64 = 1469598103934665603
	for _, c := range []byte(name) {
		seed ^= uint64(c)
		seed *= 1099511628211
	}
	for i := range out {
		seed = seed*6364136223846793005 + 1442695040888963407
		u := float32(seed>>40) / float32(1<<24)
		out[i] = (u*2 - 1) * 0.05
	}
	return out
}

func tinyTensorGGUF(t *testing.T) []byte {
	t.Helper()
	var b bytes.Buffer
	writeMinimalHeader(&b, 11, 1)
	writeKVUint32(&b, "general.alignment", 32)
	writeTensorInfoForTest(&b, "f32.weight", []uint64{2}, TensorF32, 0)
	writeTensorInfoForTest(&b, "f16.weight", []uint64{3}, TensorF16, 32)
	writeTensorInfoForTest(&b, "bf16.weight", []uint64{2}, TensorBF16, 64)
	writeTensorInfoForTest(&b, "q8.weight", []uint64{32}, TensorQ8_0, 96)
	writeTensorInfoForTest(&b, "q4k.weight", []uint64{qkK}, TensorQ4_K, 160)
	writeTensorInfoForTest(&b, "q6k.weight", []uint64{qkK}, TensorQ6_K, 320)
	writeTensorInfoForTest(&b, "q5_0.weight", []uint64{qk5}, TensorQ5_0, 544)
	writeTensorInfoForTest(&b, "q5_1.weight", []uint64{qk5}, TensorQ5_1, 576)
	writeTensorInfoForTest(&b, "q5k.weight", []uint64{qkK}, TensorQ5_K, 608)
	writeTensorInfoForTest(&b, "q2k.weight", []uint64{qkK}, TensorQ2_K, 800)
	writeTensorInfoForTest(&b, "q3k.weight", []uint64{qkK}, TensorQ3_K, 896)
	padToAlignment(&b, 32)
	dataStart := b.Len()
	writeF32ForTest(&b, 1.25)
	writeF32ForTest(&b, -2.5)
	padToLen(&b, dataStart+32)
	writeU16ForTest(&b, 0x3c00) // 1.0
	writeU16ForTest(&b, 0xc000) // -2.0
	writeU16ForTest(&b, 0x3800) // 0.5
	padToLen(&b, dataStart+64)
	writeU16ForTest(&b, uint16(math.Float32bits(3.5)>>16))
	writeU16ForTest(&b, uint16(math.Float32bits(-4)>>16))
	padToLen(&b, dataStart+96)
	writeU16ForTest(&b, 0x3c00) // scale = 1.0
	for i := 0; i < qk8_0; i++ {
		b.WriteByte(byte(int8(i - 16)))
	}
	padToLen(&b, dataStart+160)
	q4Codes, _ := q4KFixtureCodes()
	writeU16ForTest(&b, 0x3c00) // d = 1.0
	writeU16ForTest(&b, 0x0000) // dmin = 0.0
	b.Write([]byte{1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1})
	b.Write(q4Codes)
	padToLen(&b, dataStart+320)
	q6Block, _ := q6KFixtureBlock()
	b.Write(q6Block)
	padToLen(&b, dataStart+544)
	q5Block, _ := q5FixtureBlock(false)
	b.Write(q5Block)
	padToLen(&b, dataStart+576)
	q5Block, _ = q5FixtureBlock(true)
	b.Write(q5Block)
	padToLen(&b, dataStart+608)
	q5KBlock, _ := q5KFixtureBlock()
	b.Write(q5KBlock)
	padToLen(&b, dataStart+800)
	q2KBlock, _ := q2KFixtureBlock()
	b.Write(q2KBlock)
	padToLen(&b, dataStart+896)
	q3KBlock, _ := q3KFixtureBlock()
	b.Write(q3KBlock)
	return b.Bytes()
}

func q4KFixtureCodes() ([]byte, []float32) {
	codes := make([]byte, 0, qkK/2)
	want := make([]float32, qkK)
	for group := 0; group < qkK; group += 64 {
		for l := 0; l < 32; l++ {
			low := byte(l % 16)
			high := byte(15 - l%16)
			codes = append(codes, low|(high<<4))
			want[group+l] = float32(low)
			want[group+32+l] = float32(high)
		}
	}
	return codes, want
}

func q2KFixtureBlock() ([]byte, []float32) {
	scales := make([]byte, qkK/16)
	for i := range scales {
		scales[i] = 1
	}
	qs := make([]byte, qkK/4)
	want := make([]float32, qkK)
	for group := 0; group < qkK/128; group++ {
		qbase := group * 32
		obase := group * 128
		for j := 0; j < 4; j++ {
			shift := uint(2 * j)
			for l := 0; l < 16; l++ {
				low := byte((group + j + l) % 4)
				high := byte((group + j + l + 1) % 4)
				qs[qbase+l] |= low << shift
				qs[qbase+16+l] |= high << shift
				want[obase+j*32+l] = float32(low)
				want[obase+j*32+16+l] = float32(high)
			}
		}
	}

	var b bytes.Buffer
	b.Write(scales)
	b.Write(qs)
	writeU16ForTest(&b, 0x3c00) // d = 1.0
	writeU16ForTest(&b, 0x0000) // dmin = 0.0
	return b.Bytes(), want
}

func q3KFixtureBlock() ([]byte, []float32) {
	hmask := make([]byte, qkK/8)
	qs := make([]byte, qkK/4)
	scaleVals := make([]byte, qkK/16)
	for i := range scaleVals {
		scaleVals[i] = 33 + byte(i%3)
	}
	want := make([]float32, qkK)
	for group := 0; group < qkK/128; group++ {
		qbase := group * 32
		obase := group * 128
		mask := byte(1 << uint(group*4))
		for j := 0; j < 4; j++ {
			shift := uint(2 * j)
			lowScale := float32(int(scaleVals[group*8+j*2]) - 32)
			highScale := float32(int(scaleVals[group*8+j*2+1]) - 32)
			for l := 0; l < 16; l++ {
				low := int8((group*17+j*5+l)%8) - 4
				high := int8((group*19+j*7+l+1)%8) - 4
				qs[qbase+l] |= byte(low&3) << shift
				qs[qbase+16+l] |= byte(high&3) << shift
				if low >= 0 {
					hmask[l] |= mask
				}
				if high >= 0 {
					hmask[16+l] |= mask
				}
				want[obase+j*32+l] = lowScale * float32(low)
				want[obase+j*32+16+l] = highScale * float32(high)
			}
			mask <<= 1
		}
	}

	var b bytes.Buffer
	b.Write(hmask)
	b.Write(qs)
	b.Write(packQ3KFixtureScales(scaleVals))
	writeU16ForTest(&b, 0x3c00) // d = 1.0
	return b.Bytes(), want
}

func packQ3KFixtureScales(scales []byte) []byte {
	out := make([]byte, kScaleSize)
	for i := 0; i < 4; i++ {
		out[i] = (scales[i] & 0x0f) | ((scales[8+i] & 0x0f) << 4)
		out[4+i] = (scales[4+i] & 0x0f) | ((scales[12+i] & 0x0f) << 4)
		out[8+i] = ((scales[i] >> 4) & 3) |
			(((scales[4+i] >> 4) & 3) << 2) |
			(((scales[8+i] >> 4) & 3) << 4) |
			(((scales[12+i] >> 4) & 3) << 6)
	}
	return out
}

func q5FixtureBlock(withMin bool) ([]byte, []float32) {
	qs := make([]byte, qk5/2)
	var qh uint32
	want := make([]float32, qk5)
	code := func(i int) byte {
		return byte((i*5 + 1) % 32)
	}
	for j := 0; j < qk5/2; j++ {
		c0 := code(j)
		c1 := code(j + qk5/2)
		qs[j] = (c0 & 0x0f) | ((c1 & 0x0f) << 4)
		if c0&0x10 != 0 {
			qh |= 1 << uint(j)
		}
		if c1&0x10 != 0 {
			qh |= 1 << uint(j+16)
		}
		if withMin {
			want[j] = float32(c0) + 2
			want[j+qk5/2] = float32(c1) + 2
		} else {
			want[j] = float32(int(c0) - 16)
			want[j+qk5/2] = float32(int(c1) - 16)
		}
	}
	var b bytes.Buffer
	writeU16ForTest(&b, 0x3c00) // d = 1.0
	if withMin {
		writeU16ForTest(&b, 0x4000) // m = 2.0
	}
	_ = binary.Write(&b, binary.LittleEndian, qh)
	b.Write(qs)
	return b.Bytes(), want
}

func q5KFixtureBlock() ([]byte, []float32) {
	ql := make([]byte, 0, qkK/2)
	qh := make([]byte, qkK/8)
	want := make([]float32, qkK)
	scaleBits := []byte{1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1}
	code := func(group, l, offset int) byte {
		return byte((group*7 + l*3 + offset) % 32)
	}
	for group := 0; group < 4; group++ {
		u1 := byte(1 << (2 * group))
		u2 := byte(2 << (2 * group))
		for l := 0; l < 32; l++ {
			low := code(group, l, 1)
			high := code(group, l, 17)
			ql = append(ql, (low&0x0f)|((high&0x0f)<<4))
			if low&0x10 != 0 {
				qh[l] |= u1
			}
			if high&0x10 != 0 {
				qh[l] |= u2
			}
			want[group*64+l] = float32(low)
			want[group*64+32+l] = float32(high)
		}
	}
	var b bytes.Buffer
	writeU16ForTest(&b, 0x3c00) // d = 1.0
	writeU16ForTest(&b, 0x0000) // dmin = 0.0
	b.Write(scaleBits)
	b.Write(qh)
	b.Write(ql)
	return b.Bytes(), want
}

func q6KFixtureBlock() ([]byte, []float32) {
	ql := make([]byte, qkK/2)
	qh := make([]byte, qkK/4)
	scales := make([]byte, qkK/16)
	for i := range scales {
		scales[i] = 1
	}
	want := make([]float32, qkK)
	code := func(i int) byte {
		return byte((i*7 + 3) % 64)
	}
	for chunk := 0; chunk < qkK; chunk += 128 {
		qlBase := chunk / 2
		qhBase := chunk / 4
		for l := 0; l < 32; l++ {
			c1 := code(chunk + l + 0)
			c2 := code(chunk + l + 32)
			c3 := code(chunk + l + 64)
			c4 := code(chunk + l + 96)
			ql[qlBase+l] = (c1 & 0x0f) | ((c3 & 0x0f) << 4)
			ql[qlBase+32+l] = (c2 & 0x0f) | ((c4 & 0x0f) << 4)
			qh[qhBase+l] = ((c1 >> 4) & 3) | (((c2 >> 4) & 3) << 2) | (((c3 >> 4) & 3) << 4) | (((c4 >> 4) & 3) << 6)
			want[chunk+l+0] = float32(int(c1) - 32)
			want[chunk+l+32] = float32(int(c2) - 32)
			want[chunk+l+64] = float32(int(c3) - 32)
			want[chunk+l+96] = float32(int(c4) - 32)
		}
	}
	var b bytes.Buffer
	b.Write(ql)
	b.Write(qh)
	b.Write(scales)
	writeU16ForTest(&b, 0x3c00) // d = 1.0
	return b.Bytes(), want
}

func writeTensorInfoForTest(b *bytes.Buffer, name string, dims []uint64, typ TensorType, off uint64) {
	writeStringForTest(b, name)
	_ = binary.Write(b, binary.LittleEndian, uint32(len(dims)))
	for _, d := range dims {
		_ = binary.Write(b, binary.LittleEndian, d)
	}
	_ = binary.Write(b, binary.LittleEndian, uint32(typ))
	_ = binary.Write(b, binary.LittleEndian, off)
}

func writeF32ForTest(b *bytes.Buffer, v float32) {
	_ = binary.Write(b, binary.LittleEndian, math.Float32bits(v))
}

func writeU16ForTest(b *bytes.Buffer, v uint16) {
	_ = binary.Write(b, binary.LittleEndian, v)
}

func padToAlignment(b *bytes.Buffer, align int) {
	padToLen(b, b.Len()+(align-b.Len()%align)%align)
}

func padToLen(b *bytes.Buffer, n int) {
	for b.Len() < n {
		b.WriteByte(0)
	}
}
