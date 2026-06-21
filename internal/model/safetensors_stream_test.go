package model

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"
	"testing"
)

func TestSingleFileSafetensorsStreamMatchesReadFile(t *testing.T) {
	path := writeTinySafetensors(t, map[string]tinySTTensor{
		"model.embed_tokens.weight": {
			dtype: "F32",
			shape: []int{2, 32},
			data:  f32TestBytes(sequenceFloats(64, 0.01)),
		},
		"model.layers.0.self_attn.q_proj.weight": {
			dtype: "BF16",
			shape: []int{2, 32},
			data:  bf16TestBytes(sequenceFloats(64, -0.07)),
		},
		"model.layers.0.self_attn.k_proj.weight": {
			dtype: "F16",
			shape: []int{2, 32},
			data:  f16TestBytes(sequenceFloats(64, 0.19)),
		},
		"model.norm.weight": {
			dtype: "BF16",
			shape: []int{32},
			data:  bf16TestBytes(sequenceFloats(32, 0.25)),
		},
		"lm_head.weight": {
			dtype: "F32",
			shape: []int{2, 32},
			data:  f32TestBytes(sequenceFloats(64, 0.5)),
		},
	})
	cfg := Config{HiddenSize: 32, NumLayers: 1, VocabSize: 2, TieWordEmbeddings: false}

	want, err := readFileLoadSafetensorsForTest(path, cfg)
	if err != nil {
		t.Fatalf("readfile LoadSafetensors: %v", err)
	}
	got, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("stream LoadSafetensors: %v", err)
	}
	assertModelRawEqual(t, want, got)

	wantQ, err := readFileLoadSafetensorsQuantForTest(path, cfg)
	if err != nil {
		t.Fatalf("readfile LoadSafetensorsQuant: %v", err)
	}
	gotQ, err := LoadSafetensorsQuant(path, cfg)
	if err != nil {
		t.Fatalf("stream LoadSafetensorsQuant: %v", err)
	}
	assertModelRawEqual(t, wantQ, gotQ)
	assertQ8MapsEqual(t, wantQ.q8w, gotQ.q8w)
}

func TestSafetensorsDecodesF16(t *testing.T) {
	src := u16TestBytes([]uint16{
		0x0000, // +0
		0x8000, // -0
		0x3c00, // +1
		0xc000, // -2
		0x0400, // min normal: 2^-14
		0x0001, // min subnormal: 2^-24
		0x7c00, // +Inf
		0xfc00, // -Inf
		0x7e00, // quiet NaN
	})
	got, err := decodeSafetensorF32("f16.weight", stEntry{Dtype: "F16"}, src)
	if err != nil {
		t.Fatalf("decode F16: %v", err)
	}
	wantBits := []uint32{
		0x00000000,
		0x80000000,
		0x3f800000,
		0xc0000000,
		0x38800000,
		0x33800000,
		0x7f800000,
		0xff800000,
		0x7fc00000,
	}
	if len(got) != len(wantBits)*4 {
		t.Fatalf("decoded byte length = %d, want %d", len(got), len(wantBits)*4)
	}
	for i, want := range wantBits {
		bits := binary.LittleEndian.Uint32(got[i*4:])
		if bits != want {
			t.Fatalf("F16[%d] decoded bits = %#08x, want %#08x", i, bits, want)
		}
	}
}

func TestSafetensorsSkipsGPTNeoXAttentionMask(t *testing.T) {
	const maskName = "gpt_neox.layers.0.attention.bias"
	path := writeTinySafetensors(t, map[string]tinySTTensor{
		maskName: {
			dtype: "U8",
			shape: []int{1, 1, 2, 2},
			data:  []byte{1, 0, 1, 0},
		},
		"model.layers.0.self_attn.q_proj.weight": {
			dtype: "F32",
			shape: []int{2, 32},
			data:  f32TestBytes(sequenceFloats(64, 0.07)),
		},
		"model.norm.weight": {
			dtype: "F32",
			shape: []int{32},
			data:  f32TestBytes(sequenceFloats(32, 0.25)),
		},
	})
	cfg := Config{HiddenSize: 32, NumLayers: 1, VocabSize: 2, TieWordEmbeddings: true}

	m, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	if m.has(maskName) {
		t.Fatal("regular safetensors loader retained GPT-NeoX attention mask as a model tensor")
	}

	q, err := LoadSafetensorsQuant(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensorsQuant: %v", err)
	}
	if q.has(maskName) {
		t.Fatal("quant safetensors loader retained GPT-NeoX attention mask as a model tensor")
	}

	unsupported := writeTinySafetensors(t, map[string]tinySTTensor{
		"other.layers.0.attention.bias": {
			dtype: "U8",
			shape: []int{1, 1, 2, 2},
			data:  []byte{1, 0, 1, 0},
		},
	})
	if _, err := LoadSafetensors(unsupported, cfg); err == nil {
		t.Fatal("LoadSafetensors accepted an unrelated U8 attention.bias tensor")
	}
	if _, err := LoadSafetensorsQuant(unsupported, cfg); err == nil {
		t.Fatal("LoadSafetensorsQuant accepted an unrelated U8 attention.bias tensor")
	}
}

func TestSingleFileSafetensorsReaderAtDoesNotReadWholeFile(t *testing.T) {
	buf := tinySafetensorsBytes(t, map[string]tinySTTensor{
		"model.embed_tokens.weight": {
			dtype: "F32",
			shape: []int{4, 32},
			data:  f32TestBytes(sequenceFloats(128, 0.01)),
		},
		"model.layers.0.self_attn.q_proj.weight": {
			dtype: "BF16",
			shape: []int{64, 32},
			data:  bf16TestBytes(sequenceFloats(2048, -0.07)),
		},
		"model.layers.0.mlp.down_proj.weight": {
			dtype: "F32",
			shape: []int{32, 64},
			data:  f32TestBytes(sequenceFloats(2048, 0.13)),
		},
		"model.norm.weight": {
			dtype: "BF16",
			shape: []int{32},
			data:  bf16TestBytes(sequenceFloats(32, 0.25)),
		},
	})
	cfg := Config{HiddenSize: 32, NumLayers: 1, VocabSize: 4, TieWordEmbeddings: true}

	regularReader := &recordingReaderAt{data: buf}
	sf, err := newSafetensorsFile(regularReader, int64(len(buf)), nil)
	if err != nil {
		t.Fatalf("newSafetensorsFile regular: %v", err)
	}
	got, err := loadSafetensorsFile(sf, cfg)
	if err != nil {
		t.Fatalf("loadSafetensorsFile: %v", err)
	}
	if len(got.raw) == 0 {
		t.Fatal("regular streaming load returned empty raw model")
	}
	assertNoWholeFileRead(t, "regular", regularReader, len(buf))

	quantReader := &recordingReaderAt{data: buf}
	sf, err = newSafetensorsFile(quantReader, int64(len(buf)), nil)
	if err != nil {
		t.Fatalf("newSafetensorsFile quant: %v", err)
	}
	m := &Model{Cfg: cfg, manifest: map[string]tensorMeta{}, q8w: map[string]*q8Tensor{}}
	var raw []byte
	off := 0
	if err := quantizeFileInto(sf, m, safetensorsTiedHeader(sf.hdr), &raw, &off, defaultLoadOptions()); err != nil {
		t.Fatalf("quantizeFileInto: %v", err)
	}
	m.raw = raw
	if len(m.q8w) == 0 {
		t.Fatal("quant streaming load returned no Q8 tensors")
	}
	assertNoWholeFileRead(t, "quant", quantReader, len(buf))
}

func TestShardedSafetensorsQuantDirStreamsEachShard(t *testing.T) {
	dir := t.TempDir()
	shardA, shardB := "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
	tensorsA := map[string]tinySTTensor{
		"model.embed_tokens.weight": {
			dtype: "F32",
			shape: []int{4, 32},
			data:  f32TestBytes(sequenceFloats(128, 0.01)),
		},
		"model.layers.0.self_attn.q_proj.weight": {
			dtype: "BF16",
			shape: []int{2, 32},
			data:  bf16TestBytes(sequenceFloats(64, -0.07)),
		},
	}
	tensorsB := map[string]tinySTTensor{
		"model.layers.0.mlp.down_proj.weight": {
			dtype: "F32",
			shape: []int{2, 32},
			data:  f32TestBytes(sequenceFloats(64, 0.13)),
		},
		"model.norm.weight": {
			dtype: "BF16",
			shape: []int{32},
			data:  bf16TestBytes(sequenceFloats(32, 0.25)),
		},
	}
	bufA := tinySafetensorsBytes(t, tensorsA)
	bufB := tinySafetensorsBytes(t, tensorsB)
	weightMap := map[string]string{}
	for name := range tensorsA {
		weightMap[name] = shardA
	}
	for name := range tensorsB {
		weightMap[name] = shardB
	}
	index, err := json.Marshal(struct {
		WeightMap map[string]string `json:"weight_map"`
	}{WeightMap: weightMap})
	if err != nil {
		t.Fatalf("marshal sharded index: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), index, 0o644); err != nil {
		t.Fatalf("write sharded index: %v", err)
	}

	pathA, pathB := filepath.Join(dir, shardA), filepath.Join(dir, shardB)
	readers := map[string]*recordingReaderAt{
		pathA: {data: bufA},
		pathB: {data: bufB},
	}
	open := func(path string) (*safetensorsFile, error) {
		r, ok := readers[path]
		if !ok {
			return nil, os.ErrNotExist
		}
		return newSafetensorsFile(r, int64(len(r.data)), nil)
	}
	cfg := Config{HiddenSize: 32, NumLayers: 1, VocabSize: 4, TieWordEmbeddings: true}

	got, err := loadSafetensorsQuantDir(dir, cfg, open)
	if err != nil {
		t.Fatalf("loadSafetensorsQuantDir: %v", err)
	}
	want := &Model{Cfg: cfg, manifest: map[string]tensorMeta{}, q8w: map[string]*q8Tensor{}}
	var raw []byte
	off := 0
	if err := quantizeBlobInto(bufA, want, true, &raw, &off); err != nil {
		t.Fatalf("quantizeBlobInto shard A: %v", err)
	}
	if err := quantizeBlobInto(bufB, want, true, &raw, &off); err != nil {
		t.Fatalf("quantizeBlobInto shard B: %v", err)
	}
	want.raw = raw
	assertModelRawEqual(t, want, got)
	assertQ8MapsEqual(t, want.q8w, got.q8w)
	assertNoWholeFileRead(t, shardA, readers[pathA], len(bufA))
	assertNoWholeFileRead(t, shardB, readers[pathB], len(bufB))
}

func TestShardedSafetensorsRegularDirStreamsEachShard(t *testing.T) {
	dir := t.TempDir()
	shardA, shardB := "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
	tensorsA := map[string]tinySTTensor{
		"model.embed_tokens.weight": {
			dtype: "F32",
			shape: []int{4, 32},
			data:  f32TestBytes(sequenceFloats(128, 0.01)),
		},
		"model.layers.0.self_attn.q_proj.weight": {
			dtype: "BF16",
			shape: []int{2, 32},
			data:  bf16TestBytes(sequenceFloats(64, -0.07)),
		},
	}
	tensorsB := map[string]tinySTTensor{
		"model.layers.0.mlp.down_proj.weight": {
			dtype: "F32",
			shape: []int{2, 32},
			data:  f32TestBytes(sequenceFloats(64, 0.13)),
		},
		"model.norm.weight": {
			dtype: "BF16",
			shape: []int{32},
			data:  bf16TestBytes(sequenceFloats(32, 0.25)),
		},
	}
	bufA := tinySafetensorsBytes(t, tensorsA)
	bufB := tinySafetensorsBytes(t, tensorsB)
	weightMap := map[string]string{}
	for name := range tensorsA {
		weightMap[name] = shardA
	}
	for name := range tensorsB {
		weightMap[name] = shardB
	}
	index, err := json.Marshal(struct {
		WeightMap map[string]string `json:"weight_map"`
	}{WeightMap: weightMap})
	if err != nil {
		t.Fatalf("marshal sharded index: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors.index.json"), index, 0o644); err != nil {
		t.Fatalf("write sharded index: %v", err)
	}

	pathA, pathB := filepath.Join(dir, shardA), filepath.Join(dir, shardB)
	readers := map[string]*recordingReaderAt{
		pathA: {data: bufA},
		pathB: {data: bufB},
	}
	open := func(path string) (*safetensorsFile, error) {
		r, ok := readers[path]
		if !ok {
			return nil, os.ErrNotExist
		}
		return newSafetensorsFile(r, int64(len(r.data)), nil)
	}
	cfg := Config{HiddenSize: 32, NumLayers: 1, VocabSize: 4, TieWordEmbeddings: true}

	got, err := loadSafetensorsDir(dir, cfg, open)
	if err != nil {
		t.Fatalf("LoadSafetensorsDir: %v", err)
	}
	want := &Model{Cfg: cfg, manifest: map[string]tensorMeta{}}
	var raw []byte
	off := 0
	if err := appendReadFileSafetensorsForTest(bufA, want.manifest, &raw, &off); err != nil {
		t.Fatalf("readfile shard A: %v", err)
	}
	if err := appendReadFileSafetensorsForTest(bufB, want.manifest, &raw, &off); err != nil {
		t.Fatalf("readfile shard B: %v", err)
	}
	want.raw = raw
	assertModelRawEqual(t, want, got)
	assertNoWholeFileRead(t, shardA, readers[pathA], len(bufA))
	assertNoWholeFileRead(t, shardB, readers[pathB], len(bufB))
}

func TestSafetensorsLoadsGPTOSSMXFP4Experts(t *testing.T) {
	cfg := Config{
		HiddenSize:       4,
		IntermediateSize: 2,
		NumLayers:        1,
		NumExperts:       1,
		NumExpertsPerTok: 1,
		ModelType:        "gpt_oss",
	}
	path := writeTinySafetensors(t, map[string]tinySTTensor{
		layerName(0, "mlp.experts.gate_up_proj_blocks"): {
			dtype: "U8",
			shape: []int{1, 4, 1, 2},
			data:  []byte{0x21, 0x43, 0x65, 0x97, 0xBA, 0xDC, 0xFE, 0x10},
		},
		layerName(0, "mlp.experts.gate_up_proj_scales"): {
			dtype: "U8",
			shape: []int{1, 4, 1},
			data:  []byte{127, 127, 127, 127},
		},
		layerName(0, "mlp.experts.down_proj_blocks"): {
			dtype: "U8",
			shape: []int{1, 4, 1, 1},
			data:  []byte{0x21, 0x43, 0x65, 0x97},
		},
		layerName(0, "mlp.experts.down_proj_scales"): {
			dtype: "U8",
			shape: []int{1, 4, 1},
			data:  []byte{127, 127, 127, 127},
		},
	})

	m, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	assertFloat32BitsEqual(t, "gpt-oss MXFP4 gate", []float32{
		0.5, 1, 1.5, 2,
		-1, -1.5, -2, -3,
	}, m.tensor(expertName(0, 0, "gate_proj.weight")))
	assertFloat32BitsEqual(t, "gpt-oss MXFP4 up", []float32{
		3, 4, 6, -0.5,
		-4, -6, 0, 0.5,
	}, m.tensor(expertName(0, 0, "up_proj.weight")))
	assertFloat32BitsEqual(t, "gpt-oss MXFP4 down", []float32{
		0.5, 1,
		1.5, 2,
		3, 4,
		6, -0.5,
	}, m.tensor(expertName(0, 0, "down_proj.weight")))

	for _, name := range []string{
		layerName(0, "mlp.experts.gate_up_proj_blocks"),
		layerName(0, "mlp.experts.gate_up_proj_scales"),
		layerName(0, "mlp.experts.gate_up_proj"),
		layerName(0, "mlp.experts.down_proj_blocks"),
		layerName(0, "mlp.experts.down_proj_scales"),
		layerName(0, "mlp.experts.down_proj"),
	} {
		if m.has(name) {
			t.Fatalf("gpt-oss MXFP4 source tensor %s still present after load", name)
		}
	}
}

func TestOptionalCachedGPTOSSMXFP4SafetensorsLoads(t *testing.T) {
	cfgPath, stPath, ok := findCachedGPTOSSMXFP4ForTest(t)
	if !ok {
		t.Skip("tiny-random/gpt-oss-mxfp4 is not present in the local Hugging Face cache")
	}
	var cfg Config
	if err := readJSON(cfgPath, &cfg); err != nil {
		t.Fatalf("read config: %v", err)
	}
	m, err := LoadSafetensors(stPath, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors(%s): %v", stPath, err)
	}

	for _, tt := range []struct {
		name string
		want []int
	}{
		{expertName(0, 0, "gate_proj.weight"), []int{cfg.IntermediateSize, cfg.HiddenSize}},
		{expertName(0, 0, "up_proj.weight"), []int{cfg.IntermediateSize, cfg.HiddenSize}},
		{expertName(0, 0, "down_proj.weight"), []int{cfg.HiddenSize, cfg.IntermediateSize}},
	} {
		meta, ok := m.manifest[tt.name]
		if !ok {
			t.Fatalf("missing canonical tensor %s", tt.name)
		}
		if !sameShape(meta.Shape, tt.want) {
			t.Fatalf("%s shape = %v, want %v", tt.name, meta.Shape, tt.want)
		}
		if meta.Nbytes != tensorElemCount(tt.want)*4 {
			t.Fatalf("%s nbytes = %d, want %d", tt.name, meta.Nbytes, tensorElemCount(tt.want)*4)
		}
	}
	for _, name := range []string{
		layerName(0, "mlp.experts.gate_up_proj_blocks"),
		layerName(0, "mlp.experts.gate_up_proj_scales"),
		layerName(0, "mlp.experts.gate_up_proj"),
		layerName(0, "mlp.experts.down_proj_blocks"),
		layerName(0, "mlp.experts.down_proj_scales"),
		layerName(0, "mlp.experts.down_proj"),
	} {
		if m.has(name) {
			t.Fatalf("cached gpt-oss MXFP4 source tensor %s still present after load", name)
		}
	}
}

func TestOptionalGPTOSSMXFP4ForwardMatchesHFOracle(t *testing.T) {
	const oracleDir = ".cache/oracle-gptoss-mxfp4"
	_, doc := loadFixtureDir(t, oracleDir, true)
	resolved, _ := resolveOracleDir(oracleDir)
	_, stPath, ok := findCachedGPTOSSMXFP4ForTest(t)
	if !ok {
		t.Skip("tiny-random/gpt-oss-mxfp4 is not present in the local Hugging Face cache")
	}
	var cfg Config
	if err := readJSON(filepath.Join(resolved, "config.json"), &cfg); err != nil {
		t.Fatalf("read config: %v", err)
	}
	if !cfg.isGPTOSS() || !cfg.IsMoE() {
		t.Fatalf("%s config is not gpt-oss MoE: family=%q experts=%d topk=%d",
			oracleDir, cfg.archFamilyKey(), cfg.NumExperts, cfg.NumExpertsPerTok)
	}
	m, err := LoadSafetensors(stPath, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors(%s): %v", stPath, err)
	}
	for _, name := range []string{
		layerName(0, "mlp.experts.gate_up_proj_blocks"),
		layerName(0, "mlp.experts.gate_up_proj_scales"),
		layerName(0, "mlp.experts.down_proj_blocks"),
		layerName(0, "mlp.experts.down_proj_scales"),
	} {
		if m.has(name) {
			t.Fatalf("cached gpt-oss MXFP4 source tensor %s still present after load", name)
		}
	}
	assertForwardMatchesHFOracle(t, resolved, m, doc)
}

func TestOptionalCachedCanonicalFamilySafetensorsLoad(t *testing.T) {
	tests := []struct {
		name        string
		repo        string
		family      string
		topology    BlockTopology
		qkNormShape string
		wantTensors []string
		check       func(t *testing.T, cfg Config, m *Model)
	}{
		{
			name:     "gemma",
			repo:     "Xenova/tiny-random-GemmaForCausalLM",
			family:   "gemma",
			topology: PreNorm,
			wantTensors: []string{
				layerName(0, "input_layernorm.weight"),
				layerName(0, "post_attention_layernorm.weight"),
			},
		},
		{
			name:        "gemma3",
			repo:        "bumblebee-testing/tiny-random-Gemma3ForCausalLM",
			family:      "gemma3",
			topology:    SandwichNorm,
			qkNormShape: "head",
			wantTensors: []string{
				layerName(0, "input_layernorm.weight"),
				layerName(0, "post_attention_layernorm.weight"),
				layerName(0, "pre_feedforward_layernorm.weight"),
				layerName(0, "post_feedforward_layernorm.weight"),
			},
			check: func(t *testing.T, cfg Config, m *Model) {
				t.Helper()
				if len(cfg.LayerTypes) == 0 || len(cfg.Window) == 0 || len(cfg.RopeThetaPerLayer) == 0 {
					t.Fatalf("gemma3 layer axes not derived: layer_types=%v window=%v rope_theta_per_layer=%v",
						cfg.LayerTypes, cfg.Window, cfg.RopeThetaPerLayer)
				}
				if !cfg.QKNorm || cfg.QueryPreAttnScalar == 0 {
					t.Fatalf("gemma3 qk/query scale axes not derived: qknorm=%v query_pre_attn_scalar=%d",
						cfg.QKNorm, cfg.QueryPreAttnScalar)
				}
			},
		},
		{
			name:     "cohere",
			repo:     "optimum-intel-internal-testing/tiny-random-CohereForCausalLM",
			family:   "cohere",
			topology: ParallelResidual,
			wantTensors: []string{
				layerName(0, "input_layernorm.weight"),
			},
			check: func(t *testing.T, cfg Config, m *Model) {
				t.Helper()
				if !cfg.LayerNorm || cfg.LogitScale != 0.0625 {
					t.Fatalf("cohere axes not derived: layernorm=%v logit_scale=%v", cfg.LayerNorm, cfg.LogitScale)
				}
			},
		},
		{
			name:        "olmo2",
			repo:        "optimum-intel-internal-testing/tiny-random-olmo2",
			family:      "olmo2",
			topology:    PostNorm,
			qkNormShape: "projection",
			wantTensors: []string{
				layerName(0, "post_attention_layernorm.weight"),
				layerName(0, "post_feedforward_layernorm.weight"),
			},
			check: func(t *testing.T, cfg Config, m *Model) {
				t.Helper()
				if !cfg.QKNorm {
					t.Fatal("olmo2 QKNorm axis not derived")
				}
				if m.has(layerName(0, "input_layernorm.weight")) {
					t.Fatal("olmo2 raw checkpoint unexpectedly has pre-attention input_layernorm")
				}
			},
		},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			cfgPath, stPath, ok := findCachedHFModelSnapshotForTest(t, tt.repo)
			if !ok {
				t.Skip(tt.repo + " is not present in the local Hugging Face cache")
			}
			var cfg Config
			if err := readJSON(cfgPath, &cfg); err != nil {
				t.Fatalf("read config: %v", err)
			}
			if !strings.Contains(cfg.archFamilyKey(), tt.family) {
				t.Fatalf("%s family = %q, want %s", tt.repo, cfg.archFamilyKey(), tt.family)
			}
			if cfg.BlockTopology != tt.topology {
				t.Fatalf("%s topology = %v, want %v", tt.repo, cfg.BlockTopology, tt.topology)
			}
			m, err := LoadSafetensors(stPath, cfg)
			if err != nil {
				t.Fatalf("LoadSafetensors(%s): %v", stPath, err)
			}
			assertCanonicalDenseLayerShapes(t, m, cfg, 0)
			for _, name := range tt.wantTensors {
				assertTensorShape(t, m, name, []int{cfg.HiddenSize})
			}
			switch tt.qkNormShape {
			case "head":
				assertTensorShape(t, m, layerName(0, "self_attn.q_norm.weight"), []int{cfg.HeadDim})
				assertTensorShape(t, m, layerName(0, "self_attn.k_norm.weight"), []int{cfg.HeadDim})
			case "projection":
				assertTensorShape(t, m, layerName(0, "self_attn.q_norm.weight"), []int{cfg.NumHeads * cfg.HeadDim})
				assertTensorShape(t, m, layerName(0, "self_attn.k_norm.weight"), []int{cfg.NumKVHeads * cfg.HeadDim})
			}
			if tt.check != nil {
				tt.check(t, cfg, m)
			}
		})
	}
}

func TestOptionalCachedF16SourceFormatSafetensorsLoad(t *testing.T) {
	tests := []struct {
		name   string
		repo   string
		family string
		check  func(t *testing.T, cfg Config, m *Model)
	}{
		{
			name:   "falcon",
			repo:   "yujiepan/falcon-tiny-random",
			family: "falcon",
			check: func(t *testing.T, cfg Config, m *Model) {
				t.Helper()
				assertCanonicalAttentionShapes(t, m, cfg, 0)
				assertTensorShape(t, m, layerName(0, "input_layernorm.weight"), []int{cfg.HiddenSize})
				assertTensorShape(t, m, layerName(0, "input_layernorm.bias"), []int{cfg.HiddenSize})
				assertTensorShape(t, m, layerName(0, "mlp.gate_proj.weight"), []int{cfg.IntermediateSize, cfg.HiddenSize})
				assertTensorShape(t, m, layerName(0, "mlp.down_proj.weight"), []int{cfg.HiddenSize, cfg.IntermediateSize})
				if cfg.NumKVHeads != 1 || cfg.IntermediateSize != 4*cfg.HiddenSize {
					t.Fatalf("falcon raw config aliases not derived: kv=%d intermediate=%d hidden=%d",
						cfg.NumKVHeads, cfg.IntermediateSize, cfg.HiddenSize)
				}
			},
		},
		{
			name:   "mixtral",
			repo:   "yujiepan/mixtral-tiny-random",
			family: "mixtral",
			check: func(t *testing.T, cfg Config, m *Model) {
				t.Helper()
				assertCanonicalAttentionShapes(t, m, cfg, 0)
				assertTensorShape(t, m, routerName(0), []int{cfg.NumExperts, cfg.HiddenSize})
				assertTensorShape(t, m, expertName(0, 0, "gate_proj.weight"), []int{cfg.IntermediateSize, cfg.HiddenSize})
				assertTensorShape(t, m, expertName(0, 0, "up_proj.weight"), []int{cfg.IntermediateSize, cfg.HiddenSize})
				assertTensorShape(t, m, expertName(0, 0, "down_proj.weight"), []int{cfg.HiddenSize, cfg.IntermediateSize})
				if !cfg.IsMoE() || cfg.NumExperts != 8 || cfg.NumExpertsPerTok != 2 {
					t.Fatalf("mixtral raw config did not preserve MoE fields: experts=%d topk=%d",
						cfg.NumExperts, cfg.NumExpertsPerTok)
				}
			},
		},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			cfgPath, stPath, ok := findCachedHFModelSnapshotForTest(t, tt.repo)
			if !ok {
				t.Skip(tt.repo + " is not present in the local Hugging Face cache")
			}
			var cfg Config
			if err := readJSON(cfgPath, &cfg); err != nil {
				t.Fatalf("read config: %v", err)
			}
			if !strings.Contains(cfg.archFamilyKey(), tt.family) {
				t.Fatalf("%s family = %q, want %s", tt.repo, cfg.archFamilyKey(), tt.family)
			}
			m, err := LoadSafetensors(stPath, cfg)
			if err != nil {
				t.Fatalf("LoadSafetensors(%s): %v", stPath, err)
			}
			assertTensorShape(t, m, "model.embed_tokens.weight", []int{cfg.VocabSize, cfg.HiddenSize})
			assertTensorShape(t, m, "model.norm.weight", []int{cfg.HiddenSize})
			tt.check(t, cfg, m)
		})
	}
}

func TestOptionalCachedGPTNeoXSourceFormatSafetensorsLoad(t *testing.T) {
	const oracleDir = ".cache/oracle-gptneox"
	resolved, ok := resolveOracleDir(oracleDir)
	if !ok {
		t.Skip("no exported GPT-NeoX oracle config")
	}
	stPath, ok := findCachedHFModelSafetensorsForTest(t, "optimum-intel-internal-testing/tiny-random-GPTNeoXForCausalLM")
	if !ok {
		t.Skip("tiny-random GPT-NeoX safetensors are not present in the local Hugging Face cache")
	}
	var cfg Config
	if err := readJSON(filepath.Join(resolved, "config.json"), &cfg); err != nil {
		t.Fatalf("read config: %v", err)
	}
	m, err := LoadSafetensors(stPath, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors(%s): %v", stPath, err)
	}
	if m.has("gpt_neox.layers.0.attention.bias") {
		t.Fatal("raw GPT-NeoX attention mask was retained as a model tensor")
	}
	assertTensorShape(t, m, "model.embed_tokens.weight", []int{cfg.VocabSize, cfg.HiddenSize})
	assertTensorShape(t, m, "lm_head.weight", []int{cfg.VocabSize, cfg.HiddenSize})
	assertTensorShape(t, m, "model.norm.weight", []int{cfg.HiddenSize})
	assertTensorShape(t, m, "model.norm.bias", []int{cfg.HiddenSize})
	assertCanonicalAttentionShapes(t, m, cfg, 0)
	assertTensorShape(t, m, layerName(0, "self_attn.q_proj.bias"), []int{cfg.NumHeads * cfg.HeadDim})
	assertTensorShape(t, m, layerName(0, "self_attn.k_proj.bias"), []int{cfg.NumKVHeads * cfg.HeadDim})
	assertTensorShape(t, m, layerName(0, "self_attn.v_proj.bias"), []int{cfg.NumKVHeads * cfg.HeadDim})
	assertTensorShape(t, m, layerName(0, "input_layernorm.bias"), []int{cfg.HiddenSize})
	assertTensorShape(t, m, layerName(0, "post_attention_layernorm.bias"), []int{cfg.HiddenSize})
	assertTensorShape(t, m, layerName(0, "mlp.gate_proj.weight"), []int{cfg.IntermediateSize, cfg.HiddenSize})
	assertTensorShape(t, m, layerName(0, "mlp.gate_proj.bias"), []int{cfg.IntermediateSize})
	assertTensorShape(t, m, layerName(0, "mlp.down_proj.weight"), []int{cfg.HiddenSize, cfg.IntermediateSize})
	assertTensorShape(t, m, layerName(0, "mlp.down_proj.bias"), []int{cfg.HiddenSize})
}

func TestOptionalCachedParallelFamilySourceFormatSafetensorsLoad(t *testing.T) {
	tests := []struct {
		name      string
		repo      string
		family    string
		oracleDir string
		check     func(t *testing.T, cfg Config, m *Model)
	}{
		{
			name:      "mpt",
			repo:      "optimum-intel-internal-testing/tiny-random-MptForCausalLM",
			family:    "mpt",
			oracleDir: ".cache/oracle-mpt",
			check: func(t *testing.T, cfg Config, m *Model) {
				t.Helper()
				assertCanonicalAttentionShapes(t, m, cfg, 0)
				if m.has(layerName(0, "self_attn.qkv_proj.weight")) {
					t.Fatal("mpt fused qkv tensor still present after canonical split")
				}
				assertTensorShape(t, m, layerName(0, "input_layernorm.weight"), []int{cfg.HiddenSize})
				assertTensorShape(t, m, layerName(0, "post_attention_layernorm.weight"), []int{cfg.HiddenSize})
				assertTensorShape(t, m, layerName(0, "mlp.gate_proj.weight"), []int{cfg.IntermediateSize, cfg.HiddenSize})
				assertTensorShape(t, m, layerName(0, "mlp.down_proj.weight"), []int{cfg.HiddenSize, cfg.IntermediateSize})
				if !cfg.LayerNorm || !cfg.DenseMLP || !cfg.ActGeluErf || !cfg.Alibi {
					t.Fatalf("mpt raw config axes not derived: layerNorm=%v denseMLP=%v geluErf=%v alibi=%v",
						cfg.LayerNorm, cfg.DenseMLP, cfg.ActGeluErf, cfg.Alibi)
				}
			},
		},
		{
			name:   "stablelm",
			repo:   "optimum-intel-internal-testing/tiny-random-StableLmForCausalLM",
			family: "stablelm",
			check: func(t *testing.T, cfg Config, m *Model) {
				t.Helper()
				assertTensorShape(t, m, "model.embed_tokens.weight", []int{cfg.VocabSize, cfg.HiddenSize})
				assertCanonicalAttentionShapes(t, m, cfg, 0)
				assertTensorShape(t, m, layerName(0, "input_layernorm.weight"), []int{cfg.HiddenSize})
				assertTensorShape(t, m, layerName(0, "input_layernorm.bias"), []int{cfg.HiddenSize})
				assertTensorShape(t, m, layerName(0, "post_attention_layernorm.weight"), []int{cfg.HiddenSize})
				assertTensorShape(t, m, layerName(0, "post_attention_layernorm.bias"), []int{cfg.HiddenSize})
				assertTensorShape(t, m, layerName(0, "mlp.gate_proj.weight"), []int{cfg.IntermediateSize, cfg.HiddenSize})
				assertTensorShape(t, m, layerName(0, "mlp.up_proj.weight"), []int{cfg.IntermediateSize, cfg.HiddenSize})
				assertTensorShape(t, m, layerName(0, "mlp.down_proj.weight"), []int{cfg.HiddenSize, cfg.IntermediateSize})
				if !cfg.LayerNorm {
					t.Fatal("stablelm raw config did not derive LayerNorm")
				}
			},
		},
		{
			name:   "gpt-oss-f32",
			repo:   "optimum-internal-testing/tiny-random-gpt-oss",
			family: "gptoss",
			check: func(t *testing.T, cfg Config, m *Model) {
				t.Helper()
				if !cfg.isGPTOSS() || !cfg.IsMoE() {
					t.Fatalf("gpt-oss raw config is not MoE: family=%q experts=%d topk=%d",
						cfg.archFamilyKey(), cfg.NumExperts, cfg.NumExpertsPerTok)
				}
				assertTensorShape(t, m, "model.embed_tokens.weight", []int{cfg.VocabSize, cfg.HiddenSize})
				assertCanonicalAttentionShapes(t, m, cfg, 0)
				assertTensorShape(t, m, layerName(0, "self_attn.q_proj.bias"), []int{cfg.NumHeads * cfg.HeadDim})
				assertTensorShape(t, m, layerName(0, "self_attn.k_proj.bias"), []int{cfg.NumKVHeads * cfg.HeadDim})
				assertTensorShape(t, m, layerName(0, "self_attn.v_proj.bias"), []int{cfg.NumKVHeads * cfg.HeadDim})
				assertTensorShape(t, m, layerName(0, "self_attn.sinks"), []int{cfg.NumHeads})
				assertTensorShape(t, m, routerName(0), []int{cfg.NumExperts, cfg.HiddenSize})
				assertTensorShape(t, m, routerBiasName(0), []int{cfg.NumExperts})
				assertTensorShape(t, m, expertName(0, 0, "gate_proj.weight"), []int{cfg.IntermediateSize, cfg.HiddenSize})
				assertTensorShape(t, m, expertName(0, 0, "up_proj.weight"), []int{cfg.IntermediateSize, cfg.HiddenSize})
				assertTensorShape(t, m, expertName(0, 0, "down_proj.weight"), []int{cfg.HiddenSize, cfg.IntermediateSize})
				assertTensorShape(t, m, expertName(0, 0, "gate_proj.bias"), []int{cfg.IntermediateSize})
				assertTensorShape(t, m, expertName(0, 0, "up_proj.bias"), []int{cfg.IntermediateSize})
				assertTensorShape(t, m, expertName(0, 0, "down_proj.bias"), []int{cfg.HiddenSize})
				for _, name := range []string{
					layerName(0, "mlp.experts.gate_up_proj"),
					layerName(0, "mlp.experts.gate_up_proj_bias"),
					layerName(0, "mlp.experts.down_proj"),
					layerName(0, "mlp.experts.down_proj_bias"),
				} {
					if m.has(name) {
						t.Fatalf("gpt-oss F32 source tensor %s still present after load", name)
					}
				}
			},
		},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			cfgPath, stPath, ok := findCachedHFModelSnapshotForTest(t, tt.repo)
			if tt.oracleDir != "" {
				resolved, hasOracle := resolveOracleDir(tt.oracleDir)
				if !hasOracle {
					t.Skip("no exported oracle config for " + tt.name)
				}
				stPath, ok = findCachedHFModelSafetensorsForTest(t, tt.repo)
				cfgPath = filepath.Join(resolved, "config.json")
			}
			if !ok {
				t.Skip(tt.repo + " is not present in the local Hugging Face cache")
			}
			var cfg Config
			if err := readJSON(cfgPath, &cfg); err != nil {
				t.Fatalf("read config: %v", err)
			}
			if !strings.Contains(cfg.archFamilyKey(), tt.family) {
				t.Fatalf("%s family = %q, want %s", tt.repo, cfg.archFamilyKey(), tt.family)
			}
			m, err := LoadSafetensors(stPath, cfg)
			if err != nil {
				t.Fatalf("LoadSafetensors(%s): %v", stPath, err)
			}
			assertTensorShape(t, m, "model.norm.weight", []int{cfg.HiddenSize})
			if m.has("model.norm.bias") {
				assertTensorShape(t, m, "model.norm.bias", []int{cfg.HiddenSize})
			}
			if m.has("lm_head.weight") {
				assertTensorShape(t, m, "lm_head.weight", []int{cfg.VocabSize, cfg.HiddenSize})
			}
			tt.check(t, cfg, m)
		})
	}
}

func TestSafetensorsStreamRejectsMalformedPayloads(t *testing.T) {
	cfg := Config{HiddenSize: 32, NumLayers: 1, VocabSize: 2, TieWordEmbeddings: false}
	tests := []struct {
		name   string
		header map[string]stEntry
		data   []byte
	}{
		{
			name: "offset overrun",
			header: map[string]stEntry{
				"model.norm.weight": {Dtype: "F32", Shape: []int{1}, DataOffsets: []int{0, 8}},
			},
			data: make([]byte, 4),
		},
		{
			name: "odd bf16 length",
			header: map[string]stEntry{
				"model.norm.weight": {Dtype: "BF16", Shape: []int{1}, DataOffsets: []int{0, 1}},
			},
			data: []byte{0},
		},
		{
			name: "odd f16 length",
			header: map[string]stEntry{
				"model.norm.weight": {Dtype: "F16", Shape: []int{1}, DataOffsets: []int{0, 1}},
			},
			data: []byte{0},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			path := writeRawSafetensors(t, tt.header, tt.data)
			if _, err := LoadSafetensors(path, cfg); err == nil {
				t.Fatal("LoadSafetensors accepted malformed payload")
			}
			if _, err := LoadSafetensorsQuant(path, cfg); err == nil {
				t.Fatal("LoadSafetensorsQuant accepted malformed payload")
			}
		})
	}
}

func TestSafetensorsPreSizeRejectsShapeOverflow(t *testing.T) {
	cfg := Config{ModelType: "glm_moe_dsa"}
	path := writeRawSafetensors(t, map[string]stEntry{
		"model.norm.weight": {
			Dtype:       "F32",
			Shape:       []int{math.MaxInt, 2},
			DataOffsets: []int{0, 4},
		},
	}, make([]byte, 4))

	if _, err := LoadSafetensors(path, cfg); err == nil {
		t.Fatal("LoadSafetensors accepted a shape that overflows pre-size accounting")
	}
}

type tinySTTensor struct {
	dtype string
	shape []int
	data  []byte
}

func writeTinySafetensors(t *testing.T, tensors map[string]tinySTTensor) string {
	t.Helper()
	out := tinySafetensorsBytes(t, tensors)
	path := filepath.Join(t.TempDir(), "model.safetensors")
	if err := os.WriteFile(path, out, 0o644); err != nil {
		t.Fatalf("write safetensors: %v", err)
	}
	return path
}

func tinySafetensorsBytes(t *testing.T, tensors map[string]tinySTTensor) []byte {
	t.Helper()
	names := make([]string, 0, len(tensors))
	for name := range tensors {
		names = append(names, name)
	}
	sort.Strings(names)

	hdr := map[string]stEntry{}
	var data []byte
	for _, name := range names {
		tt := tensors[name]
		start := len(data)
		data = append(data, tt.data...)
		hdr[name] = stEntry{Dtype: tt.dtype, Shape: tt.shape, DataOffsets: []int{start, len(data)}}
	}
	header, err := json.Marshal(hdr)
	if err != nil {
		t.Fatalf("marshal header: %v", err)
	}
	out := make([]byte, 8, 8+len(header)+len(data))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(header)))
	out = append(out, header...)
	out = append(out, data...)
	return out
}

func writeRawSafetensors(t *testing.T, hdr map[string]stEntry, data []byte) string {
	t.Helper()
	header, err := json.Marshal(hdr)
	if err != nil {
		t.Fatalf("marshal header: %v", err)
	}
	out := make([]byte, 8, 8+len(header)+len(data))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(header)))
	out = append(out, header...)
	out = append(out, data...)
	path := filepath.Join(t.TempDir(), "bad.safetensors")
	if err := os.WriteFile(path, out, 0o644); err != nil {
		t.Fatalf("write safetensors: %v", err)
	}
	return path
}

func sequenceFloats(n int, seed float32) []float32 {
	v := make([]float32, n)
	for i := range v {
		x := seed + float32(i%17-8)*0.03125 + float32(i/17)*0.0078125
		v[i] = x
	}
	return v
}

func f32TestBytes(vals []float32) []byte {
	b := make([]byte, len(vals)*4)
	for i, v := range vals {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(v))
	}
	return b
}

func bf16TestBytes(vals []float32) []byte {
	b := make([]byte, len(vals)*2)
	for i, v := range vals {
		binary.LittleEndian.PutUint16(b[i*2:], uint16(math.Float32bits(v)>>16))
	}
	return b
}

func f16TestBytes(vals []float32) []byte {
	b := make([]byte, len(vals)*2)
	for i, v := range vals {
		binary.LittleEndian.PutUint16(b[i*2:], f32bitsToF16bits(math.Float32bits(v)))
	}
	return b
}

func f32bitsToF16bits(bits uint32) uint16 {
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits >> 23) & 0xff)
	frac := bits & 0x7fffff
	if exp == 0xff {
		if frac == 0 {
			return sign | 0x7c00
		}
		return sign | 0x7e00
	}
	e := exp - 127 + 15
	if e >= 0x1f {
		return sign | 0x7c00
	}
	if e <= 0 {
		if e < -10 {
			return sign
		}
		mant := frac | 0x800000
		shift := uint(14 - e)
		half := mant >> shift
		if (mant>>(shift-1))&1 != 0 {
			half++
		}
		return sign | uint16(half)
	}
	half := uint16(e<<10) | uint16(frac>>13)
	if frac&0x00001000 != 0 {
		half++
	}
	return sign | half
}

func u16TestBytes(vals []uint16) []byte {
	b := make([]byte, len(vals)*2)
	for i, v := range vals {
		binary.LittleEndian.PutUint16(b[i*2:], v)
	}
	return b
}

type recordingReaderAt struct {
	data     []byte
	reads    int
	maxRead  int
	fullRead bool
}

func (r *recordingReaderAt) ReadAt(p []byte, off int64) (int, error) {
	r.reads++
	if len(p) > r.maxRead {
		r.maxRead = len(p)
	}
	if len(p) >= len(r.data) {
		r.fullRead = true
	}
	if off < 0 || off > int64(len(r.data)) {
		return 0, io.EOF
	}
	n := copy(p, r.data[int(off):])
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}

func assertNoWholeFileRead(t *testing.T, name string, r *recordingReaderAt, total int) {
	t.Helper()
	if r.reads == 0 {
		t.Fatalf("%s loader did not read from ReaderAt", name)
	}
	if r.fullRead || r.maxRead >= total {
		t.Fatalf("%s loader read the whole safetensors file: max read %d, total %d", name, r.maxRead, total)
	}
}

func findCachedGPTOSSMXFP4ForTest(t *testing.T) (string, string, bool) {
	t.Helper()
	return findCachedHFModelSnapshotForTest(t, "tiny-random/gpt-oss-mxfp4")
}

func findCachedHFModelSnapshotForTest(t *testing.T, repo string) (string, string, bool) {
	t.Helper()
	cacheName := "models--" + strings.ReplaceAll(repo, "/", "--")
	patterns := []string{}
	if home, err := os.UserHomeDir(); err == nil {
		patterns = append(patterns, filepath.Join(home, ".cache", "huggingface", "hub", cacheName, "snapshots", "*"))
	}
	patterns = append(patterns, filepath.Join("/mnt", "*", "Users", "*", ".cache", "huggingface", "hub", cacheName, "snapshots", "*"))

	var snapshots []string
	for _, pattern := range patterns {
		matches, err := filepath.Glob(pattern)
		if err != nil {
			t.Fatalf("glob Hugging Face cache: %v", err)
		}
		snapshots = append(snapshots, matches...)
	}
	sort.Strings(snapshots)
	for _, dir := range snapshots {
		cfgPath := filepath.Join(dir, "config.json")
		stPath := filepath.Join(dir, "model.safetensors")
		if fileExists(cfgPath) && fileExists(stPath) {
			return cfgPath, stPath, true
		}
	}
	return "", "", false
}

func findCachedHFModelSafetensorsForTest(t *testing.T, repo string) (string, bool) {
	t.Helper()
	cacheName := "models--" + strings.ReplaceAll(repo, "/", "--")
	patterns := []string{}
	if home, err := os.UserHomeDir(); err == nil {
		patterns = append(patterns, filepath.Join(home, ".cache", "huggingface", "hub", cacheName, "snapshots", "*", "model.safetensors"))
	}
	patterns = append(patterns, filepath.Join("/mnt", "*", "Users", "*", ".cache", "huggingface", "hub", cacheName, "snapshots", "*", "model.safetensors"))

	var matches []string
	for _, pattern := range patterns {
		found, err := filepath.Glob(pattern)
		if err != nil {
			t.Fatalf("glob Hugging Face cache: %v", err)
		}
		matches = append(matches, found...)
	}
	sort.Strings(matches)
	for _, path := range matches {
		if fileExists(path) {
			return path, true
		}
	}
	return "", false
}

func assertCanonicalDenseLayerShapes(t *testing.T, m *Model, cfg Config, layer int) {
	t.Helper()
	H, I := cfg.HiddenSize, cfg.IntermediateSize
	assertTensorShape(t, m, "model.embed_tokens.weight", []int{cfg.VocabSize, H})
	assertTensorShape(t, m, "model.norm.weight", []int{H})
	assertCanonicalAttentionShapes(t, m, cfg, layer)
	assertTensorShape(t, m, layerName(layer, "mlp.gate_proj.weight"), []int{I, H})
	assertTensorShape(t, m, layerName(layer, "mlp.up_proj.weight"), []int{I, H})
	assertTensorShape(t, m, layerName(layer, "mlp.down_proj.weight"), []int{H, I})
}

func assertCanonicalAttentionShapes(t *testing.T, m *Model, cfg Config, layer int) {
	t.Helper()
	H, hd, nH, nKV := cfg.HiddenSize, cfg.HeadDim, cfg.NumHeads, cfg.NumKVHeads
	assertTensorShape(t, m, layerName(layer, "self_attn.q_proj.weight"), []int{nH * hd, H})
	assertTensorShape(t, m, layerName(layer, "self_attn.k_proj.weight"), []int{nKV * hd, H})
	assertTensorShape(t, m, layerName(layer, "self_attn.v_proj.weight"), []int{nKV * hd, H})
	assertTensorShape(t, m, layerName(layer, "self_attn.o_proj.weight"), []int{H, nH * hd})
}

func assertTensorShape(t *testing.T, m *Model, name string, want []int) {
	t.Helper()
	meta, ok := m.manifest[name]
	if !ok {
		t.Fatalf("missing tensor %s", name)
	}
	if !sameShape(meta.Shape, want) {
		t.Fatalf("%s shape = %v, want %v", name, meta.Shape, want)
	}
	if meta.Nbytes != tensorElemCount(want)*4 {
		t.Fatalf("%s nbytes = %d, want %d", name, meta.Nbytes, tensorElemCount(want)*4)
	}
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func tensorElemCount(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

func readFileLoadSafetensorsForTest(path string, cfg Config) (*Model, error) {
	buf, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	man := map[string]tensorMeta{}
	var raw []byte
	off := 0
	if err := appendReadFileSafetensorsForTest(buf, man, &raw, &off); err != nil {
		return nil, err
	}
	return &Model{Cfg: cfg, manifest: man, raw: raw}, nil
}

func appendReadFileSafetensorsForTest(buf []byte, man map[string]tensorMeta, raw *[]byte, off *int) error {
	hdr, dataBase, err := parseSafetensorsHeader(buf)
	if err != nil {
		return err
	}
	for _, name := range safetensorsTensorNames(hdr) {
		if _, ok := man[name]; ok {
			return fmt.Errorf("duplicate tensor %s", name)
		}
		var e stEntry
		if err := json.Unmarshal(hdr[name], &e); err != nil {
			return err
		}
		src, err := safetensorsBufferBytes(buf, dataBase, e)
		if err != nil {
			return err
		}
		fb, err := decodeSafetensorF32(name, e, src)
		if err != nil {
			return err
		}
		man[name] = tensorMeta{Dtype: "f32", Shape: e.Shape, Offset: *off, Nbytes: len(fb)}
		*raw = append(*raw, fb...)
		*off += len(fb)
	}
	return nil
}

func readFileLoadSafetensorsQuantForTest(path string, cfg Config) (*Model, error) {
	buf, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	tied, err := safetensorsTied(buf)
	if err != nil {
		return nil, err
	}
	m := &Model{Cfg: cfg, manifest: map[string]tensorMeta{}, q8w: map[string]*q8Tensor{}}
	var raw []byte
	off := 0
	if err := quantizeBlobInto(buf, m, tied, &raw, &off); err != nil {
		return nil, err
	}
	m.raw = raw
	return m, nil
}

func assertModelRawEqual(t *testing.T, want, got *Model) {
	t.Helper()
	if !reflect.DeepEqual(want.manifest, got.manifest) {
		t.Fatalf("manifest mismatch\nwant=%#v\ngot =%#v", want.manifest, got.manifest)
	}
	if !bytes.Equal(want.raw, got.raw) {
		t.Fatalf("raw bytes differ: want %d bytes, got %d bytes", len(want.raw), len(got.raw))
	}
}

func assertQ8MapsEqual(t *testing.T, want, got map[string]*q8Tensor) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("q8 map length: want %d, got %d", len(want), len(got))
	}
	for name, w := range want {
		g, ok := got[name]
		if !ok {
			t.Fatalf("missing q8 tensor %s", name)
		}
		if w.out != g.out || w.in != g.in || w.nblk != g.nblk {
			t.Fatalf("%s shape: want (%d,%d,%d), got (%d,%d,%d)", name, w.out, w.in, w.nblk, g.out, g.in, g.nblk)
		}
		if !reflect.DeepEqual(w.q, g.q) {
			t.Fatalf("%s q codes differ", name)
		}
		for i := range w.d {
			if math.Float32bits(w.d[i]) != math.Float32bits(g.d[i]) {
				t.Fatalf("%s scale[%d]: want %v, got %v", name, i, w.d[i], g.d[i])
			}
		}
	}
}
