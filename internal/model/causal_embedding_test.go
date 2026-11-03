package model

import (
	"context"
	"math"
	"reflect"
	"testing"
)

func causalEmbeddingFixture() *Model {
	return NewSynthetic(Config{ModelType: "llama", HiddenSize: 8, NumLayers: 2, NumHeads: 2, NumKVHeads: 1, HeadDim: 4,
		IntermediateSize: 12, VocabSize: 16, RMSNormEps: 1e-5, RopeTheta: 10000, MaxPositionEmbeddings: 32})
}

// fak-test:runtime fast est=200ms lane=default
func TestCausalEmbeddingFinalNormPoolingAndHeadBypass(t *testing.T) {
	m := causalEmbeddingFixture()
	ids := []int{1, 2, 3}
	full := m.Forward(ids)
	hidden := full.Hidden[len(full.Hidden)-1]
	for _, pooling := range []string{"mean", "last"} {
		got, err := m.CausalEmbedding(context.Background(), ids, pooling)
		if err != nil {
			t.Fatal(err)
		}
		sums := make([]float64, m.Cfg.HiddenSize)
		begin := 0
		if pooling == "last" {
			begin = len(ids) - 1
		}
		for pos := begin; pos < len(ids); pos++ {
			row := m.finalNorm(hidden[pos*m.Cfg.HiddenSize : (pos+1)*m.Cfg.HiddenSize])
			for j, v := range row {
				sums[j] += float64(v)
			}
		}
		if pooling == "mean" {
			for j := range sums {
				sums[j] /= float64(len(ids))
			}
		}
		n := 0.0
		for _, v := range sums {
			n = math.Hypot(n, v)
		}
		for j, v := range got {
			if math.Abs(float64(v)-sums[j]/n) > 1e-6 {
				t.Fatalf("%s coordinate %d: %g", pooling, j, v)
			}
		}
		// This head cannot satisfy a vocabulary projection. Hidden-only execution
		// must remain identical and does not need an LM-head tensor at all.
		m.manifest["lm_head.weight"] = tensorMeta{Dtype: "F32", Shape: []int{1, 1}, Offset: 0, Nbytes: 4}
		again, err := m.CausalEmbedding(context.Background(), ids, pooling)
		if err != nil || !reflect.DeepEqual(got, again) {
			t.Fatalf("LM head consumed: %v", err)
		}
		delete(m.manifest, "lm_head.weight")
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestCausalEmbeddingRefusalAndOwnership(t *testing.T) {
	cases := []func(*Model){
		func(m *Model) { m.Cfg.ModelType = "bert" },
		func(m *Model) { m.Cfg.QKNorm = true },
		func(m *Model) { m.Cfg.HeadDim = 3 },
		func(m *Model) { m.Cfg.EnableResidualHook = true },
		func(m *Model) { m.epRanks = 2 },
		func(m *Model) {
			v := m.manifest["model.norm.weight"]
			v.Offset = len(m.raw)
			m.manifest["model.norm.weight"] = v
		},
		func(m *Model) {
			delete(m.manifest, layerName(0, "self_attn.q_proj.weight"))
			m.q8w = map[string]*q8Tensor{layerName(0, "self_attn.q_proj.weight"): {out: 8, in: 7, nblk: 0}}
		},
	}
	for i, mutate := range cases {
		m := causalEmbeddingFixture()
		mutate(m)
		if _, err := m.CausalEmbedding(context.Background(), []int{1}, "mean"); err == nil {
			t.Fatalf("case %d admitted", i)
		}
	}
	m := causalEmbeddingFixture()
	for _, ids := range [][]int{nil, {-1}, {m.Cfg.VocabSize}, make([]int, 33)} {
		if _, err := m.CausalEmbedding(context.Background(), ids, "mean"); err == nil {
			t.Fatal("invalid sequence admitted")
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := m.CausalEmbedding(ctx, []int{1}, "mean"); err != context.Canceled {
		t.Fatalf("cancel: %v", err)
	}
	if _, err := m.CausalEmbedding(context.Background(), []int{1}, "cls"); err == nil {
		t.Fatal("unsupported pooling")
	}
	if !m.holdModelWeights() {
		t.Fatal("weight hold unavailable")
	}
	m.releaseWeightSession()
	m.CloseWeights()
	if _, err := m.CausalEmbedding(context.Background(), []int{1}, "mean"); err == nil {
		t.Fatal("closed weights accepted")
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestCausalEmbeddingDoesNotMutateInputs(t *testing.T) {
	m := causalEmbeddingFixture()
	ids := []int{1, 2, 3}
	before := append([]byte(nil), m.raw...)
	a, err := m.CausalEmbedding(context.Background(), ids, "mean")
	if err != nil {
		t.Fatal(err)
	}
	b, err := m.CausalEmbedding(context.Background(), ids, "mean")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a, b) || !reflect.DeepEqual(ids, []int{1, 2, 3}) || !reflect.DeepEqual(before, m.raw) {
		t.Fatal("request or weights mutated")
	}
}

// fak-test:runtime fast est=200ms lane=default
func TestCausalEmbeddingLeanQ8OrdinaryResidentRoute(t *testing.T) {
	m := NewSynthetic(Config{ModelType: "llama", HiddenSize: 32, NumLayers: 1, NumHeads: 4, NumKVHeads: 2, HeadDim: 8, IntermediateSize: 64, VocabSize: 16, RMSNormEps: 1e-5, RopeTheta: 10000})
	m.Quantize()
	for name := range m.q8w {
		if name != "model.embed_tokens.weight" {
			delete(m.manifest, name)
		}
	}
	ids := []int{1, 2, 3}
	full := m.Forward(ids)
	want := m.finalNorm(full.Hidden[len(full.Hidden)-1][2*32 : 3*32])
	norm := 0.0
	for _, v := range want {
		norm = math.Hypot(norm, float64(v))
	}
	got, err := m.CausalEmbedding(context.Background(), ids, "last")
	if err != nil {
		t.Fatal(err)
	}
	for i, v := range got {
		if math.Abs(float64(v)-float64(want[i])/norm) > 1e-6 {
			t.Fatalf("resident result differs at %d", i)
		}
	}
	if m.CausalEmbeddingPrecision() != "q8_0-projections-f32-state" {
		t.Fatal("precision declaration missing")
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestCausalEmbeddingRefusesImplicitNormOverride(t *testing.T) {
	for _, suffix := range []string{"self_attn.sinks", "pre_feedforward_layernorm.weight", "pre_feedforward_layernorm.bias", "post_feedforward_layernorm.weight", "post_feedforward_layernorm.bias"} {
		m := causalEmbeddingFixture()
		m.manifest[layerName(0, suffix)] = tensorMeta{Dtype: "F32", Shape: []int{1}, Offset: 0, Nbytes: 4}
		if _, err := m.CausalEmbedding(context.Background(), []int{1}, "mean"); err == nil {
			t.Fatalf("implicit unsupported norm %s admitted", suffix)
		}
	}
}
