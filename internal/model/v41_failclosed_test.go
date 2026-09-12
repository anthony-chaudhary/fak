package model

import (
	"errors"
	"os"
	"path/filepath"
	"testing"
)

// TestV41FailClosed covers the admission seam hardened by issue #12902: every
// loader and forward-selection entrypoint that can receive a V4.1 config must
// return the typed ErrV41NativeUnsupported rather than reaching a generic path
// or folding partial weights. Complement of the metadata tests in
// v41_config_test.go, which pin the config guard itself.

// TestV41FailClosedQuantBuilderRefusesBeforeWeights covers the quant-on-load
// builder the GGUF path (internal/ggufload) reaches via QuantModel/QuantModelQ4K.
// That path never passes through newModel/NewFromF32Tensors, so before this guard
// a GGUF declaring general.architecture="deepseek_v41" could build a *Model and
// reach a generic forward. Every mutator plus Build must refuse, and no tensor may
// be folded into the model.
func TestV41FailClosedQuantBuilderRefusesBeforeWeights(t *testing.T) {
	_, cfg := readDeepSeekV41Config(t)

	b := NewQuantBuilder(cfg, false)
	mutators := map[string]func() error{
		"AddF32Tensor": func() error {
			return b.AddF32Tensor("model.layers.0.self_attn.q_proj.weight", []int{2, 2}, []float32{1, 2, 3, 4})
		},
		"AddResidentQ4K": func() error {
			return b.AddResidentQ4K("model.layers.0.self_attn.q_proj.weight", []int{2, 256}, make([]byte, 2*(256/qkK)*q4kBlockBytes))
		},
		"AddLazyQ4K": func() error {
			return b.AddLazyQ4K("model.layers.0.self_attn.q_proj.weight", []int{2, 256}, LazyQ4KRange{})
		},
		"AddResidentQ6K": func() error {
			return b.AddResidentQ6K("model.layers.0.self_attn.q_proj.weight", []int{2, 256}, make([]byte, 2*(256/qkK)*kindQ6K.blockBytes()))
		},
		"SetQ2KEmbedding": func() error {
			return b.SetQ2KEmbedding(&Q2KEmbedding{})
		},
	}
	for name, mutate := range mutators {
		t.Run(name, func(t *testing.T) {
			if err := mutate(); !errors.Is(err, ErrV41NativeUnsupported) {
				t.Fatalf("error=%v want ErrV41NativeUnsupported", err)
			}
		})
	}
	if _, err := b.Build(); !errors.Is(err, ErrV41NativeUnsupported) {
		t.Fatalf("Build error=%v want ErrV41NativeUnsupported", err)
	}
	if len(b.m.manifest) != 0 || len(b.m.q8w) != 0 || len(b.m.q4kw) != 0 || len(b.m.kqw) != 0 {
		t.Fatalf("refused builder folded weights: manifest=%d q8w=%d q4kw=%d kqw=%d",
			len(b.m.manifest), len(b.m.q8w), len(b.m.q4kw), len(b.m.kqw))
	}

	t.Run("non V4.1 builder unaffected", func(t *testing.T) {
		llama := Config{ModelType: "llama", HiddenSize: 32, NumLayers: 1, NumHeads: 1, NumKVHeads: 1, HeadDim: 32}
		ok := NewQuantBuilder(llama, false)
		weights := make([]float32, 4*32)
		for i := range weights {
			weights[i] = float32(i + 1)
		}
		if err := ok.AddF32Tensor("model.layers.0.self_attn.q_proj.weight", []int{4, 32}, weights); err != nil {
			t.Fatalf("non-V4.1 AddF32Tensor error=%v", err)
		}
		if _, err := ok.Build(); err != nil {
			t.Fatalf("non-V4.1 Build error=%v", err)
		}
	})
}

// TestV41FailClosedCanonicalMTPMutatorsRefuse proves the two canonical-MTP
// mutators, which write directly to the resident stores, still refuse a V4.1
// config. They are otherwise gated only on isQwen35TextFamily(), and a crafted
// V4.1 config that also declares a linear_attention LayerTypes entry satisfies
// that predicate — so without the refusal these would fold weights for V4.1.
func TestV41FailClosedCanonicalMTPMutatorsRefuse(t *testing.T) {
	base := Config{
		ModelType:                 "deepseek_v41",
		DeepSeekV41:               &DeepSeekV41Config{WrapperModelType: "deepseek_v41", TextModelType: "deepseek_v41_text"},
		LayerTypes:                []string{"linear_attention"},
		MTPNumHiddenLayers:        1,
		HiddenSize:                256,
		HeadDim:                   256,
		NumHeads:                  1,
		NumKVHeads:                1,
		MTPUseDedicatedEmbeddings: false,
	}
	if !base.IsDeepSeekV41() {
		t.Fatal("probe config is not recognized as V4.1")
	}
	if !base.isQwen35TextFamily() {
		t.Fatal("probe config does not satisfy the canonical-MTP precondition — test would be vacuous")
	}

	// Exact payload sizes for the declared shapes, so a guard-removed run would
	// SUCCEED and fold weights (proving these tests witness the guard, not a
	// shape/length rejection): Q4_K [256,256] = (65536/256)*144 bytes; Q8_0
	// [256,512] = (131072/32)*34 bytes.
	q4kRaw := make([]byte, (256*256/qkK)*q4kBlockBytes)
	q8Raw := make([]byte, (256*512/kindQ8_0.blockWeights())*kindQ8_0.blockBytes())
	cases := map[string]func() error{
		"AddCanonicalMTPQ4K": func() error {
			return NewQuantBuilder(base, false).AddCanonicalMTPQ4K("mtp.layers.0.self_attn.q_proj.weight", []int{256, 256}, q4kRaw)
		},
		"AddCanonicalMTPFCQ8": func() error {
			return NewQuantBuilder(base, false).AddCanonicalMTPFCQ8("mtp.fc.weight", []int{256, 512}, q8Raw)
		},
	}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			if err := mutate(); !errors.Is(err, ErrV41NativeUnsupported) {
				t.Fatalf("error=%v want ErrV41NativeUnsupported", err)
			}
		})
	}
}

// TestV41FailClosedAWQLoaderRefusesBeforeWeightIO covers LoadAWQ, which reads
// config.json and builds a *Model directly, bypassing newModel. It must refuse
// from the config alone, before opening model.safetensors/pytorch_model.bin.
func TestV41FailClosedAWQLoaderRefusesBeforeWeightIO(t *testing.T) {
	raw, _ := readDeepSeekV41Config(t)
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "config.json"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
	// A sentinel weight file: if the loader refuses from the config, this is never read.
	weightText := []byte("this is not a valid weight container; reaching it is the failure")
	if err := os.WriteFile(filepath.Join(dir, "model.safetensors"), weightText, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadAWQ(dir); !errors.Is(err, ErrV41NativeUnsupported) {
		t.Fatalf("LoadAWQ error=%v want ErrV41NativeUnsupported", err)
	}
}

// TestV41FailClosedClassifyForwardPathRefusesGeneric covers the forward-selection
// seam: a V4.1 config must not classify as the generic ForwardAttnSeqGQA path.
func TestV41FailClosedClassifyForwardPathRefusesGeneric(t *testing.T) {
	_, cfg := readDeepSeekV41Config(t)
	path, err := ClassifyForwardPath(cfg, nil)
	if !errors.Is(err, ErrV41NativeUnsupported) {
		t.Fatalf("ClassifyForwardPath error=%v want ErrV41NativeUnsupported (path=%q)", err, path)
	}

	t.Run("non V4.1 still classifies", func(t *testing.T) {
		llama := Config{ModelType: "llama", HiddenSize: 4096, NumLayers: 32, NumHeads: 32, NumKVHeads: 8, HeadDim: 128}
		got, err := ClassifyForwardPath(llama, nil)
		if err != nil {
			t.Fatalf("llama ClassifyForwardPath error=%v", err)
		}
		if got != ForwardAttnSeqGQA {
			t.Fatalf("llama forward path=%q want %q", got, ForwardAttnSeqGQA)
		}
	})
}

// TestV41FailClosedAllF32AndQuantLoadersReject widens the loader matrix in
// v41_config_test.go to every public f32/quant entrypoint, and proves the weight
// opener is never reached: the config-only directory holds no weights at all.
func TestV41FailClosedAllF32AndQuantLoadersReject(t *testing.T) {
	raw, cfg := readDeepSeekV41Config(t)
	configOnly := t.TempDir()
	if err := os.WriteFile(filepath.Join(configOnly, "config.json"), raw, 0o644); err != nil {
		t.Fatal(err)
	}
	opened := 0
	opener := func(string) (*safetensorsFile, error) {
		opened++
		return nil, errors.New("weight opener reached")
	}
	tests := map[string]func() error{
		"LoadSafetensors": func() error {
			_, err := LoadSafetensors(filepath.Join(configOnly, "absent.safetensors"), cfg)
			return err
		},
		"LoadSafetensorsDir": func() error { _, err := LoadSafetensorsDir(configOnly, cfg); return err },
		"LoadSafetensorsQuant": func() error {
			_, err := loadSafetensorsQuantFile(filepath.Join(configOnly, "absent.safetensors"), cfg, opener)
			return err
		},
		"LoadSafetensorsQuantDir":       func() error { _, err := loadSafetensorsQuantDir(configOnly, cfg, opener); return err },
		"LoadSafetensorsQuantConfigDir": func() error { _, err := LoadSafetensorsQuantConfigDir(configOnly); return err },
		"Load":                          func() error { _, err := Load(configOnly); return err },
		"NewFromF32Tensors":             func() error { _, err := NewFromF32Tensors(cfg, nil); return err },
	}
	for name, load := range tests {
		t.Run(name, func(t *testing.T) {
			if err := load(); !errors.Is(err, ErrV41NativeUnsupported) {
				t.Fatalf("error=%v want ErrV41NativeUnsupported", err)
			}
		})
	}
	if opened != 0 {
		t.Fatalf("opened %d weight files before refusal", opened)
	}
}
