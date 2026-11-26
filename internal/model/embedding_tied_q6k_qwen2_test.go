package model

import (
	"fmt"
	"math/rand"
	"strings"
	"testing"
)

// The tied Q6_K table was proven on a Qwen3.5 hybrid only. These tests re-prove its two
// assumptions on a dense Qwen2 forward (full attention, no linear-attention layers): rows are
// [vocab][hidden/256 superblocks] so a gather equals the dequantized table row for row, and the
// tied head reads the same rows as its output projection, so a host Q4K session over the shared
// table agrees with the legacy f32-gather + Q8-head pair over identical weights.

func qwen2DenseTiedQ6KTestCfg() Config {
	return Config{
		HiddenSize:        256,
		NumLayers:         2,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           64,
		IntermediateSize:  256,
		VocabSize:         512,
		RMSNormEps:        1e-6,
		RopeTheta:         1000000,
		TieWordEmbeddings: true,
		EOSTokenID:        -1,
		HiddenAct:         "silu",
		ModelType:         "qwen2",
	}
}

func tiedQ6KQwen2ModelPair(t *testing.T) (legacy, packed *Model, table []float32) {
	t.Helper()
	cfg := qwen2DenseTiedQ6KTestCfg()
	if cfg.IsQwen35Hybrid() || cfg.IsMoE() {
		t.Fatal("fixture must be a dense non-hybrid Qwen2 config")
	}
	vocab, hidden := cfg.VocabSize, cfg.HiddenSize
	raw := randomTiedQ6KRaw(t, vocab, hidden, 2048)
	table = dequantQ6KTableForTest(raw, vocab, hidden)

	legacy = NewSynthetic(cfg)
	legacy.Quantize()
	copy(legacy.tensor(tiedEmbeddingName), table)
	legacy.q8w[tiedEmbeddingName] = quantizeQ8(table, vocab, hidden)
	legacy.q8head = legacy.q8w[tiedEmbeddingName]

	packed = NewSynthetic(cfg)
	packed.Quantize()
	delete(packed.manifest, tiedEmbeddingName)
	delete(packed.q8w, tiedEmbeddingName)
	packed.q8head = nil
	if err := packed.attachTiedQ6KEmbedding([]int{vocab, hidden}, raw); err != nil {
		t.Fatalf("attachTiedQ6KEmbedding: %v", err)
	}
	return legacy, packed, table
}

func TestTiedQ6KEmbeddingQwen2GatherAndHeadMatchF32(t *testing.T) {
	_, packed, table := tiedQ6KQwen2ModelPair(t)
	cfg := packed.Cfg
	if !packed.TiedQ6KEmbeddingShared() || packed.HasF32(tiedEmbeddingName) || packed.HasQ8(tiedEmbeddingName) {
		t.Fatal("Qwen2 tied Q6_K model must hold one shared packed table and no f32/Q8 copy")
	}
	if got := packed.kqHeadName(); got != tiedEmbeddingName {
		t.Fatalf("kqHeadName = %q, want the tied embedding", got)
	}

	row := make([]float32, cfg.HiddenSize)
	for id := 0; id < cfg.VocabSize; id++ {
		if err := packed.Q2KEmbedding.GatherRow(id, row, cfg.embedScale()); err != nil {
			t.Fatal(err)
		}
		want := table[id*cfg.HiddenSize : (id+1)*cfg.HiddenSize]
		for i := range row {
			if row[i] != want[i] {
				t.Fatalf("gather row %d[%d] = %v, want %v", id, i, row[i], want[i])
			}
		}
	}

	s := packed.NewSession()
	s.Q4K = true
	x := make([]float32, cfg.HiddenSize)
	rng := rand.New(rand.NewSource(7))
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	got := s.headResident(x)
	want := make([]float32, cfg.VocabSize)
	for o := range want {
		var acc float64
		for i, v := range table[o*cfg.HiddenSize : (o+1)*cfg.HiddenSize] {
			acc += float64(v) * float64(x[i])
		}
		want[o] = float32(acc)
	}
	assertCosineAtLeast(t, "Qwen2 tied Q6_K head vs f32 reference", want, got, 0.9999)
	if argmax(got) != argmax(want) {
		t.Fatalf("Qwen2 tied Q6_K head argmax = %d, want %d", argmax(got), argmax(want))
	}
}

func TestTiedQ6KEmbeddingQwen2PrefillDecodeMatchLegacy(t *testing.T) {
	legacy, packed, _ := tiedQ6KQwen2ModelPair(t)
	prompt := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61}

	ref := packed.NewSession()
	ref.Q4K = true
	var hidden []float32
	for _, id := range prompt {
		hidden = ref.tokenHiddenQ(id, ref.Cache.Len())
	}
	want := ref.headResident(hidden)

	got := packed.NewSession()
	got.Q4K = true
	gotLogits := got.Prefill(prompt)
	assertQuantLogitsClose(t, "Qwen2 packed tied prefill vs decode loop", want, gotLogits)

	base := legacy.NewSession()
	base.Q4K = true
	baseLogits := base.Prefill(prompt)
	assertCosineAtLeast(t, "Qwen2 packed tied prefill vs legacy two-copy", baseLogits, gotLogits, 0.999)

	next := argmax(gotLogits)
	stepped := got.Step(next)
	baseStepped := base.Step(next)
	if len(stepped) != packed.Cfg.VocabSize || hasNaN(stepped) {
		t.Fatalf("Qwen2 packed Step logits invalid: len=%d", len(stepped))
	}
	assertCosineAtLeast(t, "Qwen2 packed tied Step vs legacy two-copy", baseStepped, stepped, 0.999)
}

// A dense Qwen2 tied Q6_K model runs host Q4K sessions (with or without the Quant flag) and
// refuses Metal shapes with a typed reason, since the dense Metal routes index the f32 table.
func TestTiedQ6KEmbeddingQwen2SessionShapes(t *testing.T) {
	_, packed, _ := tiedQ6KQwen2ModelPair(t)
	prompt := []int{3, 7, 11, 5}
	for _, tc := range []struct {
		label                string
		quant, metal, metalQ bool
		refused              bool
	}{
		{label: "q4k"},
		{label: "q4k+quant", quant: true},
		{label: "q4k+metalq4k", quant: true, metalQ: true, refused: true},
		{label: "q4k+metal+metalq4k", quant: true, metal: true, metalQ: true, refused: true},
	} {
		s := packed.NewSession()
		s.Q4K, s.Quant, s.Metal, s.MetalQ4K = true, tc.quant, tc.metal, tc.metalQ
		func() {
			defer func() {
				r := recover()
				if tc.refused {
					if r == nil || !strings.Contains(fmt.Sprint(r), "host Q4K session") {
						t.Fatalf("%s: want a typed host-Q4K refusal, got %v", tc.label, r)
					}
					return
				}
				if r != nil {
					t.Fatalf("%s session panicked on Qwen2 tied Q6_K model: %v", tc.label, r)
				}
			}()
			logits := s.Prefill(prompt)
			if len(logits) != packed.Cfg.VocabSize || hasNaN(logits) {
				t.Fatalf("%s prefill logits invalid", tc.label)
			}
			if next := s.Step(argmax(logits)); len(next) != packed.Cfg.VocabSize || hasNaN(next) {
				t.Fatalf("%s step logits invalid", tc.label)
			}
		}()
	}
}
