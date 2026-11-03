package model

import (
	"math"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=1s lane=default
func TestPackedPrefillRaggedStateAndContinuation(t *testing.T) {
	for _, quant := range []bool{false, true} {
		t.Run(map[bool]string{false: "f32", true: "q8"}[quant], func(t *testing.T) {
			cfg := Config{HiddenSize: 64, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 16,
				IntermediateSize: 128, VocabSize: 257, RMSNormEps: 1e-5, RopeTheta: 10000,
				TieWordEmbeddings: true, EOSTokenID: -1}
			m := NewSynthetic(cfg)
			if quant {
				m.Quantize()
			}
			bs := m.NewBatchSession(3)
			bs.SetQuant(quant)
			ref := []*Session{m.NewSession(), m.NewSession(), m.NewSession()}
			for b, s := range ref {
				s.Quant = quant
				defer s.Close()
				defer bs.Seqs[b].Close()
			}
			prefixes := [][]int{nil, {2, 3, 4}, {8}}
			for b, prefix := range prefixes {
				ref[b].Prefill(prefix)
				bs.Seqs[b].Prefill(prefix)
			}
			rounds := [][][]int{{{11, 12, 13, 14, 15, 16, 17}, {21, 22, 23}, {31, 32, 33, 34, 35}}, {{41}, {51, 52, 53, 54}, {61, 62}}}
			for round, prompts := range rounds {
				got, used := bs.PrefillPacked(prompts)
				if !used || len(got) != 3 {
					t.Fatalf("round %d packed used=%v rows=%d", round, used, len(got))
				}
				for b, prompt := range prompts {
					want := ref[b].Prefill(prompt)
					if quant {
						if cosine(got[b], want) < 0.999 || argmax(got[b]) != argmax(want) {
							t.Fatalf("round %d lane %d Q8 logits diverged", round, b)
						}
					} else {
						for j := range want {
							if math.Float32bits(got[b][j]) != math.Float32bits(want[j]) {
								t.Fatalf("f32 logit %d/%d/%d", round, b, j)
							}
						}
					}
					a, c := bs.Seqs[b].Cache, ref[b].Cache
					if !reflect.DeepEqual(a.pos, c.pos) || !reflect.DeepEqual(a.lineage, c.lineage) {
						t.Fatalf("round %d lane %d positions/lineage", round, b)
					}
					for layer := 0; layer < cfg.NumLayers; layer++ {
						for name, pair := range map[string][2][]float32{"K": {a.K[layer], c.K[layer]}, "Kraw": {a.Kraw[layer], c.Kraw[layer]}, "V": {a.V[layer], c.V[layer]}} {
							if len(pair[0]) != len(pair[1]) {
								t.Fatalf("%s length", name)
							}
							if quant {
								if cosine(pair[0], pair[1]) < 0.999 {
									t.Fatalf("Q8 %s state lane%d", name, b)
								}
							} else {
								for j := range pair[0] {
									if math.Float32bits(pair[0][j]) != math.Float32bits(pair[1][j]) {
										t.Fatalf("f32 %s state lane%d index%d", name, b, j)
									}
								}
							}
						}
					}
				}
			}
			for b := range ref {
				got, want := bs.Seqs[b].Step(71+b), ref[b].Step(71+b)
				if cosine(got, want) < 0.999 || argmax(got) != argmax(want) {
					t.Fatalf("continuation lane%d", b)
				}
			}
		})
	}
}

// fak-test:runtime fast est=1s lane=default
func TestPackedPrefillRefusesBeforeCacheEffects(t *testing.T) {
	cfg := Config{HiddenSize: 32, NumLayers: 1, NumHeads: 2, NumKVHeads: 1, HeadDim: 16,
		IntermediateSize: 64, VocabSize: 64, RMSNormEps: 1e-5, RopeTheta: 10000, TieWordEmbeddings: true}
	m := NewSynthetic(cfg)
	for _, kind := range []string{"empty", "budget", "token", "alias", "mode", "q8_cache", "position", "device_only", "attention_observer"} {
		t.Run(kind, func(t *testing.T) {
			bs := m.NewBatchSession(2)
			owners := append([]*Session(nil), bs.Seqs...)
			for _, s := range owners {
				defer s.Close()
			}
			prompts := [][]int{{1, 2}, {3}}
			observed := 0
			defer m.SetAttnObserver(nil)
			switch kind {
			case "device_only":
				bs.Seqs[1].executionPolicy = ExecutionPolicyDeviceOnly
			case "attention_observer":
				m.SetAttnObserver(func(int, int, int, []int, []float32) { observed++ })
			case "empty":
				prompts[1] = nil
			case "budget":
				prompts[0] = make([]int, PackedPrefillMaxTokens)
			case "token":
				prompts[1][0] = cfg.VocabSize
			case "alias":
				bs.Seqs[1] = bs.Seqs[0]
			case "mode":
				bs.Seqs[1].Q4K = true
			case "q8_cache":
				bs.Seqs[1].Cache = NewKVCacheWithPrecision(cfg, KVPrecisionQ8_0)
			case "position":
				bs.Seqs[1].Prefill([]int{4})
				bs.Seqs[1].Cache.pos[0] = 9
			}
			before := []*KVCache{owners[0].Cache.Clone(), owners[1].Cache.Clone()}
			if out, used := bs.PrefillPacked(prompts); used || out != nil {
				t.Fatal("unsupported packed request accepted")
			}
			if observed != 0 {
				t.Fatal("refused packed request invoked observer")
			}
			for b, s := range owners {
				if !reflect.DeepEqual(s.Cache.Clone(), before[b]) {
					t.Fatalf("refusal mutated lane%d", b)
				}
			}
		})
	}
}
