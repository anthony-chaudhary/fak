package model

import (
	"math"
	"reflect"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestQwen35HybridQ8RestoredSuffixStaysBatched(t *testing.T) {
	t.Setenv("FAK_QWEN35_PREFILL_TOKEN_LOOP", "")
	m := NewSynthetic(qwen35HybridTestCfg())
	m.Quantize()
	prompt := make([]int, 52)
	for i := range prompt {
		prompt[i] = (7*i + 3) % m.Cfg.VocabSize
	}
	newQuant := func() *Session {
		s := m.NewSession()
		s.Quant = true
		return s
	}
	finite := func(t *testing.T, label string, values []float32) {
		t.Helper()
		for i, value := range values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("%s[%d] = %v, want finite", label, i, value)
			}
		}
	}
	checkLogits := func(t *testing.T, label string, want, got []float32) {
		t.Helper()
		if len(want) != m.Cfg.VocabSize || len(got) != m.Cfg.VocabSize {
			t.Fatalf("%s logits lengths = %d/%d, want %d", label, len(want), len(got), m.Cfg.VocabSize)
		}
		finite(t, label+" reference", want)
		finite(t, label, got)
		assertQuantLogitsClose(t, label, want, got)
	}
	checkCache := func(t *testing.T, label string, want, got *KVCache) {
		t.Helper()
		for _, cache := range []*KVCache{want, got} {
			for layer := range cache.kQ8 {
				finite(t, label+" packed K scales", cache.kQ8[layer].scales)
				finite(t, label+" packed V scales", cache.vQ8[layer].scales)
			}
			for layer := range cache.K {
				finite(t, label+" K", cache.K[layer])
				finite(t, label+" Kraw", cache.Kraw[layer])
				finite(t, label+" V", cache.V[layer])
			}
			for _, layer := range cache.linear.layers {
				for _, row := range layer.conv {
					finite(t, label+" conv", row)
				}
				for _, head := range layer.recurrent {
					finite(t, label+" recurrent", head)
				}
			}
		}
		assertKVCacheQuantClose(t, label, want, got)
		assertLinearAttnCacheQuantClose(t, label, want.linear, got.linear)
	}
	whole := newQuant()
	wantLogits := whole.Prefill(prompt)
	prefix := newQuant()
	prefix.PrefillNoLogits(prompt[:16])
	preserved := prefix.Cache.Clone()
	for _, noLogits := range []bool{false, true} {
		name := "Prefill"
		if noLogits {
			name = "PrefillNoLogits"
		}
		t.Run(name, func(t *testing.T) {
			restored := m.SessionFromPrefix(prefix.Cache)
			restored.Quant = true
			restored.PhaseProfiler = NewPhaseProfiler()
			if noLogits {
				restored.PrefillNoLogits(prompt[16:33])
			} else {
				got := restored.Prefill(prompt[16:33])
				checkLogits(t, "first restored suffix", newQuant().Prefill(prompt[:33]), got)
			}
			checkPanel := func(tokens int) {
				t.Helper()
				profile := restored.PhaseProfiler.Snapshot("prefill", tokens, 0, 0)
				// These phases are emitted by the real batched projection callers,
				// not a test dispatch hook. The old cold-only gate emits no panels.
				if got := phaseCalls(profile, "qwen35_linear_in_proj"); got != 3 {
					t.Fatalf("linear projection panels = %d, want 3 (one per recurrent layer)", got)
				}
				if got := phaseCalls(profile, "qwen35_full_qkv_proj"); got != 1 {
					t.Fatalf("full-attention projection panels = %d, want 1", got)
				}
				if got := phaseCalls(profile, "qwen35_linear_step_recurrent"); got != 0 {
					t.Fatalf("per-token recurrent forwards = %d, want 0", got)
				}
			}
			checkPanel(17)
			restored.PhaseProfiler.Reset()
			gotLogits := restored.Prefill(prompt[33:])
			checkPanel(19)
			checkLogits(t, "restored append logits", wantLogits, gotLogits)
			checkCache(t, "restored append", whole.Cache, restored.Cache)
			reference := m.SessionFromPrefix(whole.Cache)
			reference.Quant = true
			checkLogits(t, "decode after append", reference.Step(29), restored.Step(29))
			checkCache(t, "decode after append", reference.Cache, restored.Cache)
			checkCache(t, "source prefix retained", preserved, prefix.Cache)
		})
	}
	t.Run("token_loop_override", func(t *testing.T) {
		t.Setenv("FAK_QWEN35_PREFILL_TOKEN_LOOP", "1")
		restored := m.SessionFromPrefix(prefix.Cache)
		restored.Quant = true
		restored.PhaseProfiler = NewPhaseProfiler()
		got := restored.Prefill(prompt[16:33])
		checkLogits(t, "forced scalar suffix", newQuant().Prefill(prompt[:33]), got)
		profile := restored.PhaseProfiler.Snapshot("prefill", 17, 0, 0)
		if got := phaseCalls(profile, "qwen35_linear_in_proj"); got != 0 {
			t.Fatalf("forced scalar linear projection panels = %d, want 0", got)
		}
		if got := phaseCalls(profile, "qwen35_linear_step_recurrent"); got != 3*17 {
			t.Fatalf("forced scalar recurrent forwards = %d, want %d", got, 3*17)
		}
	})
	t.Run("packed_KV_keeps_scalar_storage", func(t *testing.T) {
		cfg := qwen35HybridTestCfg()
		cfg.NumKVHeads = 4 // Packed rows must be aligned to a complete Q8 block.
		packedModel := NewSynthetic(cfg)
		packedModel.Quantize()
		newPacked := func() *Session {
			s := packedModel.NewSessionWithKVPrecision(KVPrecisionQ8_0)
			s.Quant = true
			return s
		}
		reference := newPacked()
		var want []float32
		for _, id := range prompt[:32] {
			want = reference.Step(id)
		}
		base := newPacked()
		base.PrefillNoLogits(prompt[:15])
		for _, mode := range []string{"cold", "restored", "restored_no_logits"} {
			t.Run(mode, func(t *testing.T) {
				var got *Session
				ids := prompt[:32]
				if mode == "cold" {
					got = newPacked()
				} else {
					got = packedModel.SessionFromPrefix(base.Cache)
					got.Quant = true
					ids = prompt[15:32]
				}
				got.PhaseProfiler = NewPhaseProfiler()
				if mode == "restored_no_logits" {
					got.PrefillNoLogits(ids)
				} else {
					checkLogits(t, "packed prefill", want, got.Prefill(ids))
				}
				profile := got.PhaseProfiler.Snapshot("prefill", len(ids), 0, 0)
				if phaseCalls(profile, "qwen35_linear_in_proj") != 0 || phaseCalls(profile, "qwen35_linear_step_recurrent") != 3*len(ids) {
					t.Fatal("packed KV bypassed the precision-aware token loop")
				}
				if got.Cache.Len() != 32 || got.Cache.kvLen(3) != 32 || len(got.Cache.K[3]) != 0 || len(got.Cache.V[3]) != 0 {
					t.Fatal("packed prefix storage was not preserved and advanced")
				}
				// Only the full-attention layer owns packed rows. A recurrent
				// layer's unused nil slices become empty slices when cloned.
				for layer := 0; layer < 3; layer++ {
					for _, rows := range []kvPackedRow{reference.Cache.kQ8[layer], reference.Cache.vQ8[layer], got.Cache.kQ8[layer], got.Cache.vQ8[layer]} {
						if len(rows.codes) != 0 || len(rows.scales) != 0 {
							t.Fatalf("recurrent layer %d unexpectedly owns packed payload", layer)
						}
					}
				}
				if !reflect.DeepEqual(reference.Cache.kQ8[3], got.Cache.kQ8[3]) || !reflect.DeepEqual(reference.Cache.vQ8[3], got.Cache.vQ8[3]) {
					t.Fatal("packed K/V bytes differ from scalar continuation")
				}
				checkCache(t, "packed recurrent state", reference.Cache, got.Cache)
				decodeRef := packedModel.SessionFromPrefix(reference.Cache)
				decodeRef.Quant = true
				checkLogits(t, "packed decode after append", decodeRef.Step(29), got.Step(29))
			})
		}
	})
	for _, noLogits := range []bool{false, true} {
		name := "target_hidden_prefill"
		if noLogits {
			name = "target_hidden_no_logits"
		}
		t.Run(name, func(t *testing.T) {
			got := m.SessionFromPrefix(prefix.Cache)
			got.Quant, got.captureTargetHidden = true, true
			got.PhaseProfiler = NewPhaseProfiler()
			if noLogits {
				got.PrefillNoLogits(prompt[16:33])
			} else {
				checkLogits(t, "target hidden suffix", newQuant().Prefill(prompt[:33]), got.Prefill(prompt[16:33]))
			}
			for pos := 16; pos < 33; pos++ {
				hidden, err := got.TargetHiddenAt(pos)
				if err != nil || len(hidden) != m.Cfg.HiddenSize {
					t.Fatalf("target hidden at %d: width=%d, err=%v", pos, len(hidden), err)
				}
				finite(t, "target hidden", hidden)
			}
			profile := got.PhaseProfiler.Snapshot("prefill", 17, 0, 0)
			if phaseCalls(profile, "qwen35_linear_in_proj") != 0 || phaseCalls(profile, "qwen35_linear_step_recurrent") != 3*17 {
				t.Fatal("target-hidden capture bypassed the token loop")
			}
		})
	}
}
