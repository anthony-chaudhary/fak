//go:build darwin && arm64 && cgo

package model

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

func TestMetalQwen35GDNProductionPrefillContinuityAndDecodeHandoff(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)
	prompt := make([]int, 80)
	for i := range prompt {
		prompt[i] = (i*17 + 3) % cfg.VocabSize
	}

	ref := m.NewSession()
	// Keep projection placement identical in both arms. The only delta is the
	// historical host recurrence versus the resident GDN sequence operation.
	ref.Q4K, ref.MetalQ4K = true, true
	ref.PrefillNoLogits(prompt[:64])
	wantLogits := ref.Prefill(prompt[64:])

	baselineBuffers := metalgemm.GDNLiveBufferCount()
	got := m.NewSession()
	got.Q4K, got.MetalQ4K = true, true
	if err := got.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
		t.Fatalf("EnableQwen35MetalGDNPreprojectedSequence: %v", err)
	}
	states := append([]Qwen35GDNAuxState(nil), got.qwen35HAL.sequenceLayers...)
	wantBuffers := baselineBuffers + 2*linearQwen35Layers(cfg)
	if live := metalgemm.GDNLiveBufferCount(); live != wantBuffers {
		t.Fatalf("live native state buffers=%d, want %d", live, wantBuffers)
	}
	got.PrefillNoLogits(prompt[:64])
	if !reflect.DeepEqual(got.qwen35HAL.sequenceLayers, states) {
		t.Fatal("native auxiliary identity changed after first bounded panel")
	}
	gotLogits := got.Prefill(prompt[64:])
	if !reflect.DeepEqual(got.qwen35HAL.sequenceLayers, states) {
		t.Fatal("native auxiliary identity changed across outer prefill chunks")
	}
	executed, err := got.FinalizeQwen35MetalGDNPreprojectedSequence()
	if err != nil || !executed {
		t.Fatalf("finalize = executed %v err %v", executed, err)
	}
	if again, err := got.FinalizeQwen35MetalGDNPreprojectedSequence(); err != nil || again {
		t.Fatalf("second finalize = executed %v err %v, want idempotent no-op", again, err)
	}
	if live := metalgemm.GDNLiveBufferCount(); live != baselineBuffers {
		t.Fatalf("final sync retained %d native buffers, baseline %d", live, baselineBuffers)
	}

	assertQuantLogitsClose(t, "native GDN prefill logits", wantLogits, gotLogits)
	prefillCos, prefillMaxRel := cosineAndMaxRel(wantLogits, gotLogits)
	t.Logf("prefill logits cosine=%.9f maxRel=%.6g greedy=%d", prefillCos, prefillMaxRel, argmax(gotLogits))
	for layer := range ref.Cache.linear.layers {
		for head := range ref.Cache.linear.layers[layer].recurrent {
			cos, maxRel := cosineAndMaxRel(ref.Cache.linear.layers[layer].recurrent[head], got.Cache.linear.layers[layer].recurrent[head])
			t.Logf("linear state layer=%d head=%d cosine=%.9f maxRel=%.6g", layer, head, cos, maxRel)
			if cos < 0.999999 {
				t.Fatalf("linear state layer=%d head=%d cosine=%.9f, want >=0.999999", layer, head, cos)
			}
		}
	}
	next := argmax(wantLogits)
	wantDecode := ref.Step(next)
	gotDecode := got.Step(next)
	assertQuantLogitsClose(t, "first historical decode logits", wantDecode, gotDecode)
	decodeCos, decodeMaxRel := cosineAndMaxRel(wantDecode, gotDecode)
	t.Logf("first decode logits cosine=%.9f maxRel=%.6g greedy=%d", decodeCos, decodeMaxRel, argmax(gotDecode))
	if want, actual := argmax(wantDecode), argmax(gotDecode); actual != want {
		t.Fatalf("first decode greedy token=%d, want %d", actual, want)
	}
	got.Close()
	got.Close()
	if live := metalgemm.GDNLiveBufferCount(); live != baselineBuffers {
		t.Fatalf("idempotent close retained native buffers=%d, baseline %d", live, baselineBuffers)
	}
}

func TestMetalQwen35GDNProductionPreflightDeclinesBeforeMutation(t *testing.T) {
	m := NewSynthetic(qwen35HybridQ4KTestCfg())
	s := m.NewSession()
	s.Q4K = true
	before := s.Cache.Clone()
	if err := s.EnableQwen35MetalGDNPreprojectedSequence(); err == nil {
		t.Fatal("non-Metal session admitted resident GDN sequence")
	}
	assertKVCacheQuantClose(t, "preflight decline", before, s.Cache)
	if s.qwen35HAL != nil {
		t.Fatalf("preflight decline attached state: %#v", s.qwen35HAL)
	}
}
