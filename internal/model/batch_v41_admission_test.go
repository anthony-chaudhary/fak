package model

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestV41BatchAdmission(t *testing.T) {
	cfg := Config{
		HiddenSize: 32, NumLayers: 1, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		IntermediateSize: 64, VocabSize: 41, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1,
	}
	// Q8 is a predicate control only; this test executes F32 entrypoints below.
	for _, quant := range []bool{false, true} {
		if !batchDecodeFastPathOK(cfg, quant) || !batchRectFastPathOK(cfg, quant) {
			t.Fatalf("dense control rejected: quant=%v", quant)
		}
		for _, identity := range []string{"deepseek41", "deepseek_v41", "deepseek_v41_text", "metadata"} {
			v41 := cfg
			if identity == "metadata" {
				v41.DeepSeekV41 = &DeepSeekV41Config{}
			} else {
				v41.ModelType = identity
			}
			if batchDecodeFastPathOK(v41, quant) || batchRectFastPathOK(v41, quant) {
				t.Errorf("V4.1 admitted to dense batch panels: identity=%s quant=%v", identity, quant)
			}
		}
	}

	entries := []struct {
		name string
		run  func(*BatchSession)
	}{
		{"prefill", func(bs *BatchSession) { bs.PrefillEach([][]int{{1, 2}, {3, 4}}) }},
		{"decode", func(bs *BatchSession) { bs.StepBatch([]int{1, 2}) }},
		{"active-decode", func(bs *BatchSession) { bs.StepBatchActive([]int{1, 2, 3}, []bool{true, true, false}) }},
		{"prefill-no-logits", func(bs *BatchSession) { bs.PrefillEachNoLogits([][]int{{1, 2}, {3, 4}}) }},
	}
	for _, entry := range entries {
		t.Run(entry.name, func(t *testing.T) {
			// Deliberately label ordinary dense weights as V4.1 after construction.
			// This malformed in-memory layout must not bypass session admission.
			m := NewSynthetic(cfg)
			m.Cfg.ModelType = "deepseek41"
			n := 2
			if entry.name == "active-decode" {
				n = 3 // Two active lanes keep the compacted batch on the panel seam.
			}
			bs := m.NewBatchSession(n)
			for _, s := range bs.Seqs {
				t.Cleanup(s.Close)
				s.SetExecutionPolicy(ExecutionPolicyDeviceOnly)
			}
			err := panicAsError(func() { entry.run(bs) })
			var refusal *BackendForwardOperationError
			if !errors.As(err, &refusal) || refusal.Path != "device-only" || refusal.Forward != ForwardPathKind("deepseek41") {
				t.Fatalf("entry refusal = %v, want typed V4.1 device-only refusal", err)
			}
			if !bs.Seqs[0].BackendSessionClosed() || !bs.Seqs[0].HostFallbackObserved() {
				t.Fatal("entry did not run fail-closed backend validation")
			}
			for lane, s := range bs.Seqs {
				if s.Cache.Len() != 0 {
					t.Fatalf("refused entry advanced lane %d", lane)
				}
			}
		})
	}
}
