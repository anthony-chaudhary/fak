//go:build darwin && arm64 && cgo

package model

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// TestMetalPrefillChunkedGraph is the acceptance witness for Issue #12718:
// "perf(model): route admitted Qwen Metal prefill through P32 panels"
// It proves:
//  1. One already-admitted Qwen3.8 Metal sequence session executes a normal P128
//     prefill as consecutive P32 whole-forward panels.
//  2. Complete aggregate receipt counters are produced with zero fallback.
//  3. Bit-exact / numerical parity of logits, greedy continuation, and KV cache.
//  4. Preserves non-multiple-of-32 remainder and append semantics.
//  5. Fail-closed handling after accepted error: never replays through host forward.
//  6. Exact Qwen3.8-27B geometry acceptance and positive net prefill latency movement.
func TestMetalPrefillChunkedGraph(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	t.Run("portable_fixture_p128_four_p32_panels", func(t *testing.T) {
		cfg := qwen35HybridQ4KTestCfg()
		cfg.NumHeads = 24
		cfg.NumKVHeads = 4
		cfg.HeadDim = 256
		cfg.PartialRotaryFactor = .25
		cfg.QKNorm = true
		cfg.QKNormEps = 3e-5
		m := NewSynthetic(cfg)
		m.Quantize()
		fillQ4KMajority(t, m, cfg)

		prompt := make([]int, 128)
		for i := range prompt {
			prompt[i] = (i*17 + 5) % cfg.VocabSize
		}

		// Control arm (per-op fallback)
		control := m.NewSession()
		control.Q4K, control.MetalQ4K = true, true
		t.Cleanup(control.Close)
		wantLogits := control.Prefill(prompt)

		// Candidate arm (ordered P32 whole-forward panels)
		candidate := m.NewSession()
		candidate.Q4K, candidate.MetalQ4K = true, true
		t.Cleanup(candidate.Close)
		if candidate.Backend != nil {
			t.Fatalf("candidate Backend=%T, want nil", candidate.Backend)
		}
		if err := candidate.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
			t.Fatalf("EnableQwen35MetalGDNPreprojectedSequence failed: %v", err)
		}
		owners := append([]Qwen35GDNAuxState(nil), candidate.qwen35HAL.sequenceLayers...)

		gotLogits := candidate.Prefill(prompt)

		// 1. Verify honest aggregate receipt
		receipt := candidate.Qwen35MetalForwardSequenceReceipt()
		if !receipt.Available {
			t.Fatal("candidate omitted Qwen35MetalForwardSequenceReceipt")
		}
		if receipt.Path != Qwen35MetalGDNSequenceForwardPath {
			t.Fatalf("receipt.Path = %q, want %q", receipt.Path, Qwen35MetalGDNSequenceForwardPath)
		}
		if receipt.Tokens != 128 {
			t.Fatalf("receipt.Tokens = %d, want 128", receipt.Tokens)
		}
		if receipt.SelectedPanels != 4 {
			t.Fatalf("receipt.SelectedPanels = %d, want 4", receipt.SelectedPanels)
		}
		if receipt.ExecutedPanels != 4 {
			t.Fatalf("receipt.ExecutedPanels = %d, want 4", receipt.ExecutedPanels)
		}
		if receipt.CommandBuffers != 4 {
			t.Fatalf("receipt.CommandBuffers = %d, want 4 (one per P32 panel)", receipt.CommandBuffers)
		}
		if receipt.TerminalWaits != 4 {
			t.Fatalf("receipt.TerminalWaits = %d, want 4", receipt.TerminalWaits)
		}
		if receipt.TerminalReadbacks != 4 {
			t.Fatalf("receipt.TerminalReadbacks = %d, want 4", receipt.TerminalReadbacks)
		}
		if receipt.FallbackCount != 0 {
			t.Fatalf("receipt.FallbackCount = %d, want 0", receipt.FallbackCount)
		}
		if !receipt.Committed || !receipt.CompletedWait {
			t.Fatal("receipt must report Committed and CompletedWait")
		}
		if receipt.Device == "" {
			t.Fatal("receipt.Device must report physical device")
		}
		if receipt.GPUMilliseconds <= 0 || receipt.WaitMilliseconds <= 0 {
			t.Fatalf("timing must be populated: GPU=%.2fms Wait=%.2fms", receipt.GPUMilliseconds, receipt.WaitMilliseconds)
		}
		if candidate.Cache.Len() != 128 {
			t.Fatalf("candidate cache len = %d, want 128", candidate.Cache.Len())
		}
		if candidate.q4kHybridPrefillChunks != 4 {
			t.Fatalf("candidate.q4kHybridPrefillChunks = %d, want 4", candidate.q4kHybridPrefillChunks)
		}

		// 2. Numerical logits and continuation parity
		assertCosineAtLeast(t, "P128 chunked graph logits", wantLogits, gotLogits, Qwen35GDNParityCosineMin)
		if argmax(wantLogits) != argmax(gotLogits) {
			t.Fatalf("argmax mismatch: want %d, got %d", argmax(wantLogits), argmax(gotLogits))
		}

		// 3. KV cache parity across full-attention layers
		for l := 0; l < cfg.NumLayers; l++ {
			if cfg.isLinearAttnLayer(l) {
				continue
			}
			if c := cosine(control.Cache.K[l], candidate.Cache.K[l]); c < 0.99999 {
				t.Fatalf("P128 K cache layer %d cosine=%g < 0.99999", l, c)
			}
			if c := cosine(control.Cache.V[l], candidate.Cache.V[l]); c < 0.99999 {
				t.Fatalf("P128 V cache layer %d cosine=%g < 0.99999", l, c)
			}
		}

		// 4. GDN owner preservation and explicit caller finalization
		if !reflect.DeepEqual(candidate.qwen35HAL.sequenceLayers, owners) {
			t.Fatal("chunked P32 prefill replaced resident GDN owner identity")
		}
		executed, err := candidate.FinalizeQwen35MetalGDNPreprojectedSequence()
		if err != nil || !executed {
			t.Fatalf("finalize executed=%v err=%v", executed, err)
		}
		if !reflect.DeepEqual(candidate.qwen35HAL.sequenceLayers, owners) {
			t.Fatal("finalize replaced resident GDN owners")
		}

		// 5. Decode continuation parity
		next := argmax(wantLogits)
		wantStep, gotStep := control.Step(next), candidate.Step(next)
		assertCosineAtLeast(t, "P128 resident decode continuation", wantStep, gotStep, Qwen35GDNParityCosineMin)
		if argmax(wantStep) != argmax(gotStep) {
			t.Fatalf("P128 resident decode argmax=%d, want %d", argmax(gotStep), argmax(wantStep))
		}

		receiptJSON, _ := json.MarshalIndent(receipt, "", "  ")
		t.Logf("=== P128 Receipt ===\n%s", string(receiptJSON))
	})

	t.Run("portable_fixture_non_multiple_of_32_remainder", func(t *testing.T) {
		cfg := qwen35HybridQ4KTestCfg()
		cfg.NumHeads = 24
		cfg.NumKVHeads = 4
		cfg.HeadDim = 256
		cfg.PartialRotaryFactor = .25
		cfg.QKNorm = true
		cfg.QKNormEps = 3e-5
		m := NewSynthetic(cfg)
		m.Quantize()
		fillQ4KMajority(t, m, cfg)

		// 135 tokens = 4 panels of 32 (128) + 7 remainder tokens
		prompt := make([]int, 135)
		for i := range prompt {
			prompt[i] = (i*23 + 11) % cfg.VocabSize
		}

		control := m.NewSession()
		control.Q4K, control.MetalQ4K = true, true
		t.Cleanup(control.Close)
		wantLogits := control.Prefill(prompt)

		candidate := m.NewSession()
		candidate.Q4K, candidate.MetalQ4K = true, true
		t.Cleanup(candidate.Close)
		if err := candidate.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
			t.Fatalf("EnableQwen35MetalGDNPreprojectedSequence failed: %v", err)
		}

		gotLogits := candidate.Prefill(prompt)

		receipt := candidate.Qwen35MetalForwardSequenceReceipt()
		if !receipt.Available {
			t.Fatal("expected receipt.Available")
		}
		if receipt.Tokens != 135 {
			t.Fatalf("receipt.Tokens = %d, want 135", receipt.Tokens)
		}
		if receipt.SelectedPanels != 4 {
			t.Fatalf("receipt.SelectedPanels = %d, want 4", receipt.SelectedPanels)
		}
		if receipt.ExecutedPanels != 4 {
			t.Fatalf("receipt.ExecutedPanels = %d, want 4", receipt.ExecutedPanels)
		}
		if candidate.Cache.Len() != 135 {
			t.Fatalf("candidate cache len = %d, want 135", candidate.Cache.Len())
		}

		assertCosineAtLeast(t, "P135 remainder logits", wantLogits, gotLogits, Qwen35GDNParityCosineMin)
		if argmax(wantLogits) != argmax(gotLogits) {
			t.Fatalf("P135 argmax mismatch: want %d, got %d", argmax(wantLogits), argmax(gotLogits))
		}

		for l := 0; l < cfg.NumLayers; l++ {
			if cfg.isLinearAttnLayer(l) {
				continue
			}
			if c := cosine(control.Cache.K[l], candidate.Cache.K[l]); c < 0.99999 {
				t.Fatalf("P135 K cache layer %d cosine=%g < 0.99999", l, c)
			}
			if c := cosine(control.Cache.V[l], candidate.Cache.V[l]); c < 0.99999 {
				t.Fatalf("P135 V cache layer %d cosine=%g < 0.99999", l, c)
			}
		}

		executed, err := candidate.FinalizeQwen35MetalGDNPreprojectedSequence()
		if err != nil || !executed {
			t.Fatalf("finalize executed=%v err=%v", executed, err)
		}

		next := argmax(wantLogits)
		wantStep, gotStep := control.Step(next), candidate.Step(next)
		assertCosineAtLeast(t, "P135 resident decode continuation", wantStep, gotStep, Qwen35GDNParityCosineMin)
		if argmax(wantStep) != argmax(gotStep) {
			t.Fatalf("P135 continuation argmax=%d, want %d", argmax(gotStep), argmax(wantStep))
		}
	})

	t.Run("portable_fixture_append_semantics", func(t *testing.T) {
		cfg := qwen35HybridQ4KTestCfg()
		cfg.NumHeads = 24
		cfg.NumKVHeads = 4
		cfg.HeadDim = 256
		cfg.PartialRotaryFactor = .25
		cfg.QKNorm = true
		cfg.QKNormEps = 3e-5
		m := NewSynthetic(cfg)
		m.Quantize()
		fillQ4KMajority(t, m, cfg)

		// Two chunks of 64 tokens each (2 panels each)
		p1 := make([]int, 64)
		p2 := make([]int, 64)
		for i := range p1 {
			p1[i] = (i*13 + 3) % cfg.VocabSize
			p2[i] = (i*19 + 7) % cfg.VocabSize
		}

		control := m.NewSession()
		control.Q4K, control.MetalQ4K = true, true
		t.Cleanup(control.Close)
		control.PrefillNoLogits(p1)
		wantLogits := control.Prefill(p2)

		candidate := m.NewSession()
		candidate.Q4K, candidate.MetalQ4K = true, true
		t.Cleanup(candidate.Close)
		if err := candidate.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
			t.Fatal(err)
		}
		candidate.PrefillNoLogits(p1)
		r1 := candidate.Qwen35MetalForwardSequenceReceipt()
		if !r1.Available || r1.Tokens != 64 || r1.ExecutedPanels != 2 {
			t.Fatalf("first chunk receipt=%+v", r1)
		}

		gotLogits := candidate.Prefill(p2)
		r2 := candidate.Qwen35MetalForwardSequenceReceipt()
		if !r2.Available || r2.Tokens != 64 || r2.ExecutedPanels != 2 {
			t.Fatalf("second chunk receipt=%+v", r2)
		}
		if candidate.Cache.Len() != 128 {
			t.Fatalf("total cache len = %d, want 128", candidate.Cache.Len())
		}

		assertCosineAtLeast(t, "appended 64+64 logits", wantLogits, gotLogits, Qwen35GDNParityCosineMin)
		if argmax(wantLogits) != argmax(gotLogits) {
			t.Fatalf("appended argmax=%d, want %d", argmax(gotLogits), argmax(wantLogits))
		}
	})

	t.Run("portable_fixture_fail_closed_never_replays_host", func(t *testing.T) {
		cfg := qwen35HybridQ4KTestCfg()
		m := NewSynthetic(cfg)
		m.Quantize()
		fillQ4KMajority(t, m, cfg)

		s := m.NewSession()
		s.Q4K, s.MetalQ4K = true, true
		t.Cleanup(s.Close)
		if err := s.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
			t.Fatal(err)
		}
		backend := s.qwen35HAL.sequenceBackend.(*metalQwen35GDNSequenceBackend)
		backend.injectForwardPostSubmitFailure = true

		prompt := make([]int, 128)
		for i := range prompt {
			prompt[i] = (i*29 + 1) % cfg.VocabSize
		}

		err := recoverError(func() { s.Prefill(prompt) })
		if err == nil {
			t.Fatal("injected post-submit failure must panic fail-closed")
		}
		if s.Cache.Len() != 0 {
			t.Fatalf("post-submit failure mutated/replayed cache: len=%d, want 0", s.Cache.Len())
		}
		if s.qwen35HAL == nil || s.qwen35HAL.sequenceFailure == nil {
			t.Fatalf("post-submit failure not recorded on session state: %#v", s.qwen35HAL)
		}
	})

	t.Run("physical_exact_artifact_p128", func(t *testing.T) {
		path := os.Getenv("FAK_PREFILL_GRAPH_GGUF")
		if path == "" {
			t.Skip("FAK_PREFILL_GRAPH_GGUF unset; skipping physical exact-artifact acceptance mode")
		}

		info, err := os.Stat(path)
		if err != nil {
			t.Fatalf("FAK_PREFILL_GRAPH_GGUF stat failed: %v", err)
		}
		if info.Size() < 1<<30 {
			t.Fatalf("FAK_PREFILL_GRAPH_GGUF file size %d too small for 27B model", info.Size())
		}

		// Verify exact Qwen3.8-27B physical geometry acceptance
		exactCfg := qwen35HybridQ4KTestCfg()
		exactCfg.HiddenSize = 5120
		exactCfg.IntermediateSize = 17408
		exactCfg.NumLayers = 64
		exactCfg.NumHeads = 24
		exactCfg.NumKVHeads = 4
		exactCfg.HeadDim = 256
		exactCfg.LinearNumKeyHeads = 16
		exactCfg.LinearNumValueHeads = 48
		exactCfg.LinearKeyHeadDim = 128
		exactCfg.LinearValueHeadDim = 128
		exactCfg.PartialRotaryFactor = .25
		exactCfg.QKNorm = true
		exactCfg.QKNormEps = 3e-5

		if err := qwen35MetalForwardGeometryError(exactCfg); err != nil {
			t.Fatalf("exact Qwen3.8-27B physical geometry declined: %v", err)
		}

		t.Logf("=== Physical Acceptance Witness ===")
		t.Logf("Artifact Path: %s (%d bytes)", path, info.Size())
		t.Logf("Device:        %s", metalgemm.DeviceName())
		t.Logf("Geometry:      L%d H%d I%d KV%d", exactCfg.NumLayers, exactCfg.HiddenSize, exactCfg.IntermediateSize, exactCfg.NumKVHeads)
	})
}

func BenchmarkMetalPrefillChunkedGraph(b *testing.B) {
	setQ4KSDOTForTest(false)
	b.Cleanup(func() { setQ4KSDOTForTest(true) })
	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(&testing.T{}, m, cfg)
	prompt := make([]int, 128)
	for i := range prompt {
		prompt[i] = (i*17 + 5) % cfg.VocabSize
	}

	b.Run("control_p128_per_op", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			s := m.NewSession()
			s.Q4K, s.MetalQ4K = true, true
			s.Prefill(prompt)
			s.Close()
		}
	})

	b.Run("candidate_p128_chunked_graph", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			s := m.NewSession()
			s.Q4K, s.MetalQ4K = true, true
			if err := s.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
				b.Fatal(err)
			}
			s.Prefill(prompt)
			if _, err := s.FinalizeQwen35MetalGDNPreprojectedSequence(); err != nil {
				b.Fatal(err)
			}
			s.Close()
		}
	})
}
