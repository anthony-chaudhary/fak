//go:build darwin && arm64 && cgo

package model

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

func TestMetalQwen35BackendNilP32WholeSequenceSingleFenceAndDecodeOwner(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
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
	prompt := make([]int, 32)
	for i := range prompt {
		prompt[i] = (i*19 + 7) % cfg.VocabSize
	}

	ref := m.NewSession()
	ref.Q4K, ref.MetalQ4K = true, true
	want := ref.Prefill(prompt)

	got := m.NewSession()
	got.Q4K, got.MetalQ4K = true, true
	if got.Backend != nil {
		t.Fatalf("product Metal session Backend=%T, want nil", got.Backend)
	}
	if err := got.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
		t.Fatal(err)
	}
	owners := append([]Qwen35GDNAuxState(nil), got.qwen35HAL.sequenceLayers...)
	actual := got.Prefill(prompt)
	receipt, ok := got.qwen35MetalForwardSequenceReceipt()
	if !ok {
		t.Fatal("backend-nil P32 prefill did not select the model-owned forward sequence")
	}
	if receipt.Tokens != 32 || receipt.CommandBuffers != 1 || receipt.TerminalWaits != 1 || receipt.TerminalReadbacks != 1 || !receipt.Committed || !receipt.CompletedWait || receipt.Encoders <= 1 {
		t.Fatalf("whole-sequence receipt=%+v", receipt)
	}
	if !reflect.DeepEqual(got.qwen35HAL.sequenceLayers, owners) {
		t.Fatal("whole-sequence prefill replaced resident GDN owner identity")
	}
	assertCosineAtLeast(t, "P32 whole-sequence logits", want, actual, Qwen35GDNParityCosineMin)
	if argmax(want) != argmax(actual) {
		t.Fatalf("P32 whole-sequence argmax=%d, want %d", argmax(actual), argmax(want))
	}
	for l := 0; l < cfg.NumLayers; l++ {
		if cfg.isLinearAttnLayer(l) {
			continue
		}
		if c := cosine(ref.Cache.K[l], got.Cache.K[l]); c < 0.99999 {
			t.Fatalf("P32 whole-sequence K layer %d cosine=%g", l, c)
		}
		if c := cosine(ref.Cache.V[l], got.Cache.V[l]); c < 0.99999 {
			t.Fatalf("P32 whole-sequence V layer %d cosine=%g", l, c)
		}
	}
	executed, err := got.FinalizeQwen35MetalGDNPreprojectedSequence()
	if err != nil || !executed {
		t.Fatalf("finalize executed=%v err=%v", executed, err)
	}
	if !reflect.DeepEqual(got.qwen35HAL.sequenceLayers, owners) {
		t.Fatal("finalize replaced the P32 forward's resident GDN owners")
	}
	if path, selected := got.Qwen35GDNDecodePath(); !selected || path != Qwen35MetalGDNDecodeForwardPath {
		t.Fatalf("decode path=(%q,%v), want resident Metal", path, selected)
	}
	next := argmax(want)
	wantNext, gotNext := ref.Step(next), got.Step(next)
	assertCosineAtLeast(t, "P32 resident decode continuation", wantNext, gotNext, Qwen35GDNParityCosineMin)
	if argmax(wantNext) != argmax(gotNext) {
		t.Fatalf("P32 resident decode argmax=%d, want %d", argmax(gotNext), argmax(wantNext))
	}
	got.Close()
}

func TestMetalQwen35P32ExactArtifactGeometryAccepted(t *testing.T) {
	cfg := qwen35HybridQ4KTestCfg()
	cfg.HiddenSize = 5120
	cfg.IntermediateSize = 17408
	cfg.NumHeads = 24
	cfg.NumKVHeads = 4
	cfg.HeadDim = 256
	cfg.LinearNumKeyHeads = 16
	cfg.LinearNumValueHeads = 48
	cfg.LinearKeyHeadDim = 128
	cfg.LinearValueHeadDim = 128
	if err := qwen35MetalForwardGeometryError(cfg); err != nil {
		t.Fatalf("exact Qwen3.8-27B geometry declined: %v", err)
	}
	cfg.HeadDim = 258
	if err := qwen35MetalForwardGeometryError(cfg); err == nil {
		t.Fatal("head_dim above the native 256-lane boundary was admitted")
	}
}

func TestMetalQwen35BackendNilP32AppendUsesResidentPrefixAndOwners(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)
	prompt := make([]int, 64)
	for i := range prompt {
		prompt[i] = (i*29 + 11) % cfg.VocabSize
	}

	ref := m.NewSession()
	ref.Q4K, ref.MetalQ4K = true, true
	ref.PrefillNoLogits(prompt[:32])
	want := ref.Prefill(prompt[32:])

	got := m.NewSession()
	got.Q4K, got.MetalQ4K = true, true
	if got.Backend != nil {
		t.Fatalf("product Metal session Backend=%T, want nil", got.Backend)
	}
	if err := got.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
		t.Fatal(err)
	}
	owners := append([]Qwen35GDNAuxState(nil), got.qwen35HAL.sequenceLayers...)
	got.PrefillNoLogits(prompt[:32])
	firstReceipt, ok := got.qwen35MetalForwardSequenceReceipt()
	if !ok || firstReceipt.CommandBuffers != 1 || firstReceipt.TerminalWaits != 1 || firstReceipt.TerminalReadbacks != 1 {
		t.Fatalf("first P32 receipt=%+v ok=%v", firstReceipt, ok)
	}
	actual := got.Prefill(prompt[32:])
	secondReceipt, ok := got.qwen35MetalForwardSequenceReceipt()
	if !ok || secondReceipt.Tokens != 32 || secondReceipt.CommandBuffers != 1 || secondReceipt.TerminalWaits != 1 || secondReceipt.TerminalReadbacks != 1 || !secondReceipt.Committed || !secondReceipt.CompletedWait {
		t.Fatalf("appended P32 receipt=%+v ok=%v", secondReceipt, ok)
	}
	if got.Cache.Len() != 64 || got.q4kHybridPrefillChunks != 2 || got.q4kHybridPrefillLastBase != 32 {
		t.Fatalf("appended state cache=%d chunks=%d base=%d", got.Cache.Len(), got.q4kHybridPrefillChunks, got.q4kHybridPrefillLastBase)
	}
	if !reflect.DeepEqual(got.qwen35HAL.sequenceLayers, owners) {
		t.Fatal("appended P32 prefill replaced resident GDN owner identity")
	}
	assertCosineAtLeast(t, "appended P32 logits", want, actual, Qwen35GDNParityCosineMin)
	if argmax(want) != argmax(actual) {
		t.Fatalf("appended P32 argmax=%d, want %d", argmax(actual), argmax(want))
	}
	for l := 0; l < cfg.NumLayers; l++ {
		if cfg.isLinearAttnLayer(l) {
			continue
		}
		if c := cosine(ref.Cache.K[l], got.Cache.K[l]); c < 0.99999 {
			t.Fatalf("appended P32 K layer %d cosine=%g", l, c)
		}
		if c := cosine(ref.Cache.V[l], got.Cache.V[l]); c < 0.99999 {
			t.Fatalf("appended P32 V layer %d cosine=%g", l, c)
		}
	}
	executed, err := got.FinalizeQwen35MetalGDNPreprojectedSequence()
	if err != nil || !executed {
		t.Fatalf("finalize executed=%v err=%v", executed, err)
	}
	if !reflect.DeepEqual(got.qwen35HAL.sequenceLayers, owners) {
		t.Fatal("finalize replaced appended P32 resident GDN owners")
	}
	next := argmax(want)
	wantNext, gotNext := ref.Step(next), got.Step(next)
	assertCosineAtLeast(t, "appended P32 resident decode", wantNext, gotNext, Qwen35GDNParityCosineMin)
	if argmax(wantNext) != argmax(gotNext) {
		t.Fatalf("appended P32 decode argmax=%d, want %d", argmax(gotNext), argmax(wantNext))
	}
	got.Close()
}

func TestMetalQwen35BackendNilP32PostSubmitFailureIsTerminal(t *testing.T) {
	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)
	s := m.NewSession()
	s.Q4K, s.MetalQ4K = true, true
	baseline := metalgemm.GDNLiveBufferCount()
	if err := s.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
		t.Fatal(err)
	}
	backend := s.qwen35HAL.sequenceBackend.(*metalQwen35GDNSequenceBackend)
	backend.injectForwardPostSubmitFailure = true
	prompt := make([]int, 32)
	for i := range prompt {
		prompt[i] = (i*23 + 5) % cfg.VocabSize
	}
	err := recoverError(func() { s.Prefill(prompt) })
	if err == nil {
		t.Fatal("injected post-submit failure returned without fail-closed panic")
	}
	receipt, ok := s.qwen35MetalForwardSequenceReceipt()
	if !ok || !receipt.Committed || !receipt.CompletedWait || receipt.CommandBuffers != 1 || receipt.TerminalWaits != 1 || receipt.TerminalReadbacks != 0 || receipt.Encoders <= 1 {
		t.Fatalf("post-submit receipt=%+v ok=%v", receipt, ok)
	}
	if s.Cache.Len() != 0 || s.q4kHybridPrefillChunks != 0 {
		t.Fatalf("post-submit failure replayed/mutated host path: cache=%d chunks=%d", s.Cache.Len(), s.q4kHybridPrefillChunks)
	}
	if s.qwen35HAL == nil || s.qwen35HAL.sequenceFailure == nil || s.qwen35HAL.decodeAccepted {
		t.Fatalf("post-submit failure state=%#v", s.qwen35HAL)
	}
	if live := metalgemm.GDNLiveBufferCount(); live != baseline {
		t.Fatalf("post-submit failure retained %d GDN buffers, baseline %d", live, baseline)
	}
}

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
	if live := metalgemm.GDNLiveBufferCount(); live != wantBuffers {
		t.Fatalf("resident decode owns %d native buffers after finalize, want %d", live, wantBuffers)
	}
	if !reflect.DeepEqual(got.qwen35HAL.sequenceLayers, states) {
		t.Fatal("finalize replaced the prefill GDN owners")
	}
	if path, selected := got.Qwen35GDNDecodePath(); !selected || path != Qwen35MetalGDNDecodeForwardPath {
		t.Fatalf("decode path=(%q,%v), want resident Metal", path, selected)
	}

	assertQuantLogitsClose(t, "native GDN prefill logits", wantLogits, gotLogits)
	prefillCos, prefillMaxRel := cosineAndMaxRel(wantLogits, gotLogits)
	t.Logf("prefill logits cosine=%.9f maxRel=%.6g greedy=%d", prefillCos, prefillMaxRel, argmax(gotLogits))
	next := argmax(wantLogits)
	wantDecode := ref.Step(next)
	gotDecode := got.Step(next)
	assertQuantLogitsClose(t, "first resident decode logits", wantDecode, gotDecode)
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
