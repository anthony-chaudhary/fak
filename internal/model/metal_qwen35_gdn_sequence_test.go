//go:build darwin && arm64 && cgo

package model

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

func qwen35P32ExpectedTransferBytes(m *Model, base int) (upload, readback uint64) {
	const tokens = 32
	cfg := m.Cfg
	elements := tokens * cfg.HiddenSize
	fullLayers := 0
	for l := 0; l < cfg.NumLayers; l++ {
		p := func(suffix string) string { return layerName(l, suffix) }
		elements += len(m.tensor(p("input_layernorm.weight"))) + len(m.tensor(p("post_attention_layernorm.weight")))
		if cfg.isLinearAttnLayer(l) {
			elements += len(m.tensor(p("linear_attn.conv1d.weight"))) + len(m.tensor(p("linear_attn.A_log"))) + len(m.tensor(p("linear_attn.dt_bias"))) + len(m.tensor(p("linear_attn.norm.weight")))
			continue
		}
		fullLayers++
		qnorm, knorm := cfg.HeadDim, cfg.HeadDim
		if cfg.QKNorm {
			qnorm = len(m.tensor(p("self_attn.q_norm.weight")))
			knorm = len(m.tensor(p("self_attn.k_norm.weight")))
		}
		rotary := cfg.rotaryDim()
		elements += qnorm + knorm + 2*tokens*(rotary/2) + 2*base*(cfg.NumKVHeads*cfg.HeadDim)
	}
	elements += len(m.tensor("model.norm.weight"))
	upload = uint64(elements) * 4
	readbackElements := cfg.HiddenSize + fullLayers*3*tokens*(cfg.NumKVHeads*cfg.HeadDim)
	return upload, uint64(readbackElements) * 4
}

func qwen35StateIdentityExpectedGDNBytes(cfg Config) uint64 {
	_, nV, kHd, vHd, _, _, convDim := cfg.linearAttnDims()
	perLayer := (cfg.LinearConvKernelDim-1)*convDim + nV*kHd*vHd
	return uint64(linearQwen35Layers(cfg)*perLayer) * 4
}

func TestMetalQwen35StateIdentityControlAndSequenceSelectorIndependent(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)
	prompt := make([]int, 32)
	for i := range prompt {
		prompt[i] = (i*31 + 13) % cfg.VocabSize
	}

	control := m.NewSession()
	control.Q4K, control.MetalQ4K = true, true
	if err := control.EnableQwen35MetalStateIdentityReceipt(prompt); err != nil {
		t.Fatalf("enable selector-off control identity: %v", err)
	}
	control.PrefillNoLogits(prompt)
	if executed, err := control.FinalizeQwen35MetalStateIdentityReceipt(); err != nil || !executed {
		t.Fatalf("finalize selector-off control identity = executed %v err %v", executed, err)
	}
	controlIdentity, ok := control.Qwen35MetalStateIdentityReceipt()
	if !ok {
		t.Fatal("selector-off Metal control omitted opted-in state identity")
	}
	if err := ValidateQwen35MetalStateIdentityReceipt(controlIdentity); err != nil {
		t.Fatalf("selector-off control identity: %v", err)
	}
	if controlIdentity.Authority != Qwen35MetalStateAuthorityControl || controlIdentity.GDNSnapshotOps != 0 || controlIdentity.GDNSeedOps != 0 || controlIdentity.GDNStateD2HBytes != 0 || controlIdentity.GDNStateH2DBytes != 0 {
		t.Fatalf("selector-off control accounting=%+v", controlIdentity)
	}

	candidate := m.NewSession()
	candidate.Q4K, candidate.MetalQ4K = true, true
	if err := candidate.EnableQwen35MetalStateIdentityReceipt(prompt); err != nil {
		t.Fatalf("enable candidate identity: %v", err)
	}
	if err := candidate.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
		t.Fatalf("enable candidate sequence: %v", err)
	}
	candidate.PrefillNoLogits(prompt)
	graphReceipt := candidate.Qwen35MetalForwardSequenceReceipt()
	if graphReceipt.StateIdentity != nil {
		t.Fatal("candidate exposed state identity before the existing snapshot/seed finalizer succeeded")
	}
	if executed, err := candidate.FinalizeQwen35MetalGDNPreprojectedSequence(); err != nil || !executed {
		t.Fatalf("finalize candidate sequence = executed %v err %v", executed, err)
	}
	candidateIdentity, ok := candidate.Qwen35MetalStateIdentityReceipt()
	if !ok {
		t.Fatal("selector-on Metal candidate omitted opted-in state identity")
	}
	if err := ValidateQwen35MetalStateIdentityReceipt(candidateIdentity); err != nil {
		t.Fatalf("selector-on candidate identity: %v", err)
	}
	wantGDNBytes := qwen35StateIdentityExpectedGDNBytes(cfg)
	wantGDNLayers := linearQwen35Layers(cfg)
	if candidateIdentity.Authority != Qwen35MetalStateAuthoritySequence || candidateIdentity.GDNSnapshotOps != wantGDNLayers || candidateIdentity.GDNSeedOps != wantGDNLayers || candidateIdentity.GDNStateD2HBytes != wantGDNBytes || candidateIdentity.GDNStateH2DBytes != wantGDNBytes {
		t.Fatalf("selector-on candidate accounting=%+v, want layers=%d bytes=%d", candidateIdentity, wantGDNLayers, wantGDNBytes)
	}
	if candidateIdentity.OwnerGeneration == controlIdentity.OwnerGeneration {
		t.Fatal("control and candidate sessions reused an opaque owner generation")
	}
	if candidateIdentity.FullAttentionLayers != controlIdentity.FullAttentionLayers || candidateIdentity.GDNLayers != controlIdentity.GDNLayers || candidateIdentity.StateCount != controlIdentity.StateCount {
		t.Fatalf("selector changed identity coverage: control=%+v candidate=%+v", controlIdentity, candidateIdentity)
	}
	// State digests deliberately are not compared across arms. The identity is
	// exact provenance within one arm; the hardware campaign owns parity.
	finalReceipt := candidate.Qwen35MetalForwardSequenceReceipt()
	if finalReceipt.StateIdentity == nil || finalReceipt.StateIdentity.BindingSHA256 != candidateIdentity.BindingSHA256 {
		t.Fatalf("candidate forward receipt did not retain the finalized identity: %+v", finalReceipt)
	}
	if finalReceipt.HostReadbackBytes != graphReceipt.HostReadbackBytes+wantGDNBytes || finalReceipt.HostUploadBytes != graphReceipt.HostUploadBytes+wantGDNBytes {
		t.Fatalf("candidate total transfers=%d/%d, graph=%d/%d state=%d", finalReceipt.HostUploadBytes, finalReceipt.HostReadbackBytes, graphReceipt.HostUploadBytes, graphReceipt.HostReadbackBytes, wantGDNBytes)
	}
	finalReceipt.StateIdentity.States[0].SHA256 = "caller mutation"
	if got := candidate.Qwen35MetalForwardSequenceReceipt(); got.StateIdentity == nil || got.StateIdentity.States[0].SHA256 == "caller mutation" {
		t.Fatal("caller mutation changed stored nested state identity")
	}

	appended := m.NewSession()
	appended.Q4K, appended.MetalQ4K = true, true
	appended.PrefillNoLogits(prompt[:1])
	if err := appended.EnableQwen35MetalStateIdentityReceipt(prompt); err == nil {
		t.Fatal("appended session admitted a fresh-P32-only observation")
	}
	if _, ok := appended.Qwen35MetalStateIdentityReceipt(); ok {
		t.Fatal("appended session exposed state identity after refusal")
	}

	control.Close()
	candidate.Close()
	appended.Close()
	if _, ok := control.Qwen35MetalStateIdentityReceipt(); ok {
		t.Fatal("closed control session retained state identity")
	}
	if _, ok := candidate.Qwen35MetalStateIdentityReceipt(); ok {
		t.Fatal("closed candidate session retained state identity")
	}
}

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
	receipt := got.Qwen35MetalForwardSequenceReceipt()
	if !receipt.Available {
		t.Fatal("backend-nil P32 prefill did not select the model-owned forward sequence")
	}
	if receipt.Path != Qwen35MetalGDNSequenceForwardPath || receipt.Tokens != 32 || receipt.CommandBuffers != 1 || receipt.TerminalWaits != 1 || receipt.TerminalReadbacks != 1 || !receipt.Committed || !receipt.CompletedWait || receipt.Encoders <= 1 || !receipt.TimingAvailable || receipt.GPUMilliseconds <= 0 || receipt.WaitMilliseconds <= 0 {
		t.Fatalf("whole-sequence receipt=%+v", receipt)
	}
	wantUpload, wantReadback := qwen35P32ExpectedTransferBytes(m, 0)
	if receipt.HostUploadBytes != wantUpload || receipt.HostReadbackBytes != wantReadback {
		t.Fatalf("whole-sequence transfer bytes = upload %d readback %d, want %d/%d", receipt.HostUploadBytes, receipt.HostReadbackBytes, wantUpload, wantReadback)
	}
	receipt.Tokens = 0
	if got.Qwen35MetalForwardSequenceReceipt().Tokens != 32 {
		t.Fatal("caller mutation changed the stored whole-sequence receipt")
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
	firstReceipt := got.Qwen35MetalForwardSequenceReceipt()
	if !firstReceipt.Available || firstReceipt.CommandBuffers != 1 || firstReceipt.TerminalWaits != 1 || firstReceipt.TerminalReadbacks != 1 {
		t.Fatalf("first P32 receipt=%+v", firstReceipt)
	}
	firstUpload, wantReadback := qwen35P32ExpectedTransferBytes(m, 0)
	if firstReceipt.HostUploadBytes != firstUpload || firstReceipt.HostReadbackBytes != wantReadback {
		t.Fatalf("first P32 transfer bytes = upload %d readback %d, want %d/%d", firstReceipt.HostUploadBytes, firstReceipt.HostReadbackBytes, firstUpload, wantReadback)
	}
	actual := got.Prefill(prompt[32:])
	secondReceipt := got.Qwen35MetalForwardSequenceReceipt()
	if !secondReceipt.Available || secondReceipt.Tokens != 32 || secondReceipt.CommandBuffers != 1 || secondReceipt.TerminalWaits != 1 || secondReceipt.TerminalReadbacks != 1 || !secondReceipt.Committed || !secondReceipt.CompletedWait {
		t.Fatalf("appended P32 receipt=%+v", secondReceipt)
	}
	secondUpload, secondReadback := qwen35P32ExpectedTransferBytes(m, 32)
	if secondReceipt.HostUploadBytes != secondUpload || secondReceipt.HostReadbackBytes != secondReadback || secondUpload <= firstUpload {
		t.Fatalf("appended P32 transfer bytes = upload %d readback %d, want %d/%d (fresh upload %d)", secondReceipt.HostUploadBytes, secondReceipt.HostReadbackBytes, secondUpload, secondReadback, firstUpload)
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
	receipt := s.Qwen35MetalForwardSequenceReceipt()
	if !receipt.Available || receipt.Path != Qwen35MetalGDNSequenceForwardPath || receipt.Tokens != 32 || !receipt.Committed || !receipt.CompletedWait || receipt.CommandBuffers != 1 || receipt.TerminalWaits != 1 || receipt.TerminalReadbacks != 0 || receipt.Encoders <= 1 || !receipt.TimingAvailable || receipt.GPUMilliseconds <= 0 || receipt.WaitMilliseconds <= 0 {
		t.Fatalf("post-submit receipt=%+v", receipt)
	}
	wantUpload, _ := qwen35P32ExpectedTransferBytes(m, 0)
	if receipt.HostUploadBytes != wantUpload || receipt.HostReadbackBytes != 0 {
		t.Fatalf("post-submit transfer bytes = upload %d readback %d, want %d/0", receipt.HostUploadBytes, receipt.HostReadbackBytes, wantUpload)
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
	if receipt := s.Qwen35MetalForwardSequenceReceipt(); receipt != (Qwen35MetalForwardSequenceReceipt{}) {
		t.Fatalf("unavailable path returned execution evidence: %+v", receipt)
	}
}

func BenchmarkMetalQwen35P32SequenceVsControl(b *testing.B) {
	setQ4KSDOTForTest(false)
	b.Cleanup(func() { setQ4KSDOTForTest(true) })
	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(&testing.T{}, m, cfg)
	prompt := make([]int, 32)
	for i := range prompt {
		prompt[i] = (i*19 + 7) % cfg.VocabSize
	}

	b.Run("control_per_op", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			s := m.NewSession()
			s.Q4K, s.MetalQ4K = true, true
			s.Prefill(prompt)
			s.Close()
		}
	})

	b.Run("candidate_sequence", func(b *testing.B) {
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
