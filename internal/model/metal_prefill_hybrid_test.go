//go:build darwin && arm64 && cgo

package model

// metal_prefill_hybrid_test.go — the on-device correctness gate for the Metal hybrid
// (Qwen3.6 Gated-DeltaNet) prefill twin prefillBatchedMetalQwen35Hybrid (#71). Built ONLY under
// `-tags fakmetal`. The twin's entire CPU-side orchestration — both RMSNorms, the conv1d+SiLU
// mixer, the q/k L2-norm, the per-head delta-rule recurrent scan, the gated RMSNorm readout, the
// full-attention RoPE/GQA/output-gate, and every residual — is already proven host-independently
// against the CPU template by TestQwen35HybridViaMMMatchesCPUTemplate (no Mac, no cgo). This file
// adds the ONE residual that test cannot reach: the GPU f16 GEMM numerics the twin substitutes
// for the projection/MLP matmuls. It is the Mac-gated witness named as the last open step in
// experiments/qwen36/metal-hybrid-prefill-status-2026-06-28.md §3 (step 2).
//
// It holds the Metal hybrid prefill (s.Metal -> prefillBatchedMetalQwen35Hybrid) to the proven
// CPU Q8 hybrid prefill (s.Quant -> prefillQwen35HybridQ) on the SAME quantized weights: both
// read the identical Q8 store, so the only divergence is the projection backend — the CPU's
// qgemm8 vs the GPU's dequant-Q8->f16 MatMul — and the two must agree up to GPU f16
// float-accumulation order. That is the exact parity class TestMetalDecodeResidentMatchesCPU
// establishes for the resident decode forward (#67): an f16-dequant GPU GEMM held to the CPU Q8
// reference, logit cosine ~1.0 with the same argmax. A real wiring bug in the twin (wrong weight
// name, wrong per-layer-kind upload set, wrong GEMM stride) diverges O(1) per layer and trips the
// logit cosine / argmax — or the per-full-attention-layer KV cosine — below.

import (
	"math"
	"slices"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// TestPrefillQwen35HybridMetalMatchesCPU prefills the same 16-token prompt through the CPU Q8
// hybrid path and the Metal hybrid twin from a fresh synthetic Qwen3.6-hybrid model, and asserts
// the device prefill produces the same final-token distribution (logit cosine + argmax) and
// advances the KV cache to the same per-layer state within the f16-GEMM drift band.
func TestPrefillQwen35HybridMetalMatchesCPU(t *testing.T) {
	if !metalgemm.MPSAvailable() {
		t.Skip("MPS f16 projection capability unavailable")
	}
	assertPrefillQwen35HybridMetalMatchesCPU(t)
}

// TestPrefillQwen35HybridMetalDeclinesBeforeMutationWithoutMPS pins the #1112
// capability decline independently of the host's MPS setting. A missing MPS
// projection lane must use the existing fak-native Q8 hybrid path before GPU
// weights or prompt state are touched; it never consumes an unwritten output.
func TestPrefillQwen35HybridMetalDeclinesBeforeMutationWithoutMPS(t *testing.T) {
	if !metalgemm.Available() {
		t.Skip("no Metal device available")
	}
	previous := metalHybridMPSAvailable
	metalHybridMPSAvailable = func() bool { return false }
	t.Cleanup(func() { metalHybridMPSAvailable = previous })
	m := assertPrefillQwen35HybridMetalMatchesCPU(t)
	metalHybridMu.Lock()
	_, uploaded := metalHybridWt[m]
	metalHybridMu.Unlock()
	if uploaded {
		t.Fatal("MPS-unavailable decline uploaded Metal hybrid weights")
	}
}

func assertPrefillQwen35HybridMetalMatchesCPU(t *testing.T) *Model {
	t.Helper()
	defer metalgemm.Reset()

	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize() // build the Q8 store the twin uploads (dequantQ8 -> f16) and the CPU path dots
	// 16 tokens: meets qwen35HybridQBatchMinPrompt so both Prefills take the batched hybrid route.
	prompt := []int{3, 7, 11, 5, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61}
	if !q8Qwen35HybridPrefillOK(cfg, len(prompt)) {
		t.Fatal("q8Qwen35HybridPrefillOK declined the synthetic hybrid cfg — neither path would take the hybrid route")
	}

	// Reference: the proven CPU Q8 hybrid prefill (s.Quant -> prefillQwen35HybridQ).
	ref := m.NewSession()
	ref.Quant = true
	want := ref.Prefill(prompt)

	// Device: the Metal hybrid twin (s.Metal -> prefillBatchedMetalQwen35Hybrid).
	got := m.NewSession()
	got.Metal = true
	gotLogits := got.Prefill(prompt)

	// Load-bearing gate: the full-prefill logits — a function of every projection on every layer,
	// the GDN recurrence, the full attention, and the head — match up to f16 accumulation order.
	cos, maxRel := cosineAndMaxRel(want, gotLogits)
	if argmaxF(want) != argmaxF(gotLogits) || cos < 0.999 {
		t.Errorf("metal hybrid prefill logits: cpu argmax=%d gpu argmax=%d cos=%.6f maxRel=%.4g (want same argmax, cos>=0.999)\n  cpu[:6]=%v\n  gpu[:6]=%v",
			argmaxF(want), argmaxF(gotLogits), cos, maxRel, head6(want), head6(gotLogits))
	} else {
		t.Logf("metal hybrid prefill logits: argmax=%d cos=%.6f maxRel=%.4g OK", argmaxF(gotLogits), cos, maxRel)
	}

	// The device prefill must advance the SAME cache shape decode/Evict/Clone consumes.
	if ref.Cache.Len() != got.Cache.Len() {
		t.Fatalf("metal hybrid prefill cache len = %d, want %d", got.Cache.Len(), ref.Cache.Len())
	}
	// Per-layer KV parity within the f16-GEMM band: the full-attention layers populate K/Kraw/V
	// (the linear-attention layers carry the recurrent/conv state, already covered transitively by
	// the logits above), so a self_attn projection-upload bug localizes here even when it partially
	// cancels in the pooled logits.
	for l := 0; l < cfg.NumLayers; l++ {
		if len(ref.Cache.K[l]) == 0 {
			continue // linear-attention layer: no K/Kraw/V store
		}
		if c := cosine(ref.Cache.K[l], got.Cache.K[l]); c < 0.999 {
			t.Errorf("metal hybrid prefill K layer %d cosine=%.6f (want >=0.999)", l, c)
		}
		if c := cosine(ref.Cache.Kraw[l], got.Cache.Kraw[l]); c < 0.999 {
			t.Errorf("metal hybrid prefill Kraw layer %d cosine=%.6f (want >=0.999)", l, c)
		}
		if c := cosine(ref.Cache.V[l], got.Cache.V[l]); c < 0.999 {
			t.Errorf("metal hybrid prefill V layer %d cosine=%.6f (want >=0.999)", l, c)
		}
	}
	return m
}

// TestMetalQwen35P1PublishesTargetHidden is the on-device witness that the raw
// pre-final-norm residual hidden by BOTH whole-sequence Metal graphs — the P32
// Qwen35MetalForwardSequence prefill and the P1 Qwen35MetalDecodeToken decode —
// is read back and published through Session.TargetHiddenAt. The raw residual x
// is the LAST terminal graph result when capture is armed; the normalized hidden
// (LastRMSNorm(x)) stays at outputs[0]. Capture must be observation-only: the
// capture-on P32/P1 logits, encoder counts, and command-buffer shape must be
// bit-identical to a control session, with the sole delta the extra raw-residual
// readback bytes (P32: 32*H*4, P1: H*4). Proven by TestMetalQwen35P1PublishesTargetHidden.
func TestMetalQwen35P1PublishesTargetHidden(t *testing.T) {
	if !metalgemm.Available() || metalgemm.DeviceName() == "" {
		t.Fatal("raw target-hidden publication requires a physical Metal device")
	}
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	cfg := qwen35HybridQ4KTestCfg()
	// QKNorm is the whole-token P1 decode admission precondition (the P1 graph
	// carries per-head Q/K RMSNorm); without it Qwen35MetalDecodeToken declines
	// fail-open and the P1 raw-hidden readback is never encoded.
	cfg.QKNorm = true
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)
	H := cfg.HiddenSize
	prompt := make([]int, 32)
	for i := range prompt {
		prompt[i] = (i*19 + 7) % cfg.VocabSize
	}

	// The first resident GDN preprojected decode in a process advances global
	// Metal GDN state (the one-time state promotion); a primer session absorbs
	// that transition so the two measured arms observe identical device state
	// and their logits are comparable bit-for-bit.
	primer := m.NewSession()
	primer.Q4K, primer.MetalQ4K = true, true
	if err := primer.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
		primer.Close()
		t.Fatalf("primer EnableQwen35MetalGDNPreprojectedSequence: %v", err)
	}
	primer.Prefill(prompt)
	if executed, err := primer.FinalizeQwen35MetalGDNPreprojectedSequence(); err != nil || !executed {
		primer.Close()
		t.Fatalf("primer finalize = executed %v err %v", executed, err)
	}
	primer.Step(7)

	control := m.NewSession()
	control.Q4K, control.MetalQ4K = true, true
	if err := control.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
		control.Close()
		t.Fatalf("control EnableQwen35MetalGDNPreprojectedSequence: %v", err)
	}
	// The forward logits slice is call-owned device memory reused by the next
	// forward on the same session, so snapshot it before the session advances.
	wantP32 := append([]float32(nil), control.Prefill(prompt)...)
	wantP32Receipt := control.Qwen35MetalForwardSequenceReceipt()
	if executed, err := control.FinalizeQwen35MetalGDNPreprojectedSequence(); err != nil || !executed {
		control.Close()
		t.Fatalf("control finalize = executed %v err %v", executed, err)
	}
	wantP1 := append([]float32(nil), control.Step(7)...)
	wantP1Receipt := control.Qwen35MetalForwardSequenceReceipt()
	if len(control.targetHidden) != 0 || len(control.targetHiddenTokens) != 0 {
		control.Close()
		t.Fatalf("control captured %d hidden rows, want 0", len(control.targetHidden))
	}

	captured := m.NewSession()
	captured.Q4K, captured.MetalQ4K = true, true
	captured.captureTargetHidden = true
	if err := captured.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
		captured.Close()
		t.Fatalf("captured EnableQwen35MetalGDNPreprojectedSequence: %v", err)
	}
	gotP32 := append([]float32(nil), captured.Prefill(prompt)...)
	gotP32Receipt := captured.Qwen35MetalForwardSequenceReceipt()
	assertFloat32BitsEqual(t, "capture-on P32 logits", wantP32, gotP32)
	if gotP32Receipt.Tokens != 32 || !gotP32Receipt.Available || !gotP32Receipt.Committed || !gotP32Receipt.CompletedWait ||
		gotP32Receipt.CommandBuffers != 1 || gotP32Receipt.TerminalWaits != 1 || gotP32Receipt.TerminalReadbacks != 1 ||
		gotP32Receipt.IntermediateWaits != 0 || gotP32Receipt.IntermediateReadbacks != 0 {
		captured.Close()
		t.Fatalf("captured P32 receipt=%+v", gotP32Receipt)
	}
	if gotP32Receipt.HostReadbackBytes != wantP32Receipt.HostReadbackBytes+uint64(32*H*4) {
		captured.Close()
		t.Fatalf("captured P32 readback=%d, want plain %d + %d", gotP32Receipt.HostReadbackBytes, wantP32Receipt.HostReadbackBytes, 32*H*4)
	}
	if gotP32Receipt.Encoders != wantP32Receipt.Encoders {
		captured.Close()
		t.Fatalf("captured P32 encoders=%d, want %d", gotP32Receipt.Encoders, wantP32Receipt.Encoders)
	}
	if len(captured.targetHidden) != 32 || len(captured.targetHiddenTokens) != 32 {
		captured.Close()
		t.Fatalf("captured P32 rows=%d tokens=%d, want 32/32", len(captured.targetHidden), len(captured.targetHiddenTokens))
	}
	for pos := 0; pos < 32; pos++ {
		if captured.targetHiddenTokens[pos] != (pos*19+7)%cfg.VocabSize {
			captured.Close()
			t.Fatalf("P32 target token[%d]=%d, want %d", pos, captured.targetHiddenTokens[pos], (pos*19+7)%cfg.VocabSize)
		}
		if len(captured.targetHidden[pos]) != H {
			captured.Close()
			t.Fatalf("P32 target hidden[%d] len=%d, want %d", pos, len(captured.targetHidden[pos]), H)
		}
		for _, value := range captured.targetHidden[pos] {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				captured.Close()
				t.Fatalf("P32 target hidden[%d] has non-finite value", pos)
			}
		}
	}

	if executed, err := captured.FinalizeQwen35MetalGDNPreprojectedSequence(); err != nil || !executed {
		captured.Close()
		t.Fatalf("captured finalize = executed %v err %v", executed, err)
	}
	gotP1 := append([]float32(nil), captured.Step(7)...)
	gotP1Receipt := captured.Qwen35MetalForwardSequenceReceipt()
	assertFloat32BitsEqual(t, "capture-on P1 logits", wantP1, gotP1)
	if gotP1Receipt.HostReadbackBytes != wantP1Receipt.HostReadbackBytes+uint64(H*4) {
		captured.Close()
		t.Fatalf("captured P1 readback=%d, want plain %d + %d", gotP1Receipt.HostReadbackBytes, wantP1Receipt.HostReadbackBytes, H*4)
	}
	if gotP1Receipt.Encoders != wantP1Receipt.Encoders {
		captured.Close()
		t.Fatalf("captured P1 encoders=%d, want %d", gotP1Receipt.Encoders, wantP1Receipt.Encoders)
	}
	if captured.Cache.Len() != 33 || len(captured.targetHidden) != 33 || len(captured.targetHiddenTokens) != 33 {
		captured.Close()
		t.Fatalf("captured cache=%d rows=%d tokens=%d, want 33/33/33", captured.Cache.Len(), len(captured.targetHidden), len(captured.targetHiddenTokens))
	}
	if captured.targetHiddenTokens[32] != 7 {
		captured.Close()
		t.Fatalf("captured token[32]=%d, want 7", captured.targetHiddenTokens[32])
	}

	raw, err := captured.TargetHiddenAt(32)
	if err != nil {
		captured.Close()
		t.Fatalf("TargetHiddenAt(32): %v", err)
	}
	if len(raw) != H {
		captured.Close()
		t.Fatalf("TargetHiddenAt(32) len=%d, want %d", len(raw), H)
	}
	// The published vector is the PRE-final-norm residual, not the normalized
	// hidden the LM head consumed: running the resident head over it must not
	// reproduce the P1 logits bit-for-bit (those came from headResident(norm)).
	if assertFloat32BitsEqualOptional(wantP1, captured.headResident(raw)) {
		captured.Close()
		t.Fatal("TargetHiddenAt(32) reproduced the P1 logits through the head, so it is the normalized hidden, not the raw residual")
	}

	echo, err := captured.TargetHiddenAt(32)
	if err != nil {
		captured.Close()
		t.Fatalf("second TargetHiddenAt(32): %v", err)
	}
	assertFloat32BitsEqual(t, "TargetHiddenAt defensive copy", raw, echo)
	echo[0] = echo[0] + 1
	again, err := captured.TargetHiddenAt(32)
	if err != nil {
		captured.Close()
		t.Fatalf("third TargetHiddenAt(32): %v", err)
	}
	if again[0] == echo[0] {
		captured.Close()
		t.Fatalf("TargetHiddenAt returned a mutable reference: again[0]=%g echo[0]=%g", again[0], echo[0])
	}
	if cloneTargetHidden(captured.targetHidden)[32][0] != raw[0] {
		captured.Close()
		t.Fatal("caller mutation leaked into the stored target hidden")
	}
	control.Close()
	captured.Close()
	primer.Close()
}

// TestQwen35MetalP32TwoP1PersistentDeviceKV is the #13428 physical spine: the
// production P32->P1 route retains one fixed-capacity device KV owner and each
// decode token uses all full-attention device slices without uploading a prefix.
func TestQwen35MetalP32TwoP1PersistentDeviceKV(t *testing.T) {
	if !metalgemm.Available() || metalgemm.DeviceName() == "" {
		t.Skip("persistent decode KV requires a physical Metal device")
	}
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	cfg := qwen35HybridQ4KTestCfg()
	cfg.QKNorm = true
	// Keep the real Qwen3.8 count of sixteen full-attention planes while retaining
	// the synthetic 256-wide fixture's sub-64 MiB allocation envelope.
	cfg.NumLayers = 17
	cfg.LayerTypes = make([]string, cfg.NumLayers)
	cfg.LayerTypes[0] = "linear_attention"
	for i := 1; i < len(cfg.LayerTypes); i++ {
		cfg.LayerTypes[i] = "full_attention"
	}
	cfg.FullAttentionInterval = 1
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)
	prompt := make([]int, 32)
	for i := range prompt {
		prompt[i] = (i*19 + 7) % cfg.VocabSize
	}

	run := func(persistent bool) (*Session, [][]float32, []Qwen35MetalForwardSequenceReceipt) {
		t.Helper()
		if persistent {
			t.Setenv("FAK_QWEN35_PERSISTENT_DECODE_DKV", "1")
		} else {
			t.Setenv("FAK_QWEN35_PERSISTENT_DECODE_DKV", "0")
		}
		s := m.NewSession()
		s.Q4K, s.MetalQ4K = true, true
		s.Cache.Reserve(35)
		if err := s.EnableQwen35MetalGDNPreprojectedSequence(); err != nil {
			t.Fatal(err)
		}
		s.Prefill(prompt)
		if executed, err := s.FinalizeQwen35MetalGDNPreprojectedSequence(); err != nil || !executed {
			t.Fatalf("finalize persistent=%v executed=%v err=%v", persistent, executed, err)
		}
		out := make([][]float32, 2)
		receipts := make([]Qwen35MetalForwardSequenceReceipt, 2)
		for i, id := range []int{7, 11} {
			out[i] = append([]float32(nil), s.Step(id)...)
			r := s.Qwen35MetalForwardSequenceReceipt()
			receipts[i] = r
			if persistent {
				if r.DeviceKVAttentionLayers != 16 || r.DeviceKVPrefixUploadBytes != 0 ||
					r.DeviceKVSuffixReadbackBytes != uint64(16*3*cfg.NumKVHeads*cfg.HeadDim*4) ||
					r.CommandBuffers != 1 || r.TerminalWaits != 1 || r.IntermediateReadbacks != 0 {
					t.Fatalf("persistent P1[%d] receipt=%+v", i, r)
				}
			}
		}
		return s, out, receipts
	}

	control, want, controlReceipts := run(false)
	defer control.Close()
	candidate, got, candidateReceipts := run(true)
	defer candidate.Close()
	for i := range want {
		if cosine(want[i], got[i]) < 0.9999 || argmaxF(want[i]) != argmaxF(got[i]) {
			t.Fatalf("P1[%d] parity cosine=%g argmax=%d/%d", i, cosine(want[i], got[i]), argmaxF(want[i]), argmaxF(got[i]))
		}
		base := 32 + i
		wantSaved := uint64(16 * 2 * base * cfg.NumKVHeads * cfg.HeadDim * 4)
		if controlReceipts[i].HostUploadBytes < candidateReceipts[i].HostUploadBytes ||
			controlReceipts[i].HostUploadBytes-candidateReceipts[i].HostUploadBytes != wantSaved {
			t.Fatalf("P1[%d] host upload control=%d device=%d saved=%d want=%d", i,
				controlReceipts[i].HostUploadBytes, candidateReceipts[i].HostUploadBytes,
				controlReceipts[i].HostUploadBytes-candidateReceipts[i].HostUploadBytes, wantSaved)
		}
	}
	if candidate.Cache.Len() != control.Cache.Len() || !slices.Equal(control.Cache.lineage.ids, candidate.Cache.lineage.ids) {
		t.Fatalf("candidate cache positions=%d lineage=%d control=%d", candidate.Cache.Len(), len(candidate.Cache.lineage.ids), control.Cache.Len())
	}
	for l := 0; l < cfg.NumLayers; l++ {
		if cfg.isLinearAttnLayer(l) {
			continue
		}
		if cosine(control.Cache.Kraw[l], candidate.Cache.Kraw[l]) < 0.9999 ||
			cosine(control.Cache.K[l], candidate.Cache.K[l]) < 0.9999 ||
			cosine(control.Cache.V[l], candidate.Cache.V[l]) < 0.9999 {
			t.Fatalf("device KV host reconcile parity failed at layer %d", l)
		}
	}
	backend := candidate.qwen35HAL.sequenceBackend.(*metalQwen35GDNSequenceBackend)
	backend.mu.Lock()
	capacity, rows := backend.deviceKVCapacity, backend.deviceKVRows
	backend.mu.Unlock()
	if capacity != 35 || rows != 34 {
		t.Fatalf("device KV capacity=%d rows=%d want 35/34", capacity, rows)
	}
	if live := backend.decodeDeviceKV(candidate); live == nil {
		t.Fatal("accepted-error arm did not begin with an active persistent device KV")
	}
	backend.injectForwardPostSubmitFailure = true
	_, _, accepted, err := backend.Qwen35MetalDecodeToken(candidate, 13)
	if !accepted || err == nil {
		t.Fatalf("injected accepted error accepted=%v err=%v", accepted, err)
	}
	backend.mu.Lock()
	live := backend.deviceKV
	backend.mu.Unlock()
	if live != nil {
		t.Fatal("accepted graph error retained uncertain device KV")
	}
	// Reattach one physical owner to the still-live backend so Session.Close's
	// final FreeQwen35GDNAuxState path, rather than error invalidation, owns it.
	teardownKV := metalgemm.NewDeviceKV(16, 1, cfg.NumKVHeads*cfg.HeadDim)
	if teardownKV == nil {
		t.Fatal("teardown DeviceKV allocation declined")
	}
	backend.mu.Lock()
	backend.deviceKV = teardownKV
	backend.deviceKVCache = candidate.Cache
	backend.deviceKVPersistent = true
	backend.mu.Unlock()
	candidate.Close()
	backend.mu.Lock()
	live = backend.deviceKV
	backend.mu.Unlock()
	if live != nil {
		t.Fatal("Session.Close retained final device KV owner")
	}
}

// assertFloat32BitsEqualOptional reports whether two float32 slices are
// bit-identical WITHOUT failing the test; it is the inverse probe the
// raw-hidden witness needs (identity must NOT hold).
func assertFloat32BitsEqualOptional(want, got []float32) bool {
	if len(want) != len(got) {
		return false
	}
	for i := range want {
		if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
			return false
		}
	}
	return true
}
