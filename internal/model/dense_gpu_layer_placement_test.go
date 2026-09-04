package model

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func newTestDense4LayerConfig() Config {
	return Config{
		HiddenSize:        16,
		NumLayers:         4,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           4,
		IntermediateSize:  32,
		VocabSize:         64,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        63,
	}
}

// TestDenseGPULayerPlacementParityWithHost verifies output parity between
// monolithic full-host execution and split [0, DenseGPULayers) GPU + [DenseGPULayers, NumLayers) CPU execution.
func TestDenseGPULayerPlacementParityWithHost(t *testing.T) {
	cfg := newTestDense4LayerConfig()
	m := NewSynthetic(cfg)
	be := compute.Pick("cpu-ref")
	if !compute.RequireReference(be) {
		t.Fatal("cpu-ref must be a reference backend")
	}

	prompt := []int{3, 7, 11, 19, 23}

	hostSession := m.NewSession()
	hostPrefill := hostSession.Prefill(prompt)

	splitSession := m.NewBackendSession(be)
	splitSession.DenseGPULayers = 2
	splitPrefill := splitSession.Prefill(prompt)

	assertSameF32(t, "prefill parity", hostPrefill, splitPrefill)

	if splitSession.Cache.Len() != len(prompt) {
		t.Fatalf("split host cache len = %d, want %d", splitSession.Cache.Len(), len(prompt))
	}
	if splitSession.halKV.Len() != len(prompt) {
		t.Fatalf("split device KV len = %d, want %d", splitSession.halKV.Len(), len(prompt))
	}

	decodeTokens := []int{5, 13, 21}
	for i, id := range decodeTokens {
		hostStep := hostSession.Step(id)
		splitStep := splitSession.Step(id)
		assertSameF32(t, fmt.Sprintf("step %d (id=%d)", i, id), hostStep, splitStep)

		if splitSession.Cache.Len() != len(prompt)+i+1 {
			t.Fatalf("split host cache length drifted at step %d: got %d want %d", i, splitSession.Cache.Len(), len(prompt)+i+1)
		}
		if splitSession.halKV.Len() != splitSession.Cache.Len() {
			t.Fatalf("split device KV length (%d) != host cache length (%d) at step %d", splitSession.halKV.Len(), splitSession.Cache.Len(), i)
		}
	}

	// Verify GPULayers alias produces identical output.
	aliasSession := m.NewBackendSession(be)
	aliasSession.GPULayers = 2
	aliasPrefill := aliasSession.Prefill(prompt)
	assertSameF32(t, "alias prefill parity", hostPrefill, aliasPrefill)

	splitSessionForAlias := m.NewBackendSession(be)
	splitSessionForAlias.DenseGPULayers = 2
	_ = splitSessionForAlias.Prefill(prompt)

	for i, id := range decodeTokens {
		aliasStep := aliasSession.Step(id)
		splitStep := splitSessionForAlias.Step(id)
		assertSameF32(t, fmt.Sprintf("alias step %d", i), splitStep, aliasStep)
	}

	// Verify Generate greedy decode parity.
	hostGen := m.NewSession().Generate(prompt, 6)
	splitGenSession := m.NewBackendSession(be)
	splitGenSession.DenseGPULayers = 2
	splitGen := splitGenSession.Generate(prompt, 6)
	if !reflect.DeepEqual(hostGen, splitGen) {
		t.Fatalf("Generate parity mismatch: host=%v split=%v", hostGen, splitGen)
	}
}

// handoffRecordingBackend intercepts Backend operations to witness explicit activation handoff
// and assert that device execution does not touch layers at or past the split boundary.
type handoffRecordingBackend struct {
	compute.Backend
	attentionLayers []int
	readActivations []int // lengths of tensors read
	freeActivations []int // lengths of tensors freed
	hiddenSize      int
}

func (b *handoffRecordingBackend) Attention(q compute.Tensor, kv compute.KVStore, layer int, causal bool, grp int, scale float32) compute.Tensor {
	b.attentionLayers = append(b.attentionLayers, layer)
	return b.Backend.Attention(q, kv, layer, causal, grp, scale)
}

func (b *handoffRecordingBackend) Read(t compute.Tensor) []float32 {
	bufLen := t.Numel()
	if bufLen == b.hiddenSize {
		b.readActivations = append(b.readActivations, bufLen)
	}
	return b.Backend.Read(t)
}

func (b *handoffRecordingBackend) Free(t compute.Tensor) {
	bufLen := t.Numel()
	if bufLen == b.hiddenSize {
		b.freeActivations = append(b.freeActivations, bufLen)
	}
	b.Backend.Free(t)
}

// TestDenseGPULayerExplicitActivationHandoff verifies that at the layer boundary,
// the activation tensor x is explicitly read from device to host, freed on device,
// and device attention never executes layers >= DenseGPULayers.
func TestDenseGPULayerExplicitActivationHandoff(t *testing.T) {
	cfg := newTestDense4LayerConfig()
	m := NewSynthetic(cfg)
	baseBackend := compute.Pick("cpu-ref")

	rec := &handoffRecordingBackend{
		Backend:    baseBackend,
		hiddenSize: cfg.HiddenSize,
	}

	s := m.NewBackendSession(rec)
	s.DenseGPULayers = 2

	prompt := []int{3, 7, 11}
	_ = s.Prefill(prompt)

	// In prefill of 3 tokens, each token ran layers 0 and 1 on device.
	// For each token, after layer 1, x was read back to host and freed on device.
	if len(rec.readActivations) < len(prompt) {
		t.Fatalf("expected at least %d activation reads at boundary, got %d", len(prompt), len(rec.readActivations))
	}
	if len(rec.freeActivations) < len(prompt) {
		t.Fatalf("expected at least %d activation frees at boundary, got %d", len(prompt), len(rec.freeActivations))
	}

	// Verify device Attention was only ever called for layers < DenseGPULayers (0 and 1).
	for _, l := range rec.attentionLayers {
		if l >= 2 {
			t.Fatalf("device executed layer %d which is >= DenseGPULayers 2", l)
		}
	}

	// Reset counters and verify decode step handoff.
	rec.attentionLayers = nil
	rec.readActivations = nil
	rec.freeActivations = nil

	_ = s.Step(5)

	if len(rec.readActivations) != 1 {
		t.Fatalf("decode step: expected 1 activation read at boundary, got %d", len(rec.readActivations))
	}
	if len(rec.freeActivations) != 1 {
		t.Fatalf("decode step: expected 1 activation free at boundary, got %d", len(rec.freeActivations))
	}
	for _, l := range rec.attentionLayers {
		if l >= 2 {
			t.Fatalf("decode step: device executed layer %d which is >= DenseGPULayers 2", l)
		}
	}
}

// TestDenseGPULayersFailClosed verifies that out-of-bounds layer counts, conflicts,
// and missing backends fail closed (panic).
func TestDenseGPULayersFailClosed(t *testing.T) {
	cfg := newTestDense4LayerConfig()
	m := NewSynthetic(cfg)
	be := compute.Pick("cpu-ref")

	assertPanics := func(name string, fn func()) {
		t.Helper()
		defer func() {
			if r := recover(); r == nil {
				t.Fatalf("%s expected panic, got none", name)
			}
		}()
		fn()
	}

	assertPanics("negative DenseGPULayers", func() {
		s := m.NewBackendSession(be)
		s.DenseGPULayers = -1
		s.Prefill([]int{1, 2})
	})

	assertPanics("DenseGPULayers exceeds NumLayers", func() {
		s := m.NewBackendSession(be)
		s.DenseGPULayers = cfg.NumLayers + 1
		s.Prefill([]int{1, 2})
	})

	assertPanics("negative GPULayers", func() {
		s := m.NewBackendSession(be)
		s.GPULayers = -3
		s.Step(1)
	})

	assertPanics("GPULayers exceeds NumLayers", func() {
		s := m.NewBackendSession(be)
		s.GPULayers = cfg.NumLayers + 5
		s.Step(1)
	})

	assertPanics("conflicting DenseGPULayers and GPULayers", func() {
		s := m.NewBackendSession(be)
		s.DenseGPULayers = 2
		s.GPULayers = 3
		s.Prefill([]int{1, 2})
	})

	assertPanics("DenseGPULayers with nil Backend", func() {
		s := m.NewSession()
		s.DenseGPULayers = 2
		s.Prefill([]int{1, 2})
	})

	// Non-panicking valid boundaries:
	// DenseGPULayers == 0 (all device)
	s0 := m.NewBackendSession(be)
	s0.DenseGPULayers = 0
	_ = s0.Prefill([]int{1, 2})

	// DenseGPULayers == NumLayers (all device)
	sFull := m.NewBackendSession(be)
	sFull.DenseGPULayers = cfg.NumLayers
	_ = sFull.Prefill([]int{1, 2})
}

type hybridSplitQwenBackend struct {
	*recordingQwen35Backend
}

func (b *hybridSplitQwenBackend) Qwen35GDNDecode(
	normalizedInput,
	inProjQKV, inProjZ, inProjB, inProjA,
	conv1D, aLog, dtBias, norm, outProj,
	convState, recurrentState compute.Tensor,
	numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
	rmsNormEpsilon float32,
) (output, nextConvState, nextRecurrentState compute.Tensor, err error) {
	site := b.tensorSites[inProjQKV.Buf()]
	layer := 0
	for l := 0; l < b.model.Cfg.NumLayers; l++ {
		if strings.Contains(site, layerName(l, "linear_attn.in_proj_qkv.weight")) {
			layer = l
			break
		}
	}
	xn := append([]float32(nil), b.Backend.Read(normalizedInput)...)
	out := b.reference.linearAttnStep(layer, xn, residentKernel{b.model})
	return compute.NewF32(b.Backend, []int{b.model.Cfg.HiddenSize}, out), convState, recurrentState, nil
}

// TestDenseGPULayerPlacementQwen35HybridParity verifies parity on a Qwen3.5 hybrid architecture
// having both linear attention (GDN) and full attention layers.
func TestDenseGPULayerPlacementQwen35HybridParity(t *testing.T) {
	cfg := Config{
		ModelType:           "qwen3_5",
		HiddenSize:          16,
		NumLayers:           4,
		NumHeads:            4,
		NumKVHeads:          2,
		HeadDim:             4,
		IntermediateSize:    32,
		VocabSize:           64,
		RMSNormEps:          1e-5,
		RopeTheta:           10000,
		TieWordEmbeddings:   true,
		LayerTypes:          []string{"linear_attention", "full_attention", "linear_attention", "full_attention"},
		LinearConvKernelDim: 3,
		LinearKeyHeadDim:    4,
		LinearNumKeyHeads:   2,
		LinearValueHeadDim:  4,
		LinearNumValueHeads: 4,
		AttnOutputGate:      true,
	}
	m := NewSynthetic(cfg)
	be := &hybridSplitQwenBackend{recordingQwen35Backend: newRecordingQwen35Backend(m)}

	prompt := []int{2, 5, 8, 14}
	hostS := m.NewSession()
	hostPrefill := hostS.Prefill(prompt)

	splitS, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatalf("NewBackendSessionChecked: %v", err)
	}
	splitS.DenseGPULayers = 2
	splitPrefill := splitS.Prefill(prompt)

	assertSameF32(t, "qwen35 hybrid prefill parity", hostPrefill, splitPrefill)

	for i, id := range []int{10, 20} {
		hostStep := hostS.Step(id)
		splitStep := splitS.Step(id)
		assertSameF32(t, fmt.Sprintf("qwen35 hybrid step %d", i), hostStep, splitStep)
	}
}

// TestDenseGPULayerPrefixSnapshotPreservesPlacement verifies that PrefixSnapshot captures
// and restores DenseGPULayers across snapshots.
func TestDenseGPULayerPrefixSnapshotPreservesPlacement(t *testing.T) {
	cfg := newTestDense4LayerConfig()
	m := NewSynthetic(cfg)
	be := compute.Pick("cpu-ref")

	s := m.NewBackendSession(be)
	s.DenseGPULayers = 2

	prompt := []int{3, 7, 11}
	_ = s.Prefill(prompt)

	snap, err := s.PrefixSnapshot()
	if err != nil {
		t.Fatalf("PrefixSnapshot: %v", err)
	}
	defer snap.Close()

	if snap.DenseGPULayers != 2 {
		t.Fatalf("snapshot DenseGPULayers = %d, want 2", snap.DenseGPULayers)
	}

	restored := m.NewBackendSession(be)
	if err := snap.Restore(restored); err != nil {
		t.Fatalf("Restore: %v", err)
	}

	if restored.DenseGPULayers != 2 {
		t.Fatalf("restored DenseGPULayers = %d, want 2", restored.DenseGPULayers)
	}

	// Continued decode on restored session matches original session continuation
	origStep := s.Step(17)
	restoredStep := restored.Step(17)
	assertSameF32(t, "restored step parity", origStep, restoredStep)
}
