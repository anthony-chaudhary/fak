package model

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// tileSyntheticOutProjForTest rewrites the GDN out_proj of the given layers (every GDN layer when
// none are given) of an f32 synthetic model into the GGUF tiled input-column order
// (W_tiled[:, c] = W[:, src(c)] per head), marks them tiled, and returns the names. The model
// must then reproduce its untouched twin.
func tileSyntheticOutProjForTest(t *testing.T, m *Model, layers ...int) []string {
	t.Helper()
	nK, nV, _, vHd, _, valDim, _ := m.Cfg.linearAttnDims()
	src, err := compute.Qwen35TiledValueHeadSource(nK, nV)
	if err != nil {
		t.Fatal(err)
	}
	if len(layers) == 0 {
		for l := 0; l < m.Cfg.NumLayers; l++ {
			if m.Cfg.isLinearAttnLayer(l) {
				layers = append(layers, l)
			}
		}
	}
	var names []string
	for _, l := range layers {
		if !m.Cfg.isLinearAttnLayer(l) {
			t.Fatalf("layer %d is not a GDN layer", l)
		}
		name := layerName(l, "linear_attn.out_proj.weight")
		w := m.tensor(name)
		grouped := append([]float32(nil), w...)
		for row := 0; row < m.Cfg.HiddenSize; row++ {
			base := row * valDim
			for c, h := range src {
				copy(w[base+c*vHd:base+(c+1)*vHd], grouped[base+h*vHd:base+(h+1)*vHd])
			}
		}
		names = append(names, name)
	}
	m.qwen35TiledOutProj = map[string]struct{}{}
	for _, name := range names {
		m.qwen35TiledOutProj[name] = struct{}{}
	}
	return names
}

type tiledRecordingQwen35Backend struct {
	*recordingQwen35Backend
	tiledCalls int
}

func (b *tiledRecordingQwen35Backend) Qwen35GDNDecodeTiledOut(
	normalizedInput,
	inProjQKV, inProjZ, inProjB, inProjA,
	conv1D, aLog, dtBias, norm, outProj,
	convState, recurrentState compute.Tensor,
	numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
	rmsNormEpsilon float32,
) (output, nextConvState, nextRecurrentState compute.Tensor, err error) {
	b.tiledCalls++
	return b.recordingQwen35Backend.Qwen35GDNDecode(normalizedInput, inProjQKV, inProjZ, inProjB, inProjA,
		conv1D, aLog, dtBias, norm, outProj, convState, recurrentState,
		numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel, rmsNormEpsilon)
}

// fak-test:runtime fast est=50ms
func TestQwen35TiledOutProjectionPredicate(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	cases := []struct {
		canon string
		want  bool
	}{
		{"model.layers.0.linear_attn.out_proj.weight", true},
		{"model.layers.0.linear_attn.in_proj_qkv.weight", false},
		{"model.layers.3.self_attn.o_proj.weight", false},
		{"mtp.layers.0.linear_attn.out_proj.weight", false},
	}
	for _, tc := range cases {
		if got := Qwen35TiledOutProjection(cfg, tc.canon); got != tc.want {
			t.Errorf("%s: got %v, want %v", tc.canon, got, tc.want)
		}
	}
	same := cfg
	same.LinearNumValueHeads = same.LinearNumKeyHeads
	if Qwen35TiledOutProjection(same, "model.layers.0.linear_attn.out_proj.weight") {
		t.Error("nV == nK has no column reorder to defer and must not be admitted")
	}
	dense := cfg
	dense.LayerTypes = nil
	if Qwen35TiledOutProjection(dense, "model.layers.0.linear_attn.out_proj.weight") {
		t.Error("non-hybrid config admitted")
	}
}

// fak-test:runtime fast est=50ms
func TestQwen35TiledOutProjBuilderRefusesForeignAndMisshapedTensors(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	cfg.HiddenSize, cfg.LinearValueHeadDim = 256, 64 // valDim 256: one Q4_K super-block per row
	shape := []int{256, 256}
	raw := make([]byte, 256*q4kBlockBytes)
	if err := NewQuantBuilder(cfg, false).AddQwen35TiledOutProj("model.layers.0.linear_attn.in_proj_z.weight", shape, raw, "Q4_K"); err == nil {
		t.Error("in_proj_z admitted as a tiled out_proj")
	}
	if err := NewQuantBuilder(cfg, false).AddQwen35TiledOutProj("model.layers.0.linear_attn.out_proj.weight", []int{256, 128}, raw, "Q4_K"); err == nil {
		t.Error("wrong reduction width admitted")
	}
	if err := NewQuantBuilder(cfg, false).AddQwen35TiledOutProj("model.layers.0.linear_attn.out_proj.weight", shape, raw[:len(raw)-1], "Q4_K"); err == nil {
		t.Error("short payload admitted")
	}
	b := NewQuantBuilder(cfg, false)
	if err := b.AddQwen35TiledOutProj("model.layers.0.linear_attn.out_proj.weight", shape, raw, "Q4_K"); err != nil {
		t.Fatalf("valid tiled out_proj refused: %v", err)
	}
	if err := b.AddQwen35TiledOutProj("model.layers.0.linear_attn.out_proj.weight", shape, raw, "Q4_K"); err == nil {
		t.Error("duplicate tiled out_proj admitted")
	}
	if !b.m.Qwen35OutProjTiled("model.layers.0.linear_attn.out_proj.weight") || b.m.Qwen35OutProjTiled("model.layers.1.linear_attn.out_proj.weight") {
		t.Error("tiled flag not recorded for exactly the stored name")
	}
}

// fak-test:runtime fast est=300ms
func TestQwen35TiledOutProjHALAdmissionAndRouting(t *testing.T) {
	cfg := qwen35HybridTestCfg()
	m := NewSynthetic(cfg)
	tileSyntheticOutProjForTest(t, m, 0)
	ref := NewSynthetic(cfg).NewSession()

	_, err := m.NewBackendSessionChecked(newRecordingQwen35Backend(m))
	var unsupported *UnsupportedBackendForwardError
	if !errors.As(err, &unsupported) || unsupported.Forward != ForwardQwen35GDN {
		t.Fatalf("backend without core tiling: err=%v, want *UnsupportedBackendForwardError", err)
	}

	be := &tiledRecordingQwen35Backend{recordingQwen35Backend: newRecordingQwen35Backend(m)}
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()
	steps := []int{3, 7, 11}
	for _, id := range steps {
		got := append([]float32(nil), s.Step(id)...)
		// The recording backend computes GDN through the resident host kernel, so this also
		// pins the host input seam: tiled weights + tiled core == the untouched twin.
		if d := maxAbsDelta(got, ref.Step(id)); d > 2e-5 {
			t.Fatalf("token %d: tiled model differs from its untouched twin, max|delta|=%g", id, d)
		}
	}
	// Negative control: the same permuted columns without the tile flag must not match, so the
	// parity above is not vacuous.
	unflagged := NewSynthetic(cfg)
	tileSyntheticOutProjForTest(t, unflagged, 0)
	unflagged.qwen35TiledOutProj = nil
	plainBE := newRecordingQwen35Backend(unflagged)
	plain, err := unflagged.NewBackendSessionChecked(plainBE)
	if err != nil {
		t.Fatal(err)
	}
	defer plain.Close()
	if d := maxAbsDelta(plain.Step(steps[0]), NewSynthetic(cfg).NewSession().Step(steps[0])); d < 1e-3 {
		t.Fatalf("permuted out_proj without tiling still matched (max|delta|=%g); the witness is vacuous", d)
	}
	linear := len(be.linearLayers)
	if be.tiledCalls != len(steps) || be.gdnCalls != len(steps)*linear {
		t.Fatalf("tiled/all GDN decode calls = %d/%d, want %d/%d (only layer 0 is tiled)", be.tiledCalls, be.gdnCalls, len(steps), len(steps)*linear)
	}

	seq := newSequencePrefillBackend(m)
	ss, err := m.NewBackendSessionChecked(&tiledSequencePrefillBackend{sequencePrefillBackend: seq})
	if err != nil {
		t.Fatal(err)
	}
	defer ss.Close()
	req := ss.qwen35SequencePrefillRequest([]int{3, 7, 11}, true)
	for l, layer := range req.Layers {
		want := layer.Linear && l == 0
		if layer.GDNOutTiled != want {
			t.Errorf("layer %d GDNOutTiled = %v, want %v", l, layer.GDNOutTiled, want)
		}
	}
}

type tiledSequencePrefillBackend struct {
	*sequencePrefillBackend
}

func (b *tiledSequencePrefillBackend) Qwen35GDNDecodeTiledOut(
	normalizedInput,
	inProjQKV, inProjZ, inProjB, inProjA,
	conv1D, aLog, dtBias, norm, outProj,
	convState, recurrentState compute.Tensor,
	numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel int,
	rmsNormEpsilon float32,
) (output, nextConvState, nextRecurrentState compute.Tensor, err error) {
	return b.Qwen35GDNDecode(normalizedInput, inProjQKV, inProjZ, inProjB, inProjA,
		conv1D, aLog, dtBias, norm, outProj, convState, recurrentState,
		numKeyHeads, numValueHeads, keyHeadDim, valueHeadDim, convKernel, rmsNormEpsilon)
}
