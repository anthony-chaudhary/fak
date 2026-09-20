package model

// v41_q2k_hal_contract_test.go — the fak#13358 acceptance witness for the DeepSeek
// V4.1 mixed-quant routed-expert device seam.
//
// The leaf composes the shared device gate/up operation landed by #13357
// (q4kExpertInputHAL, the incremental seam's two-projection + SwiGLU helper) into the
// production V4.1 MoE loop through the optional v41ForwardState.expertGateUp callback.
// On a device backend whose MatMul serves a checkpoint-tier Q2_K gate/up slate, the
// gate + up projections and the SwiGLU run on the device and only the I-wide fused
// intermediate crosses back for the EXISTING host Q3_K down contraction — the gate/up
// f32 weights are never materialized on the host, and no full-device triple is claimed.
//
// Independence: the "host oracle" is the SAME reduced fixture driven with the callback
// disabled (a session with no device backend), i.e. the historical host gate/up/down
// triple. The contract compared is the session-visible token history; the two runs must
// agree within the backend's f32 reduction-order tolerance, exactly the parity rule the
// #13357 device witness uses. A mismatched fixture (device gate/up silently dropping the
// down projection, or a callback that swallows a selected failure) fails this test.
//
// [SW-VERIFIED]: the device backend forwards to cpu-ref (compute.Default()); this is a
// deterministic software contract, NOT a physical GPU qualification. No [HW-WITNESSED]
// criterion is claimed.

import (
	"errors"
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v41HalSeamBackend forwards to a cpu-ref backend while presenting DeviceMemory=true and
// counting the device MatMul/SwiGLU ops the shared gate/up operation issues. It does not
// advertise SupportsRoutedExpertKQuant, so the full expertSwiGLUHAL route declines and the
// incremental q4kExpertInputHAL seam is the one reached.
type v41HalSeamBackend struct {
	compute.Backend
	matmuls int
	swiglu  int
}

func (b *v41HalSeamBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.UploadDtype = true
	c.DeviceMemory = true
	return c
}

func (b *v41HalSeamBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	b.matmuls++
	return b.Backend.MatMul(w, x)
}

func (b *v41HalSeamBackend) SwiGLU(g, u compute.Tensor) compute.Tensor {
	b.swiglu++
	return b.Backend.SwiGLU(g, u)
}

// SupportsDeviceWeightDtype reports the resident device dtype set the one-Halo Vulkan
// target serves for the #13357 slate: Q2_K gate/up are runnable, Q3_K down is NOT (the
// descriptor has no HAL kernel), so the down projection stays on the host by design.
func (b *v41HalSeamBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	switch dt {
	case compute.F32, compute.Q8_0, compute.Q4_K, compute.Q6_K, compute.Q2_K:
		return true
	default:
		return false
	}
}

// v41MixedQuantExpertModel builds the reduced V4.1 fixture and replaces every routed
// expert's gate/up/down leaf with a resident mixed-quant slate: Q2_K gate/up (the
// DeepSeek-V4.1-Flash Q2_K expert slate) and Q3_K down (the checkpoint-tier kind with no
// device kernel). The f32 manifest entries for those leaves are removed so resolution is
// forced through the resident k-quant store, exactly as a real quantized serve loads it.
//
// The reduced V4.1 geometry is widened to H = I = 256 so a k-quant super-block (qkK=256)
// divides the projection rows: a real Q2_K/Q3_K expert slate cannot be represented at the
// sub-256 reduced axes (nblk would be zero), so the seam's own numerical contract is
// exercised at a block-aligned width instead of an artificial sub-block one.
func v41MixedQuantExpertModel(t *testing.T) *Model {
	t.Helper()
	m := v41ReducedBlockAlignedModel(t)
	H, I := m.Cfg.HiddenSize, m.Cfg.MoEIntermediateSize
	if m.kqw == nil {
		m.kqw = map[string]*kQuantTensor{}
	}
	for l := 0; l < m.Cfg.NumLayers; l++ {
		for e := 0; e < m.Cfg.NumExperts; e++ {
			stem := "ffn.experts." + itoa(e)
			w1 := layerName(l, stem+".w1.weight")
			w3 := layerName(l, stem+".w3.weight")
			w2 := layerName(l, stem+".w2.weight")
			m.kqw[w1] = q2kFixtureTensor(I, H, uint64(0x13358+3*e))
			m.kqw[w3] = q2kFixtureTensor(I, H, uint64(0x13358+3*e+1))
			m.kqw[w2] = q3kFixtureTensor(H, I)
			delete(m.manifest, w1)
			delete(m.manifest, w3)
			delete(m.manifest, w2)
		}
	}
	return m
}

// v41ReducedBlockAlignedModel is v41ReducedModelLayers with HiddenSize and
// MoEIntermediateSize raised to a k-quant block multiple (256 = qkK), so the mixed-quant
// routed-expert slate resolves through real Q2_K/Q3_K super-blocks. Every other reduced
// axis (layer/head/expert/rank geometry) is unchanged.
func v41ReducedBlockAlignedModel(t *testing.T) *Model {
	t.Helper()
	cfg := v41TestReducedConfig(t, 1, V41RouterExperts)
	cfg.NumExpertsPerTok = V41RouterTopK
	cfg.NSharedExperts = 1
	cfg.RoutedScalingFactor = 1.5
	cfg.RopeScaling = ""
	cfg.LongRope = nil
	cfg.RopeFactor = 0
	cfg.RopeOrigContext = 0
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 10000
	}
	// k-quant super-blocks are qkK=256 wide; H and I must be multiples so nblk >= 1.
	cfg.HiddenSize = 256
	cfg.MoEIntermediateSize = 256
	cfg.QLoraRank = 32
	cfg.OLoraRank = 16
	cfg.OGroups = 2

	H := cfg.HiddenSize
	I := cfg.MoEIntermediateSize
	hd := cfg.HeadDim
	nH := cfg.NumHeads
	qHeadDim := nH * hd
	oDim := cfg.OLoraRank * cfg.OGroups

	type ts = synthTensor
	tensors := []ts{
		{"model.embed_tokens.weight", []int{cfg.VocabSize, H}},
		{"lm_head.weight", []int{cfg.VocabSize, H}},
		{"model.norm.weight", []int{H}},
	}
	for l := 0; l < cfg.NumLayers; l++ {
		tensors = append(tensors,
			ts{layerName(l, "attn_norm.weight"), []int{H}},
			ts{layerName(l, "ffn_norm.weight"), []int{H}},
			ts{layerName(l, "mhc.mixes.weight"), []int{v41MHCMixWidth, H}},
			ts{layerName(l, "mhc.base"), []int{v41MHCMixWidth}},
			ts{layerName(l, "mhc.scale"), []int{3}},
			ts{layerName(l, "attn.wq_a.weight"), []int{cfg.QLoraRank, H}},
			ts{layerName(l, "attn.wq_b.weight"), []int{qHeadDim, cfg.QLoraRank}},
			ts{layerName(l, "attn.wkv.weight"), []int{v41KVLoraRankReduced(cfg), H}},
			ts{layerName(l, "attn.wo_a.weight"), []int{cfg.OLoraRank, qHeadDim}},
			ts{layerName(l, "attn.wo_b.weight"), []int{H, oDim}},
			ts{layerName(l, "attn.sink"), []int{nH}},
			ts{layerName(l, "ffn.gate.weight"), []int{cfg.NumExperts, H}},
			ts{layerName(l, "ffn.gate.e_score_correction_bias"), []int{cfg.NumExperts}},
			ts{layerName(l, "ffn.shared_experts.w1.weight"), []int{I, H}},
			ts{layerName(l, "ffn.shared_experts.w3.weight"), []int{I, H}},
			ts{layerName(l, "ffn.shared_experts.w2.weight"), []int{H, I}},
		)
		for e := 0; e < cfg.NumExperts; e++ {
			stem := "ffn.experts." + itoa(e)
			tensors = append(tensors,
				ts{layerName(l, stem+".w1.weight"), []int{I, H}},
				ts{layerName(l, stem+".w3.weight"), []int{I, H}},
				ts{layerName(l, stem+".w2.weight"), []int{H, I}},
			)
		}
	}

	man, raw := synthBuildRaw(tensors, func(name string, next func() float32) float32 {
		switch {
		case name == "model.norm.weight" || hasSuffix(name, "attn_norm.weight") || hasSuffix(name, "ffn_norm.weight"):
			return 1.0
		case hasSuffix(name, "mhc.scale"):
			return 1.0
		case hasSuffix(name, "mhc.base"):
			return 0.0
		case hasSuffix(name, "attn.sink"):
			return 0.25 * next()
		default:
			return synthMatmulFill(name, next)
		}
	})
	return &Model{Cfg: cfg, manifest: man, raw: raw}
}

// v41DecodeHistory drives the session's single-token decode route (Session.Step's V4.1
// branch) over ids, returning each step's last-position logits. The cacheless V4.1
// assembly folds each Step into the committed history and recomputes the whole history,
// so a step after the first runs the multi-token expert-major grouped contraction; the
// per-pick device gate/up seam (#13358) is exercised by the token-major contraction the
// FIRST step of a fresh session runs (seq == 1). Callers that want to witness the seam
// therefore drive a fresh session's first step; callers that want the full-history parity
// drive every step and compare against the same-shaped host session.
func v41DecodeHistory(t *testing.T, s *Session, ids []int) [][]float32 {
	t.Helper()
	out := make([][]float32, 0, len(ids))
	for _, id := range ids {
		logits := s.stepV41(id)
		out = append(out, logits)
	}
	return out
}

// TestV41Q2KGateUpHALKeepsDownOnHost is the fak#13358 positive witness: a V4.1 session
// whose routed experts carry a Q2_K gate/up slate on a device backend must run gate, up
// and SwiGLU on the device through the shared #13357 operation, keep the Q3_K down
// contraction on the host (Q3_K has no device kernel), and reproduce the historical host
// triple's logits within tolerance. The device seam is the per-pick token-major
// contraction, which is exactly the contraction a fresh session's first decode step runs
// (seq == 1); a later step recomputes the whole history through the expert-major grouped
// contraction, which is a separate #13304 path this leaf does not touch.
func TestV41Q2KGateUpHALKeepsDownOnHost(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	m := v41MixedQuantExpertModel(t)

	// Host oracle: a fresh session with no device backend, so the callback is nil and the
	// historical host gate/up/down triple runs for the same first decode step.
	hostSess := &Session{M: m}
	if hostSess.v41ExpertGateUpFunc() != nil {
		t.Fatal("a session with no backend bound a device gate/up callback")
	}
	want := v41DecodeHistory(t, hostSess, []int{1})

	// Device arm: the shared seam must activate and run exactly two MatMuls (gate, up)
	// plus one SwiGLU per routed pick on the device.
	be := &v41HalSeamBackend{Backend: compute.Default()}
	devSess := &Session{M: m, Backend: be, halW: map[string]compute.Tensor{}}
	if devSess.v41ExpertGateUpFunc() == nil {
		t.Fatal("a DeviceMemory session did not bind the device gate/up callback")
	}
	got := v41DecodeHistory(t, devSess, []int{1})

	// One decode Step (seq == 1) runs the token-major MoE contraction: NumExpertsPerTok
	// routed picks, each running gate + up (two device MatMuls) and one fused SwiGLU.
	picks := m.Cfg.NumExpertsPerTok * m.Cfg.NumLayers
	if be.matmuls != 2*picks {
		t.Fatalf("device MatMul count = %d, want %d (gate+up per routed pick)", be.matmuls, 2*picks)
	}
	if be.swiglu != picks {
		t.Fatalf("device SwiGLU count = %d, want %d (one fused SwiGLU per routed pick)", be.swiglu, picks)
	}

	// The Q3_K down projection has no device kernel, so it must not have been staged as
	// a device weight: the down contraction stayed on the host.
	for l := 0; l < m.Cfg.NumLayers; l++ {
		for e := 0; e < m.Cfg.NumExperts; e++ {
			down := layerName(l, "ffn.experts."+itoa(e)+".w2.weight")
			if _, staged := devSess.halW["kquant-raw:"+down]; staged {
				t.Fatalf("Q3_K down %s was staged on the device; it has no HAL kernel and must stay on the host", down)
			}
		}
	}

	// Token-history parity: the device gate/up + host down reproduces the host triple.
	if len(got) != len(want) {
		t.Fatalf("device arm returned %d logit rows, want %d", len(got), len(want))
	}
	for tkn := range got {
		assertV41LogitsClose(t, got[tkn], want[tkn], "device Q2_K gate/up + host Q3_K down vs host triple")
	}
}

// TestV41Q2KGateUpHALDeclinesWithoutDevice is the negative control: the SAME mixed-quant
// model on a session with NO device backend must keep the callback nil and reproduce the
// historical host triple byte-for-byte. The callback is a strict addition: a model with no
// device seam runs exactly as it did before this leaf, bit-for-bit.
func TestV41Q2KGateUpHALDeclinesWithoutDevice(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	m := v41MixedQuantExpertModel(t)
	ids := []int{1, 2}

	// A session with no backend binds no callback (the seam is absent).
	noBackend := &Session{M: m}
	if noBackend.v41ExpertGateUpFunc() != nil {
		t.Fatal("a session with no backend bound a device gate/up callback")
	}
	// A session with a NON-device backend must also leave the callback nil.
	ref := compute.Default()
	if ref.Caps().DeviceMemory {
		t.Skip("cpu-ref unexpectedly advertises DeviceMemory; the negative control cannot be exercised")
	}
	nonDevice := &Session{M: m, Backend: ref}
	if nonDevice.v41ExpertGateUpFunc() != nil {
		t.Fatal("a non-device backend bound a device gate/up callback")
	}

	// Both must reproduce the historical host triple byte-for-byte.
	base := v41DecodeHistory(t, &Session{M: m}, ids)
	for name, alt := range map[string][][]float32{
		"fresh-session":      v41DecodeHistory(t, &Session{M: m}, ids),
		"non-device-session": v41DecodeHistory(t, nonDevice, ids),
	} {
		if len(alt) != len(base) {
			t.Fatalf("%s returned %d rows, want %d", name, len(alt), len(base))
		}
		for tkn := range base {
			for i := range base[tkn] {
				if base[tkn][i] != alt[tkn][i] {
					t.Fatalf("%s not byte-identical at token %d idx %d: %v != %v", name, tkn, i, base[tkn][i], alt[tkn][i])
				}
			}
		}
	}
}

// TestV41Q2KGateUpHALRejectsSourcelessGateUp is the fail-closed control: when the gate
// or up projection is present in NO store, the shared device operation declines and the
// host triple refuses with the typed ErrV41ForwardStage naming the tensor — a device
// session must never silently swallow that into a fabricated expert output. The device
// backend issues no expert MatMul/SwiGLU for the sourceless expert.
func TestV41Q2KGateUpHALRejectsSourcelessGateUp(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	m := v41MixedQuantExpertModel(t)
	missing := layerName(0, "ffn.experts.0.w1.weight")
	delete(m.kqw, missing)
	delete(m.manifest, missing)

	be := &v41HalSeamBackend{Backend: compute.Default()}
	s := &Session{M: m, Backend: be, halW: map[string]compute.Tensor{}}
	if s.v41ExpertGateUpFunc() == nil {
		t.Fatal("a DeviceMemory session did not bind the device gate/up callback")
	}
	_, err := s.M.forwardV41([]int{1}, s.v41State())
	if err == nil {
		t.Fatal("a routed expert with a sourceless gate projection produced logits instead of a typed refusal")
	}
	if !isV41ForwardStageErr(err) {
		t.Fatalf("sourceless gate projection error = %v, want a typed ErrV41ForwardStage", err)
	}
	// The shared device operation must have DECLINED the sourceless expert rather than
	// staged a partial triple: no device MatMul/SwiGLU ran for it.
	if be.matmuls != 0 || be.swiglu != 0 {
		t.Fatalf("sourceless gate projection ran device ops matmul=%d swiglu=%d, want 0/0", be.matmuls, be.swiglu)
	}
}

// assertV41LogitsClose compares two logit rows with the scale-relative bound the device
// f32 reduction-order contract uses (see the #13357 device witness).
func assertV41LogitsClose(t *testing.T, got, want []float32, what string) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: logit width %d, want %d", what, len(got), len(want))
	}
	for i := range got {
		bound := 1e-4 * float64(max32(1, abs32(want[i])))
		if d := math.Abs(float64(got[i] - want[i])); d > bound {
			t.Fatalf("%s: logit[%d] = %v, want %v (|d|=%v > %v)", what, i, got[i], want[i], d, bound)
		}
	}
}

// isV41ForwardStageErr reports whether err wraps the typed V41 forward refusal.
func isV41ForwardStageErr(err error) bool {
	return errors.Is(err, ErrV41ForwardStage)
}

func abs32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

func max32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
