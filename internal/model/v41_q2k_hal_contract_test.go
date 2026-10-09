package model

// v41_q2k_hal_contract_test.go — the fak#13358 acceptance witness for the DeepSeek
// V4.1 mixed-quant routed-expert device seam.
//
// The leaf composes the shared device gate/up operation landed by #13357
// (q4kExpertInputHAL, the incremental seam's two-projection + SwiGLU helper) into the
// production V4.1 MoE loop through the optional v41ForwardState.expertGateUp callback.
// On a device backend whose MatMul serves a checkpoint-tier Q2_K gate/up slate, the
// gate + up projections run on the device. A zero-limit expert also runs fused
// SwiGLU there; a positive limit reads only the two I-wide projection vectors
// for the bounded host clamp/activation before the existing Q3_K down contraction. The gate/up
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
	matmuls       int
	swiglu        int
	matmulWeights []compute.Buffer
}

func (b *v41HalSeamBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.UploadDtype = true
	c.DeviceMemory = true
	return c
}

func (b *v41HalSeamBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	b.matmuls++
	b.matmulWeights = append(b.matmulWeights, w.Buf())
	return b.Backend.MatMul(w, x)
}

func v41HalSeamExpertMatMuls(t *testing.T, s *Session, b *v41HalSeamBackend, rows int) int {
	t.Helper()
	experts := map[compute.Buffer]string{}
	known := map[compute.Buffer]bool{}
	for _, weight := range s.halW {
		known[weight.Buf()] = true
	}
	for layer := 0; layer < s.M.Cfg.NumLayers; layer++ {
		for expert := 0; expert < s.M.Cfg.NumExperts; expert++ {
			for _, leaf := range []string{"w1.weight", "w3.weight", "w2.weight"} {
				name := layerName(layer, "ffn.experts."+itoa(expert)+"."+leaf)
				if weight, ok := s.halW["kquant-raw:"+name]; ok {
					wantDtype := compute.Q2_K
					if leaf == "w2.weight" {
						wantDtype = compute.Q3_K
					}
					if weight.Dtype != wantDtype {
						t.Errorf("actual expert leaf %s staged dtype=%v want %v", leaf, weight.Dtype, wantDtype)
					}
					if _, duplicate := experts[weight.Buf()]; duplicate {
						t.Fatal("expert projections share an ambiguous recorded weight identity")
					}
					experts[weight.Buf()] = leaf
				}
			}
		}
	}
	counts := map[string]int{}
	ordinary := 0
	for _, weight := range b.matmulWeights {
		if leaf, ok := experts[weight]; ok {
			counts[leaf]++
		} else {
			if !known[weight] {
				t.Fatal("recorded MatMul weight has no Session-owned identity")
			}
			ordinary++
		}
	}
	if len(b.matmulWeights) != b.matmuls {
		t.Error("aggregate MatMul attempts differ from actual weight records")
	}
	if ordinary == 0 {
		t.Error("default composed projection operations were not recorded")
	}
	total := 0
	for _, leaf := range []string{"w1.weight", "w3.weight", "w2.weight"} {
		if counts[leaf] != rows {
			t.Errorf("actual expert %s MatMul count=%d want %d", leaf, counts[leaf], rows)
		}
		total += counts[leaf]
	}
	return total
}

func v41HalSeamDefaultProjectionRows(t *testing.T, m *Model, phase string, tokens int, denseBefore, groupedBefore map[string]float64) {
	t.Helper()
	dense := v41DenseTestDelta(v41DenseTestPhase(t, m, phase), denseBefore)
	grouped := v41DenseTestDelta(v41GroupedPhase(t, m, phase), groupedBefore)
	if dense["dense_projection_device_rows"] != float64(7*tokens*m.Cfg.NumLayers) || dense["dense_projection_host_rows"] != 0 || grouped["grouped_output_device_rows"] != float64(tokens*m.Cfg.NumLayers) || grouped["grouped_output_host_rows"] != 0 {
		t.Errorf("expert component %s invocation lost default projection composition dense=%v grouped=%v", phase, dense, grouped)
	}
}

func (b *v41HalSeamBackend) SwiGLU(g, u compute.Tensor) compute.Tensor {
	b.swiglu++
	return b.Backend.SwiGLU(g, u)
}

// SupportsDeviceWeightDtype reports the resident device dtype set the one-Halo Vulkan
// target serves: Q2_K gate/up AND the Q3_K down are both runnable since fak#13677 gave
// Q3_K a Vulkan kernel, so all three projections of the pinned slate dispatch on device
// (fak#13704 added the down seam).
func (b *v41HalSeamBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	switch dt {
	case compute.F32, compute.Q8_0, compute.Q4_K, compute.Q6_K, compute.Q2_K, compute.Q3_K:
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

// TestV41Q2KGateUpDownHALRunsDevice is the fak#13358/fak#13704 positive witness: a V4.1
// session whose routed experts carry a Q2_K gate/up + Q3_K down slate on a device backend
// must run gate/up on the device, retain fused device SwiGLU at limit zero, and — now that
// fak#13677 gave Q3_K a Vulkan kernel and fak#13704 added the device down seam — run the
// down projection on the device too, so no host expert GEMM remains. The result must
// reproduce the historical host triple's logits within tolerance. The device seam is the
// per-pick token-major contraction, which is exactly the contraction a fresh session's
// first decode step runs (seq == 1); a later step recomputes the whole history through the
// expert-major grouped contraction, which is a separate #13304 path this leaf does not touch.
// fak-test:runtime medium est=30s lane=default
func TestV41Q2KGateUpHALDeviceDown(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	m := v41MixedQuantExpertModel(t)
	for _, limit := range []float64{0, 2} {
		name := "zero limit fused activation"
		if limit > 0 {
			name = "positive limit device projections"
		}
		t.Run(name, func(t *testing.T) {
			m.Cfg.SwigluLimit = limit

			// Host oracle: a fresh session with no device backend, so the callback is nil and the
			// historical host gate/up/down triple runs for the same first decode step.
			hostSess := &Session{M: m}
			if hostSess.v41ExpertGateUpFunc() != nil {
				t.Fatal("a session with no backend bound a device gate/up callback")
			}
			want := v41DecodeHistory(t, hostSess, []int{1})

			// Device arm: the shared seam must activate and run exactly two MatMuls (gate, up)
			// plus one device SwiGLU per routed pick when the limit is zero.
			be := &v41HalSeamBackend{Backend: compute.Default()}
			devSess := &Session{M: m, Backend: be, halW: map[string]compute.Tensor{}}
			if devSess.v41ExpertGateUpFunc() == nil {
				t.Fatal("a DeviceMemory session did not bind the device gate/up callback")
			}
			// This witness counts routed-only activation calls. The shared
			// default callback has its own full-pipeline dispatch witness.
			devSess.v41State().sharedActivation = nil
			denseBefore, groupedBefore := v41DenseTestPhase(t, m, "decode"), v41GroupedPhase(t, m, "decode")
			m.v41SetExpertFaultPhase(V41PhaseDecode)
			got := v41DecodeHistory(t, devSess, []int{1})
			m.v41SetExpertFaultPhase(V41PhaseUnknown)
			v41HalSeamDefaultProjectionRows(t, m, "decode", 1, denseBefore, groupedBefore)

			// One decode Step (seq == 1) runs the token-major MoE contraction: NumExpertsPerTok
			// routed picks, each running gate + up + down (fak#13704 added the device down seam),
			// so three device MatMuls per pick.
			picks := m.Cfg.NumExpertsPerTok * m.Cfg.NumLayers
			if expertMatMuls := v41HalSeamExpertMatMuls(t, devSess, be, picks); expertMatMuls != 3*picks {
				t.Fatalf("device expert MatMul count = %d, want %d (gate+up+down per routed pick)", expertMatMuls, 3*picks)
			}
			if limit == 0 && be.swiglu != picks {
				t.Fatalf("zero-limit device SwiGLU count = %d, want %d (one fused SwiGLU per routed pick)", be.swiglu, picks)
			}

			// The Q3_K down gained a Vulkan kernel in fak#13677, so with the fak#13704 device
			// down seam every routed expert's down projection is staged device-side and no
			// host expert GEMM remains.
			stagedDown := 0
			for l := 0; l < m.Cfg.NumLayers; l++ {
				for e := 0; e < m.Cfg.NumExperts; e++ {
					down := layerName(l, "ffn.experts."+itoa(e)+".w2.weight")
					if _, staged := devSess.halW["kquant-raw:"+down]; staged {
						stagedDown++
					}
				}
			}
			if stagedDown == 0 {
				t.Fatal("no Q3_K down projection was staged on the device; the fak#13704 down seam did not fire")
			}

			// Token-history parity: the device gate/up/down reproduces the host triple.
			if len(got) != len(want) {
				t.Fatalf("device arm returned %d logit rows, want %d", len(got), len(want))
			}
			for tkn := range got {
				assertV41LogitsClose(t, got[tkn], want[tkn], "device Q2_K gate/up + host Q3_K down vs host triple")
			}
		})
	}
}

// The callback is driven directly so the activation limit cannot be hidden by
// the remaining attention, expert weighting, or LM-head contractions.
// fak-test:runtime fast est=20ms lane=default
func TestV41SwigluLimitGateUpHALRetainsDeviceProjections(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })
	const H, I = 256, 256
	stem := "ffn.experts.0"
	gateName := layerName(0, stem+".w1.weight")
	upName := layerName(0, stem+".w3.weight")
	m := &Model{
		Cfg: Config{HiddenSize: H, MoEIntermediateSize: I, SwigluLimit: 2},
		kqw: map[string]*kQuantTensor{
			gateName: q2kFixtureTensor(I, H, 0x13358),
			upName:   q2kFixtureTensor(I, H, 0x13359),
		},
	}
	w1, ok := m.residentF32Mat(gateName)
	if !ok {
		t.Fatal("resident Q2_K gate missing from oracle fixture")
	}
	w3, ok := m.residentF32Mat(upName)
	if !ok {
		t.Fatal("resident Q2_K up missing from oracle fixture")
	}
	// Calibrate the input to expose finite projections near +/-5. This makes
	// upper and lower saturation non-vacuous without driving SiLU into underflow.
	var largest float32
	for _, weights := range [][]float32{w1, w3} {
		for row := 0; row < I; row++ {
			var sum float32
			for _, value := range weights[row*H : (row+1)*H] {
				sum += value
			}
			largest = max(largest, abs32(sum))
		}
	}
	if largest == 0 {
		t.Fatal("Q2_K projections are vacuous")
	}
	identity := make([]float32, I*I)
	for i := 0; i < I; i++ {
		identity[i*I+i] = 1
	}
	for _, sign := range []float32{1, -1} {
		name := "upper bounds"
		if sign < 0 {
			name = "negative gate remains unbounded"
		}
		t.Run(name, func(t *testing.T) {
			x := make([]float32, H)
			for i := range x {
				x[i] = sign * 5 / largest
			}
			// Prove both projections cross the applicable limit in this arm.
			for name, weights := range map[string][]float32{"gate": w1, "up": w3} {
				crossed := false
				for row := 0; row < I; row++ {
					var sum float32
					for col, value := range x {
						sum += weights[row*H+col] * value
					}
					crossed = crossed || sign*sum > float32(m.Cfg.SwigluLimit)
				}
				if !crossed {
					t.Fatalf("%s projection never crossed limit %g", name, m.Cfg.SwigluLimit)
				}
			}
			want := v41SwigluLimitReference(w1, w3, identity, x, I, H, float32(m.Cfg.SwigluLimit))
			be := &v41HalSeamBackend{Backend: compute.Default()}
			s := &Session{M: m, Backend: be, halW: map[string]compute.Tensor{}}
			defer s.Close()
			callback := s.v41ExpertGateUpFunc()
			if callback == nil {
				t.Fatal("device gate/up callback absent")
			}
			got, outcome, err := callback(0, stem, x)
			if err != nil || outcome != v41GateUpHandled {
				t.Fatalf("positive-limit device callback outcome=%v err=%v, want handled", outcome, err)
			}
			if be.matmuls != 2 {
				t.Fatalf("positive-limit device MatMul count=%d, want 2 (gate+up)", be.matmuls)
			}
			for _, name := range []string{gateName, upName} {
				if _, staged := s.halW["kquant-raw:"+name]; !staged {
					t.Fatalf("compressed projection %s was not staged on device", name)
				}
				if m.has(name) {
					t.Fatalf("projection %s unexpectedly has an f32 manifest fallback", name)
				}
			}
			for i, value := range got {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					t.Fatalf("device activation[%d]=%g, want finite", i, value)
				}
			}
			assertV41LogitsClose(t, got, want, "device projection activation vs independent limited f32 oracle")
		})
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

// TestV41ExpertDownDeviceSeamFires is the fak#13704 focused reproduction: on a device
// backend that serves the pinned slate's dtypes (Q2_K gate/up + Q3_K down), the down
// projection must run on the backend, staging the Q3_K down device-side, with output
// matching the host triple. Before the leaf the down contraction ran on the host.
// fak-test:runtime medium est=2s lane=default
func TestV41ExpertDownDeviceSeamFires(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	m := v41MixedQuantExpertModel(t)
	be := &v41HalSeamBackend{Backend: compute.Default()}
	s := &Session{M: m, Backend: be, halW: map[string]compute.Tensor{}}
	if s.v41ExpertDownFunc() == nil {
		t.Fatal("a DeviceMemory session did not bind the device down callback")
	}
	// A routed expert's Q3_K down must resolve and stage on the device.
	down := layerName(0, "ffn.experts.0.w2.weight")
	fused := make([]float32, m.Cfg.MoEIntermediateSize)
	for i := range fused {
		fused[i] = float32((i%13)-6) / 32
	}
	got, outcome, err := s.v41ExpertDownFunc()(0, "ffn.experts.0", fused)
	if err != nil || outcome != v41DownHandled {
		t.Fatalf("device down outcome=%v err=%v, want handled", outcome, err)
	}
	if len(got) != m.Cfg.HiddenSize {
		t.Fatalf("device down returned width %d, want %d", len(got), m.Cfg.HiddenSize)
	}
	if _, staged := s.halW["kquant-raw:"+down]; !staged {
		t.Fatalf("Q3_K down %s was not staged device-side", down)
	}
	// Host oracle over the same bytes.
	w := m.kqw[down]
	if w == nil {
		t.Fatal("fixture lost the Q3_K down weight")
	}
	want := make([]float32, m.Cfg.HiddenSize)
	kQuantMatRowsRange(w, fused, want, 0, m.Cfg.HiddenSize)
	assertV41LogitsClose(t, got, want, "device Q3_K down vs host Q3_K down")
}

// TestV41ExpertDownDeviceSeamDeclinesWithoutKernel is the fak#13704 negative control: a
// backend whose MatMul has no Q3_K case must decline the down seam, so the host arm runs
// byte-for-byte and no Q3_K weight is staged on the device.
// fak-test:runtime medium est=2s lane=default
func TestV41ExpertDownDeviceSeamDeclinesWithoutKernel(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	m := v41MixedQuantExpertModel(t)
	// A vulkan-like backend that serves Q2_K but NOT Q3_K (the pre-#13677 dtype set).
	rec := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
	noQ3 := &noQ3KSeamBackend{expertHALRecordingBackend: rec}
	s := &Session{M: m, Backend: noQ3, halW: map[string]compute.Tensor{}}
	down := layerName(0, "ffn.experts.0.w2.weight")
	fused := make([]float32, m.Cfg.MoEIntermediateSize)
	for i := range fused {
		fused[i] = float32((i%11)-5) / 24
	}
	if _, outcome, _ := s.v41ExpertDownFunc()(0, "ffn.experts.0", fused); outcome != v41DownDeclined {
		t.Fatalf("down seam outcome=%v, want declined on a backend with no Q3_K MatMul", outcome)
	}
	if _, staged := s.halW["kquant-raw:"+down]; staged {
		t.Fatalf("Q3_K down %s was staged on a backend with no Q3_K MatMul", down)
	}
}

// noQ3KSeamBackend is a device backend that serves the pre-#13677 dtype set (no Q3_K), so
// the fak#13704 down seam must decline and leave the host arm in place.
type noQ3KSeamBackend struct {
	*expertHALRecordingBackend
}

func (b *noQ3KSeamBackend) Caps() compute.Caps {
	c := b.expertHALRecordingBackend.Backend.Caps()
	c.UploadDtype = true
	c.DeviceMemory = true
	return c
}

func (b *noQ3KSeamBackend) SupportsRoutedExpertKQuant() bool { return false }

func (b *noQ3KSeamBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	switch dt {
	case compute.F32, compute.Q8_0, compute.Q4_K, compute.Q6_K, compute.Q2_K:
		return true
	default:
		return false
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
