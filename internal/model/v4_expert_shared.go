package model

import (
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// ErrV4SharedExpert is the fail-closed marker for the always-on DeepSeek V4
// shared expert. When the admitted config declares one or more shared experts
// but their resident w1/w2/w3 weights are absent, the live FFN refuses rather
// than silently degrading to a routed-only token.
var ErrV4SharedExpert = errors.New("model: DeepSeek V4 shared expert forward failed")

// v4SharedExpertName returns the canonical resident name of one shared-expert
// projection. The promoted Flash loader installs the Q8 pair under exactly this
// name (v4DenseFP8RoleForName), and the f32 test/fixture path uses the same key.
func v4SharedExpertName(layer int, projection string) string {
	return "layers." + itoa(layer) + ".ffn.shared_experts." + projection + ".weight"
}

// v4SharedExpertForward executes the always-on shared expert for one normalized
// token and returns its host f32 contribution (length hidden). It mirrors the
// pinned routed-expert arithmetic exactly:
//
//	w2(SiLU(clamp(w1(x))) * clamp(w3(x)))
//
// with the same asymmetric clamp the routed path applies (v4_expert_compose.go):
// the gate is clamped only from above, the up projection symmetrically. The
// shared expert carries no router weight. Every intermediate compute.Tensor is
// freed; the resident matmul weights are borrowed and never freed here.
func (s *Session) v4SharedExpertForward(layer int, x []float32, swigluLimit float32, hidden, intermediate int) ([]float32, error) {
	if s == nil || s.M == nil || s.Backend == nil {
		return nil, fmt.Errorf("%w: session is not ready", ErrV4SharedExpert)
	}
	if layer < 0 {
		return nil, fmt.Errorf("%w: layer=%d", ErrV4SharedExpert, layer)
	}
	if hidden <= 0 || intermediate <= 0 || len(x) != hidden {
		return nil, fmt.Errorf("%w: input width=%d hidden=%d intermediate=%d", ErrV4SharedExpert, len(x), hidden, intermediate)
	}
	if swigluLimit < 0 || math.IsNaN(float64(swigluLimit)) || math.IsInf(float64(swigluLimit), 0) {
		return nil, fmt.Errorf("%w: swiglu limit=%v", ErrV4SharedExpert, swigluLimit)
	}
	w1Name := v4SharedExpertName(layer, "w1")
	w2Name := v4SharedExpertName(layer, "w2")
	w3Name := v4SharedExpertName(layer, "w3")
	for _, name := range []string{w1Name, w2Name, w3Name} {
		if !s.M.hasWeight(name) {
			return nil, fmt.Errorf("%w: missing %s", ErrV4SharedExpert, name)
		}
	}

	xTensor := s.uploadHostF32([]int{hidden}, x, compute.MemoryActivation, "v4-shared-expert-input")
	defer s.Backend.Free(xTensor)

	gateTensor := s.Backend.MatMul(s.matWeightHAL(w1Name), xTensor)
	gate := s.Backend.Read(gateTensor)
	s.Backend.Free(gateTensor)
	upTensor := s.Backend.MatMul(s.matWeightHAL(w3Name), xTensor)
	up := s.Backend.Read(upTensor)
	s.Backend.Free(upTensor)
	if len(gate) != intermediate || len(up) != intermediate {
		return nil, fmt.Errorf("%w: shared gate=%d up=%d want intermediate=%d", ErrV4SharedExpert, len(gate), len(up), intermediate)
	}

	activated := make([]float32, intermediate)
	for i := range gate {
		gateValue, upValue := gate[i], up[i]
		if !finite32(gateValue) || !finite32(upValue) {
			return nil, fmt.Errorf("%w: non-finite projection at %d", ErrV4SharedExpert, i)
		}
		if swigluLimit > 0 {
			gateValue = min(gateValue, swigluLimit)
			upValue = max(-swigluLimit, min(upValue, swigluLimit))
		}
		activated[i] = v4SiLU(gateValue) * upValue
	}
	activation := s.uploadHostF32([]int{intermediate}, activated, compute.MemoryActivation, "v4-shared-expert-activation")
	defer s.Backend.Free(activation)

	outTensor := s.Backend.MatMul(s.matWeightHAL(w2Name), activation)
	out := s.Backend.Read(outTensor)
	s.Backend.Free(outTensor)
	if len(out) != hidden {
		return nil, fmt.Errorf("%w: shared output width=%d want %d", ErrV4SharedExpert, len(out), hidden)
	}
	for i := range out {
		if !finite32(out[i]) {
			return nil, fmt.Errorf("%w: non-finite output at %d", ErrV4SharedExpert, i)
		}
	}
	return out, nil
}
