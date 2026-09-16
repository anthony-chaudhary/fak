package model

import "github.com/anthony-chaudhary/fak/internal/compute"

// q4kExpertInputHAL runs the two Q4_K expert input projections plus SwiGLU on a device
// backend and returns the fused host INTERMEDIATE, not the expert output.
//
// It is the NARROW fallback, not the primary device route: expertSwiGLU tries
// expertSwiGLUHAL first (moe.go), and that helper already keeps all THREE projections —
// including the Q5_K/Q6_K down projection — on the backend via Backend.MatMul over a
// resident raw k-quant tensor. So this function is reached only when the full-expert
// seam DECLINED: no routed-expert k-quant capability, no halW staging map, a GELU
// activation, a projection bias, or a gate/up/down weight with no HAL-supporting
// resident representation (e.g. an IQ3_XXS down). In those configurations the down
// projection has no device kernel either, so there is no surviving "Q5_K/Q6_K down is
// still host-side" gap left for this partial route to cover; it simply rescues the
// gate/up half for encodings the full seam cannot serve. No device capability means a
// clean decline, never a semantic fallback.
func q4kExpertInputHAL(s *Session, gateName, upName string, xn any, intermediate, hidden int) ([]float32, bool) {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory ||
		!s.useHALQ4KWeights() || s.M.Cfg.ActGeluTanh || s.M.Cfg.ActGeluErf ||
		s.M.has(gateName[:len(gateName)-len("weight")]+"bias") ||
		s.M.has(upName[:len(upName)-len("weight")]+"bias") ||
		s.M.q4kw[gateName] == nil || s.M.q4kw[upName] == nil {
		return nil, false
	}
	x, ok := xn.([]float32)
	if !ok || len(x) != hidden {
		return nil, false
	}
	xd := s.uploadHostF32([]int{hidden}, x, compute.MemoryActivation, "moe expert gate/up activation")
	defer s.Backend.Free(xd)
	g := s.Backend.MatMul(s.matWeightHAL(gateName), xd)
	u := s.Backend.MatMul(s.matWeightHAL(upName), xd)
	fused := s.Backend.SwiGLU(g, u)
	out := s.Backend.Read(fused)
	s.Backend.Free(g)
	s.Backend.Free(u)
	s.Backend.Free(fused)
	if len(out) != intermediate {
		panic("model: device expert gate/up returned wrong intermediate size")
	}
	return out, true
}
