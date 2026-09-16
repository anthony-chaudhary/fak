package model

import "github.com/anthony-chaudhary/fak/internal/compute"

// q4kExpertInputAdmitted is the shared admission for the incremental device expert seam.
// It admits only a bias-free SiLU (no GELU) expert whose gate/up weights are resident
// Q4_K on a DeviceMemory backend, with an activation of the exact hidden width. A nil
// session, a non-device backend, a GELU expert, or a missing projection is a clean
// decline (false), never a semantic fallback.
func q4kExpertInputAdmitted(s *Session, gateName, upName string, xn any, hidden int) ([]float32, bool) {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory ||
		!s.useHALQ4KWeights() || s.M.Cfg.ActGeluTanh || s.M.Cfg.ActGeluErf ||
		s.M.has(gateName[:len(gateName)-len("weight")]+"bias") ||
		s.M.has(upName[:len(upName)-len("weight")]+"bias") ||
		s.M.q4kw[gateName] == nil || s.M.q4kw[upName] == nil {
		return nil, false
	}
	// The staged gate/up weights are Q4_K, so the ACTUAL backend (not the model descriptor)
	// must have a resident device MatMul kernel for Q4_K. A backend whose device seam would
	// panic on Q4_K is a clean decline here, never a panic downstream.
	if !compute.BackendSupportsDeviceWeightDtype(s.Backend, compute.Q4_K) {
		return nil, false
	}
	x, ok := xn.([]float32)
	if !ok || len(x) != hidden {
		return nil, false
	}
	return x, true
}

// q4kExpertDownDeviceWeight resolves and stages one expert down projection for the device
// seam. It uses the same resolution rule as the full expertSwiGLUHAL route
// (resolveExpertWeight), so a resident Q4_K or Q5_K/Q6_K down weight both resolve. A
// checkpoint-served (R5/#5616) weight resolves but has no resident bytes here: the
// incremental seam has no ring staging, so it declines (checkpoint weights stay on the
// full route).
//
// Admission is TWO-level and fail-closed. The MODEL descriptor check
// (SupportsHALKQuant) says the kind has a HAL kernel somewhere; the BACKEND check
// (compute.BackendSupportsDeviceWeightDtype on the resolved device dtype) says THIS
// backend's MatMul actually has a kernel for it. Both must pass. A k-quant whose
// descriptor has no HAL kernel (Q3_K), or a dtype the backend's MatMul switch has no
// case for (e.g. Q5_K on Vulkan), is a clean (compute.Tensor{},false) decline — never a
// panic and never a host-expanded fallback.
func (s *Session) q4kExpertDownDeviceWeight(downName string) (compute.Tensor, bool) {
	w, ok := s.resolveExpertWeight(downName)
	if !ok {
		return compute.Tensor{}, false
	}
	switch {
	case w.q4 != nil:
		if !compute.BackendSupportsDeviceWeightDtype(s.Backend, compute.Q4_K) {
			return compute.Tensor{}, false
		}
		return s.weightHALQ4K(downName, w.q4), true
	case w.kq != nil:
		desc, ok := LookupQuantDescriptor(w.kq.kind)
		if !ok || !desc.SupportsHAL() {
			return compute.Tensor{}, false
		}
		if !compute.BackendSupportsDeviceWeightDtype(s.Backend, desc.Dtype()) {
			return compute.Tensor{}, false
		}
		return s.weightHALKQuant(downName, w.kq), true
	default:
		return compute.Tensor{}, false
	}
}

// q4kExpertInputHAL runs the two Q4_K expert input projections plus SwiGLU on a device backend
// and returns the fused host intermediate. It is the narrow half of the seam: a caller that needs
// the I-wide intermediate on the host (e.g. a caller whose down projection has no device kernel)
// still gets it. When the down projection ALSO has a device kernel, q4kExpertGateUpDownHAL keeps
// the intermediate resident and returns the final [H] output instead. No device capability means a
// clean decline, never a semantic fallback.
func q4kExpertInputHAL(s *Session, gateName, upName string, xn any, intermediate, hidden int) ([]float32, bool) {
	x, ok := q4kExpertInputAdmitted(s, gateName, upName, xn, hidden)
	if !ok {
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

// q4kExpertGateUpDownHAL is the full incremental device seam: it resolves the down projection to a
// device kernel (Q4_K or a k-quant descriptor that supports HAL staging — Q5_K/Q6_K), then runs
// gate MatMul, up MatMul, SwiGLU and down MatMul on Backend WITHOUT ever reading the I-wide fused
// intermediate back to host, and returns the final [H] output. This is the sibling of
// expertSwiGLUHAL: that route needs the routed-expert k-quant capability and stages all three
// weights through the ring, while this one is the narrow Q4_K-gate/up incremental path for a
// DeviceMemory backend that does not advertise the routed capability.
//
// Every decline (nil session, non-device backend, GELU expert, missing/biased projection, an
// unresolvable down weight, or a kind with no device kernel) returns (nil,false) so the caller
// falls through to the host path byte-for-byte. It never panics on an unsupported kind and never
// silently runs the down projection on the host.
func q4kExpertGateUpDownHAL(s *Session, gateName, upName, downName string, xn any, intermediate, hidden int) ([]float32, bool) {
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory ||
		s.M.has(downName[:len(downName)-len("weight")]+"bias") {
		return nil, false
	}
	x, ok := q4kExpertInputAdmitted(s, gateName, upName, xn, hidden)
	if !ok {
		return nil, false
	}
	downW, ok := s.q4kExpertDownDeviceWeight(downName)
	if !ok {
		return nil, false
	}
	xd := s.uploadHostF32([]int{hidden}, x, compute.MemoryActivation, "moe expert gate/up activation")
	defer s.Backend.Free(xd)
	g := s.Backend.MatMul(s.matWeightHAL(gateName), xd)
	u := s.Backend.MatMul(s.matWeightHAL(upName), xd)
	fused := s.Backend.SwiGLU(g, u)
	out := s.Backend.MatMul(downW, fused)
	res := s.Backend.Read(out)
	s.Backend.Free(g)
	s.Backend.Free(u)
	s.Backend.Free(fused)
	s.Backend.Free(out)
	if len(res) != hidden {
		panic("model: device expert gate/up/down returned wrong hidden size")
	}
	return res, true
}
