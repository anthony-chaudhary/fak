package model

import "github.com/anthony-chaudhary/fak/internal/compute"

// expertInputDeviceAdmitted is the shared admission for the incremental device expert
// seam. It admits a bias-free SiLU (no GELU) expert on a DeviceMemory backend, with an
// activation of the exact hidden width. Each gate/up projection is resolved through the
// same tiered resolution the full expertSwiGLUHAL route uses (resolveExpertWeight), so a
// resident Q4_K, a resident k-quant, or a checkpoint-backed (R5/#5616) Q2_K all resolve;
// the ACTUAL backend must then have a resident device MatMul kernel for the resolved
// dtype (compute.BackendSupportsDeviceWeightDtype), so a backend whose MatMul would panic
// on that dtype is a clean decline, never a panic downstream. A nil session, a non-device
// backend, a GELU expert, a biased projection, an unresolvable weight, or a dtype the
// backend cannot serve is a clean decline (false), never a semantic fallback.
//
// The returned tensors are the staged gate/up device weights; the caller owns their
// lifetime. Under a bounded routed-expert ring the caller must hold them for the rest of
// its computation (stage-and-hold must be ONE span, R7/#5618), which the callers do.
func expertInputDeviceAdmitted(s *Session, gateName, upName string, xn any, hidden int) (compute.Tensor, string, compute.Tensor, string, []float32, bool) {
	none := compute.Tensor{}
	if s == nil || s.M == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory ||
		s.M.Cfg.ActGeluTanh || s.M.Cfg.ActGeluErf ||
		s.M.has(gateName[:len(gateName)-len("weight")]+"bias") ||
		s.M.has(upName[:len(upName)-len("weight")]+"bias") {
		return none, "", none, "", nil, false
	}
	x, ok := xn.([]float32)
	if !ok || len(x) != hidden {
		return none, "", none, "", nil, false
	}
	gateW, gateKey, ok := s.expertInputDeviceWeight(gateName)
	if !ok {
		return none, "", none, "", nil, false
	}
	upW, upKey, ok := s.expertInputDeviceWeight(upName)
	if !ok {
		return none, "", none, "", nil, false
	}
	return gateW, gateKey, upW, upKey, x, true
}

// expertInputDeviceWeight resolves and stages one expert gate/up projection for the
// incremental device seam, admitting resident Q4_K, resident k-quants, and
// checkpoint-backed (R5/#5616) weights alike, and declining anything the ACTUAL backend's
// MatMul cannot serve. It returns the staged device tensor together with the HAL ring key
// it was staged under, so the caller can hold it across its GEMMs under a bounded ring.
//
// Each resident tier is gated by the SAME session predicate matWeightHAL uses to select
// it (useHALQ4KWeights / useHALKQuantWeights), so a session that does not stage that
// tier keeps the host arm byte-for-byte — the Q4K=false contract pinned by
// moe_device_engine_test. A checkpoint-served weight needs no resident flag: the tier
// itself is the authorization, which is what lets a non-Q4_K V4.1 Q2_K expert slate reach
// the device seam.
func (s *Session) expertInputDeviceWeight(name string) (compute.Tensor, string, bool) {
	w, ok := s.resolveExpertWeight(name)
	if !ok {
		return compute.Tensor{}, "", false
	}
	switch {
	case w.q4 != nil:
		if !s.useHALQ4KWeights() || !compute.BackendSupportsDeviceWeightDtype(s.Backend, compute.Q4_K) {
			return compute.Tensor{}, "", false
		}
		return s.weightHALQ4K(name, w.q4), w.halKey(), true
	case w.kq != nil:
		desc, ok := LookupQuantDescriptor(w.kq.kind)
		if !ok || !desc.SupportsHAL() || !s.useHALKQuantWeights() {
			return compute.Tensor{}, "", false
		}
		if !compute.BackendSupportsDeviceWeightDtype(s.Backend, desc.Dtype()) {
			return compute.Tensor{}, "", false
		}
		return s.weightHALKQuant(name, w.kq), w.halKey(), true
	default:
		// A checkpoint-served projection is staged through the SAME bounded path as a
		// resident one — same key, same dtype, same byte accounting (R5/#5616) — so the
		// ring cannot tell the two apart and a hit costs no checkpoint read.
		if !compute.BackendSupportsDeviceWeightDtype(s.Backend, w.ck.dt) {
			return compute.Tensor{}, "", false
		}
		return s.weightHALStagedBounded(w.ck.key, name, w.ck.mk, w.ck.dt, w.ck.bytes), w.halKey(), true
	}
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
	gateW, gateKey, upW, upKey, x, ok := expertInputDeviceAdmitted(s, gateName, upName, xn, hidden)
	if !ok {
		return nil, false
	}
	// Under a bounded routed-expert ring, staging `up` could evict `gate` and Free a handle
	// the GEMMs below still need. Hold each weight for the rest of this computation and
	// release it on return; without a ring every hold is a no-op and this is byte-for-byte
	// the previous path. Staging and holding are ONE span (R7/#5618) so a peer agent's stage
	// landing between them cannot evict the handle just returned.
	for _, h := range []struct {
		name string
		key  string
	}{{gateName, gateKey}, {upName, upKey}} {
		r := s.routedExpertRing(h.name)
		if r == nil {
			continue
		}
		done := s.ringEnter(r)
		r.hold(h.key)
		done()
		defer func(r *pagedRing, key string) {
			done := s.ringEnter(r)
			r.release(key)
			done()
		}(r, h.key)
	}
	xd := s.uploadHostF32([]int{hidden}, x, compute.MemoryActivation, "moe expert gate/up activation")
	defer s.Backend.Free(xd)
	g := s.Backend.MatMul(gateW, xd)
	u := s.Backend.MatMul(upW, xd)
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
	gateW, gateKey, upW, upKey, x, ok := expertInputDeviceAdmitted(s, gateName, upName, xn, hidden)
	if !ok {
		return nil, false
	}
	downW, ok := s.q4kExpertDownDeviceWeight(downName)
	if !ok {
		return nil, false
	}
	// Hold the gate/up (and resolved down) weights for the rest of this expert's computation
	// under a bounded routed-expert ring, so staging the later projections cannot evict an
	// earlier handle the GEMMs still need. Without a ring every hold is a no-op.
	holds := []struct {
		name string
		key  string
	}{{gateName, gateKey}, {upName, upKey}}
	for _, h := range holds {
		r := s.routedExpertRing(h.name)
		if r == nil {
			continue
		}
		done := s.ringEnter(r)
		r.hold(h.key)
		done()
		defer func(r *pagedRing, key string) {
			done := s.ringEnter(r)
			r.release(key)
			done()
		}(r, h.key)
	}
	xd := s.uploadHostF32([]int{hidden}, x, compute.MemoryActivation, "moe expert gate/up activation")
	defer s.Backend.Free(xd)
	g := s.Backend.MatMul(gateW, xd)
	u := s.Backend.MatMul(upW, xd)
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
