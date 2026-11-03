package model

import "strings"

// moe_offload.go — the CPU-offload hybrid matKernel (the llama.cpp `--n-cpu-moe` equivalent).
//
// Routed-expert GEMMs dominate a 753B MoE model's bytes but fire sparsely (top-k per token) and
// are bandwidth-bound, so splitKernel keeps them in host RAM while the dense projections, router,
// attention, and the always-on shared expert run on the device backend. It routes per weight by
// predicate and introduces no new arithmetic: a weight computes exactly as its sub-kernel already
// does, so the only thing it can get wrong is the routing.
type splitKernel struct {
	host   matKernel         // where onHost(name)==true weights run (the offloaded experts)
	device matKernel         // where everything else runs (dense projections, router, attention)
	onHost func(string) bool // true => keep this weight's GEMM on host RAM
}

// prep is the identity: both sub-kernels accept the raw f32 activation at this seam.
func (k splitKernel) prep(x []float32) any { return x }

// mul routes the named weight to the host or device sub-kernel by the predicate — a pure
// placement decision over the same math.
func (k splitKernel) mul(name string, x any, out, in int) []float32 {
	if k.onHost(name) {
		return k.host.mul(name, x, out, in)
	}
	return k.device.mul(name, x, out, in)
}

// splitHostResidentExperts admits the batched host GEMV path: true when mat is a splitKernel whose
// host side is a resident kernel AND every pick (gate/up/down) is host-routed by the split's own
// predicate. It declines a graded split (ExpertSpillLayers) where any pick is device-kept: the
// batched host primitive only models host-resident experts, so a mixed layer must fall back to the
// per-expert loop rather than steal device-kept experts to the host.
func splitHostResidentExperts(mat matKernel, layer int, picks []routePick) bool {
	k, ok := mat.(splitKernel)
	if !ok {
		return false
	}
	// A hand-built splitKernel may carry a nil onHost (no host routing at all); treat that as a
	// decline rather than panicking on the nil call below.
	if k.onHost == nil {
		return false
	}
	// Both resident forms compute the same residentMatRows bytes; gating on the bare type
	// alone would silently decline the production split's sessionResidentKernel.
	switch k.host.(type) {
	case residentKernel, sessionResidentKernel:
	default:
		return false
	}
	for _, pk := range picks {
		for _, suffix := range [...]string{"gate_proj.weight", "up_proj.weight", "down_proj.weight"} {
			if !k.onHost(expertName(layer, pk.expert, suffix)) {
				return false
			}
		}
	}
	return true
}

// expertEngine is the compute engine a routed expert's projections run on. Residency and
// compute engine are independent on unified-memory APUs (CPU and iGPU share one pool), so the
// engine follows the kernel that exists for the encoding rather than a static name predicate.
type expertEngine int

const (
	// expertEngineHost is the fail-closed default: the host CPU k-quant GEMV, used whenever no
	// device kernel exists for the encoding or no device backend is attached.
	expertEngineHost expertEngine = iota
	// expertEngineDevice runs the expert's gate/up + SwiGLU on the device backend
	// (q4kExpertInputHAL) when a device kernel exists for its encoding.
	expertEngineDevice
)

// deviceKernelForExpertEncoding answers, at the model level, whether the model carries a
// representation of this expert projection that the device seam can execute. It is a pre-filter,
// never an execution promise: expertEngineForWeight still requires a DeviceMemory backend, and the
// seam declines fail-closed for any dtype the backend's resident MatMul cannot serve.
//
// Admitted encodings mirror the tiers expertInputDeviceWeight resolves:
//   - Q4_K (m.q4kw[name]): the Q4_K gate/up MatMul + SwiGLU (q4kExpertInputHAL).
//   - A HAL-capable resident k-quant (m.kqw[name], SupportsHALKQuant): the DSV4.1 Q2_K slate.
//   - The checkpoint tier (m.expertCheckpoint.staging): a streamed projection faulted on demand.
//
// Everything else answers expertEngineHost. `name` is the canonical expert tensor name (expertName).
func deviceKernelForExpertEncoding(m *Model, name string) expertEngine {
	if m == nil {
		return expertEngineHost
	}
	if m.q4kw[name] != nil {
		return expertEngineDevice
	}
	if qt := m.kqw[name]; qt != nil && SupportsHALKQuant(qt.kind) {
		return expertEngineDevice
	}
	if ck, ok := m.expertCheckpoint.staging(name); ok && ck != nil && ck.dt != 0 {
		return expertEngineDevice
	}
	return expertEngineHost
}

// expertEngineForWeight is the session-level admission wrapper: it resolves the engine by
// encoding and requires a backend capable of running a device kernel; otherwise it answers
// expertEngineHost, so the fail-closed default is structural.
func expertEngineForWeight(s *Session, name string) expertEngine {
	if s == nil || s.Backend == nil || !s.Backend.Caps().DeviceMemory {
		if s != nil {
			s.refuseHostFallback("routed expert " + name)
		}
		return expertEngineHost
	}
	engine := deviceKernelForExpertEncoding(s.M, name)
	if engine == expertEngineHost {
		s.refuseHostFallback("routed expert " + name)
	}
	return engine
}

// splitDeviceExpertInput runs the device engine for one routed expert's gate/up + SwiGLU when the
// split has routed its bytes to host RAM but a device kernel exists for its encoding. Residency is
// unchanged — only the compute moves to the iGPU over the same unified pool. It reuses
// q4kExpertInputHAL, so no new arithmetic is introduced; a decline leaves the caller on the host
// arm and ok=false writes nothing. The returned *Session is non-nil exactly when ok is true.
func (k splitKernel) splitDeviceExpertInput(layer, expert int, xn any) (*Session, []float32, bool) {
	// Only session-bearing device kernels have a device engine; a residentKernel device side
	// (no-backend degenerate split) declines.
	var s *Session
	switch dk := k.device.(type) {
	case backendKernel:
		s = dk.s
	case sessionQ4KKernel:
		s = dk.s
	default:
		return nil, nil, false
	}
	g := expertName(layer, expert, "gate_proj.weight")
	u := expertName(layer, expert, "up_proj.weight")
	if expertEngineForWeight(s, g) != expertEngineDevice || expertEngineForWeight(s, u) != expertEngineDevice {
		return nil, nil, false
	}
	out, ok := q4kExpertInputHAL(s, g, u, xn, s.M.Cfg.expertIntermediate(), s.M.Cfg.HiddenSize)
	if !ok {
		return nil, nil, false
	}
	return s, out, true
}

// sparseAttend forwards GLM-DSA's sparse-attention compute to the device side: it is attention
// work, not an expert, so it belongs with the dense projections on the GPU. When the device kernel
// does not advertise it (no-backend degenerate split), ok=false keeps the host loop, which is what
// makes the no-backend split bit-exact with the plain host forward.
func (k splitKernel) sparseAttend(q, selK, selV []float32, nSel, nH, qkHead, vHead int, scale float32) ([]float32, bool) {
	if dk, ok := k.device.(dsaSparseKernel); ok {
		return dk.sparseAttend(q, selK, selV, nSel, nH, qkHead, vHead, scale)
	}
	return nil, false
}

// indexSelect forwards the learned-indexer score + top-k selection to the device side, the
// symmetric partner of sparseAttend: the index projections stay on the device under a split, so the
// selection compute they feed belongs there too. When the device kernel does not advertise
// DSAIndexBackend, ok=false keeps the host loop, keeping the no-backend split selection-exact.
func (k splitKernel) indexSelect(indexQ, indexK, weights []float32, nKeys, nH, indexDim, queryPos, topK int, scale float32) ([]int, bool) {
	if dk, ok := k.device.(dsaIndexKernel); ok {
		return dk.indexSelect(indexQ, indexK, weights, nKeys, nH, indexDim, queryPos, topK, scale)
	}
	return nil, false
}

// isSharedExpertWeight matches the ALWAYS-ON shared-expert projections a GLM-style MoE runs for
// every token in addition to the routed top-k: model.layers.<l>.mlp.shared_experts.{gate,up,down}_proj.weight
// (glmSharedExperts). The distinguishing substring is ".mlp.shared_experts.".
func isSharedExpertWeight(name string) bool {
	return strings.Contains(name, ".mlp.shared_experts.") || strings.Contains(name, ".ffn.shared_experts.")
}

// isRoutedExpertWeight matches the per-token routed experts:
// model.layers.<l>.mlp.experts.<e>.{gate,up,down}_proj.weight (expertName). The not-shared guard
// keeps the two expert classes provably disjoint even if naming drifts; expert bias tensors match
// too, correctly — their add belongs with their GEMM.
func isRoutedExpertWeight(name string) bool {
	return (strings.Contains(name, ".mlp.experts.") || strings.Contains(name, ".ffn.experts.")) && !isSharedExpertWeight(name)
}

// isExpertWeight is the ACCOUNTING predicate: the union of the routed and shared expert classes,
// i.e. every parameter belonging to an MoE expert block. It deliberately does not match the router,
// attention/MLA projections, learned-index projections, or the LM head.
//
// It is the accounting union, NOT the runtime placement predicate: the shared expert's bytes belong
// to the expert block even though it stays device-resident, so MoEResidentWeightBytes keeps its
// byte partition stable. Callers deciding a weight's home must use hostOffloadWeight.
func isExpertWeight(name string) bool {
	return isRoutedExpertWeight(name) || isSharedExpertWeight(name)
}

// hostOffloadWeight is the runtime PLACEMENT predicate splitKernel runs: only the routed experts
// go to host RAM; everything else stays on the device. The shared expert is active on every token,
// so streaming it through the sparse, cold host path would pay a miss latency on every token — it
// is pinned device-resident by construction.
func hostOffloadWeight(name string) bool {
	return isRoutedExpertWeight(name)
}

// CPUOffloadExpertWeight reports whether the canonical tensor name belongs to an MoE expert block
// (routed or shared) — the accounting partition load-time memory planners size against. It is not
// the placement predicate; callers deciding a weight's home must use hostOffloadWeight.
func CPUOffloadExpertWeight(name string) bool {
	return isExpertWeight(name)
}

// glmDsaMatKernel selects the matKernel the GLM-DSA forward runs its GEMMs through:
//   - no Backend, no offload  -> residentKernel (unchanged host default)
//   - no Backend, offload     -> split over resident: bit-exact with residentKernel (degenerate)
//   - Backend, no offload     -> backendKernel (all-device)
//   - Backend, offload        -> split(host=resident, device=backend): the hybrid
func (s *Session) glmDsaMatKernel() matKernel {
	host := matKernel(sessionResidentKernel{s})
	device := host
	if s.Backend != nil {
		device = backendKernel{s}
	}
	if s.CPUOffloadExperts {
		// expertSpillOnHost grades by ExpertSpillLayers when set; unset it is hostOffloadWeight.
		// The shared expert is never a host-offload candidate on either arm.
		return splitKernel{host: host, device: device, onHost: s.expertSpillOnHost()}
	}
	return device
}
