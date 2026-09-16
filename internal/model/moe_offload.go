package model

import "strings"

// moe_offload.go — the CPU-offload hybrid matKernel (the llama.cpp `--n-cpu-moe` equivalent).
//
// A 753B MoE model's parameter count is dominated by its experts: in GLM-5.2 the per-layer
// expert tensors (mlp.experts.<e>.{gate,up,down}_proj) are the overwhelming bulk of the
// weights, while the dense projections (attention q_a/q_b/kv_a/kv_b/o_proj, the learned-index
// projections, the MoE router) and the per-token activation FLOPs are tiny by comparison. At
// Q4_K_M the full model is ~424 GB — far past any single GPU's VRAM — but the host has 1007 GB
// of RAM. So the serve-able split is exactly llama.cpp's `--n-cpu-moe`: keep the ROUTED expert GEMMs
// resident in host RAM (run them on the CPU Q4_K/Q8 kernel) while the DENSE projections + router
// + attention, and the always-on shared expert (active on every token), run on the GPU device
// backend. The routed experts are sparse (only top-k fire per token)
// and bandwidth-, not latency-, bound, so the host is a reasonable home for them; the dense work
// that runs every token on every position stays on the device where the FLOPs are.
//
// fak already had the two endpoints — residentKernel (host) and backendKernel (device) — and the
// GLM-DSA forward already threads ONE matKernel through every GEMM (glm_dsa_session.go). What was
// missing was a kernel that routes PER WEIGHT between the two. splitKernel is that kernel: a thin
// matKernel that dispatches each named matmul to a host kernel or a device kernel by a predicate.
// It introduces NO new arithmetic — a weight runs on whichever sub-kernel it is routed to, exactly
// as that sub-kernel already computes it — so the only thing it can get wrong is the routing, which
// the witnesses pin directly (placement-invariance bit-exactness + a device-shape probe).
type splitKernel struct {
	host   matKernel         // where onHost(name)==true weights run (the offloaded experts)
	device matKernel         // where everything else runs (dense projections, router, attention)
	onHost func(string) bool // true => keep this weight's GEMM on host RAM
}

// prep is the identity, matching both endpoints: residentKernel and backendKernel both return the
// raw f32 activation from prep (neither pre-quantizes the operand at this seam), so a single prep
// produces an operand both sub-kernels' mul accept. Were a sub-kernel to want a different operand
// form, the split would have to prep per side; today it does not, so prep stays a no-op pass.
func (k splitKernel) prep(x []float32) any { return x }

// expertSession resolves the *Session behind this split's DEVICE side, so the routed-expert device
// route stays reachable under --n-cpu-moe (#13128). A split's host side is a residentKernel (no
// session); the device side is where a backend session lives (backendKernel). Returns nil for any
// other device kernel, which keeps the split's host-CPU arm the fail-closed default.
func (k splitKernel) expertSession() *Session {
	if bk, ok := k.device.(backendKernel); ok {
		return bk.s
	}
	return nil
}

// splitDeviceExpertInput is the PER-ENCODING device-kernel predicate the offload split consults
// before pinning a routed expert to host CPU. It reports whether a routed expert's gate/up
// projections can execute on the split's DEVICE backend for this encoding. It is deliberately
// narrower than expertSwiGLUHAL: it asks only whether a device kernel EXISTS for the encoding, so a
// caller can decide placement before staging any bytes.
//
// `name` is the weight name being placed. `onHost` is the split's own placement predicate: a weight
// the split already routes to the device (onHost(name)==false) is a device candidate; a host-routed
// weight is NOT, so the host arm is unchanged.
func (k splitKernel) splitDeviceExpertInput(name string) bool {
	sess := k.expertSession()
	if sess == nil {
		return false
	}
	// The split must not steal a weight it has placed on the host.
	if k.onHost != nil && k.onHost(name) {
		return false
	}
	return sess.supportsRoutedExpertKQuant()
}

// mul routes the named weight to the host or device sub-kernel by the predicate. The chosen
// sub-kernel computes it exactly as it would in an all-host / all-device forward, so the split is
// a pure placement decision over the SAME math.
func (k splitKernel) mul(name string, x any, out, in int) []float32 {
	if k.onHost(name) {
		return k.host.mul(name, x, out, in)
	}
	return k.device.mul(name, x, out, in)
}

// splitHostResidentExperts reports whether the picked experts' batched GEMVs may run on the host,
// when mat is a splitKernel (the --n-cpu-moe hybrid) whose host side is a residentKernel AND every
// one of the picks (gate/up/down) is host-routed by the split's OWN predicate. The function is an
// ADMISSION decision — bool only, no kernel handle — because the caller runs the Model-bound
// hostBatchedGLMExperts(layer, …) primitive directly; that primitive reads m.q4kw/m.kqw by tensor
// name and needs no kernel (the split's host residentKernel would be discarded anyway, which is
// exactly what both production call sites already did). It returns false for any other kernel, for
// a non-resident host, and — critically — for a GRADED split (ExpertSpillLayers) where any pick is
// device-kept: the batched host primitive only models host-resident experts, so a mixed layer must
// DECLINE to the per-expert loop rather than silently steal device-kept experts to the host (the
// exact placement the operator's `--n-cpu-moe N` did NOT budget). This unifies the two
// batched-expert fast paths with the split placement: without it a splitKernel misses both the
// concrete `mat.(residentKernel)` assertion (glmMoeFFN.apply) and the type switch (moeFFN.apply), so
// the measured ~1.8x batched primitive is unreachable on the offload-hybrid path (#12972).
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
	if _, ok := k.host.(residentKernel); !ok {
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

// sparseAttend forwards GLM-DSA's sparse-attention compute to the DEVICE side: sparse attention is
// dense per-head softmax(scale·q·k)·V over the host-selected keys — attention work, not an expert —
// so it belongs with the dense projections on the GPU. glmDsaAttendCached type-asserts the active
// matKernel for dsaSparseKernel; implementing it here keeps the device's sparse-attention seam
// reachable through the split. When the device kernel does not advertise it (e.g. the no-backend
// degenerate split where device==host==residentKernel), this returns ok=false and the caller keeps
// the byte-for-byte host softmax/ΣwV loop — which is what makes the no-backend split bit-exact with
// the plain host forward.
func (k splitKernel) sparseAttend(q, selK, selV []float32, nSel, nH, qkHead, vHead int, scale float32) ([]float32, bool) {
	if dk, ok := k.device.(dsaSparseKernel); ok {
		return dk.sparseAttend(q, selK, selV, nSel, nH, qkHead, vHead, scale)
	}
	return nil, false
}

// indexSelect forwards GLM-DSA's learned-indexer score + top-k SELECTION to the DEVICE side, the
// symmetric partner of sparseAttend above: hostOffloadWeight keeps the learned-index PROJECTIONS on the
// device under a `--n-cpu-moe` split (only the routed expert bulk goes host-resident), so the selection
// COMPUTE they feed belongs on the device too — otherwise the offload hybrid would silently keep the
// indexer host-resident while every other DSA op runs on the kernel. glmDsaIndexStep type-asserts the
// active matKernel for dsaIndexKernel; routing through here keeps k_dsa_index_score + k_dsa_index_topk
// reachable through the split. When the device kernel does not advertise DSAIndexBackend (the
// no-backend degenerate split where device==host==residentKernel), this returns ok=false and the
// caller keeps the host f64 score+top-k loop — which is what keeps the no-backend split selection-exact
// with the plain host forward.
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
	return strings.Contains(name, ".mlp.shared_experts.")
}

// isRoutedExpertWeight matches the PER-TOKEN ROUTED experts (only the router-selected top-k fire per
// token): model.layers.<l>.mlp.experts.<e>.{gate,up,down}_proj.weight (expertName, moe.go). The
// substring ".mlp.experts." names them; the "but not shared" guard is belt-and-suspenders — the shared
// form is ".mlp.shared_experts." so ".mlp.experts." never matches it — but it keeps the two classes
// provably disjoint even if the naming ever drifts. A bias tensor of an expert
// (…experts.<e>.gate_proj.bias) also matches, which is correct — its add belongs with its GEMM.
func isRoutedExpertWeight(name string) bool {
	return strings.Contains(name, ".mlp.experts.") && !isSharedExpertWeight(name)
}

// isExpertWeight is the ACCOUNTING predicate: the union of the two disjoint expert classes
// (isRoutedExpertWeight, isSharedExpertWeight), i.e. every parameter that belongs to an MoE expert
// BLOCK. It deliberately does NOT match the router (mlp.gate.weight), the attention/MLA projections,
// the learned-index projections, or the LM head: those are the small, every-token dense GEMMs that
// stay on the device. The names are the canonical HF tensor names the GLM-DSA forward reads:
//
//	routed experts: model.layers.<l>.mlp.experts.<e>.{gate,up,down}_proj.weight  (expertName, moe.go)
//	shared experts: model.layers.<l>.mlp.shared_experts.{gate,up,down}_proj.weight (glmSharedExperts)
//
// It is the ACCOUNTING/classification union, NOT the runtime PLACEMENT predicate. The two differ by
// design (see hostOffloadWeight): under offload the routed experts move to host RAM while the
// always-on shared expert stays DEVICE-resident, but the shared expert's bytes still belong to the
// expert block, so the union keeps the byte partitions stable: MoEResidentWeightBytes still counts
// the routed experts against the per-layer expert term while the shared expert falls into the
// replicated remainder (the device base the spill sizing reserves), which is EXACTLY the placement
// hostOffloadWeight implements. Keeping this union intact is what makes that accounting stable across
// the placement change. Callers that decide a weight's HOME must use hostOffloadWeight.
func isExpertWeight(name string) bool {
	return isRoutedExpertWeight(name) || isSharedExpertWeight(name)
}

// hostOffloadWeight is the runtime PLACEMENT predicate splitKernel runs: it sends only the ROUTED
// experts (.mlp.experts.<e>.*) to host RAM and pins everything else — the always-on shared expert,
// the router, attention/MLA projections, learned-index projections, the LM head and every dense
// weight — to the device. It is the routed-only arm of isExpertWeight.
//
// The distinction from isExpertWeight is the point of #1304. The shared expert is active on EVERY
// token (hit rate 1.0), so it is resident-hot by construction; placing it on the same host path as
// the sparse, cold routed experts streams it out and pays a needless miss latency on every token.
// The shared expert is device-resident, so routing it to host is simply wrong. isExpertWeight stays
// the union (accounting: the shared expert's bytes still belong to the expert block, and
// CPUOffloadExpertWeight must keep partitioning exactly as the load-time planner expects); this
// predicate is the routed-only placement half.
func hostOffloadWeight(name string) bool {
	return isRoutedExpertWeight(name)
}

// CPUOffloadExpertWeight reports whether the canonical tensor name belongs to an MoE expert BLOCK
// (routed OR shared) — the ACCOUNTING partition load-time memory planners size against. It is the
// union isExpertWeight, deliberately NOT the runtime placement predicate hostOffloadWeight: the
// runtime keeps the always-on shared expert DEVICE-resident on EVERY token (#1304), while this
// accounting still charges the shared expert's bytes to the expert block. A caller deciding a
// weight's HOME (host vs device) must use hostOffloadWeight, not this.
func CPUOffloadExpertWeight(name string) bool {
	return isExpertWeight(name)
}

// glmDsaMatKernel selects the matKernel the GLM-DSA forward runs its dense GEMMs through, in one
// place, so decodeBandGLMDsa stays a single instruction stream. The four cases:
//   - no Backend, no offload    -> residentKernel (the host path; byte-for-byte unchanged default)
//   - no Backend, offload       -> split(host=resident, device=resident): every GEMM still on host,
//     so the forward is BIT-EXACT with residentKernel — the
//     placement-invariance witness, and the honest degenerate config
//     (offload requested but no device => nothing actually moves).
//   - Backend, no offload       -> backendKernel (the all-device path; unchanged)
//   - Backend, offload          -> split(host=resident, device=backend): THE HYBRID — expert GEMMs
//     on host RAM, dense projections + router + attention on the device.
//
// CPUOffloadExperts gates the split; with no Backend it costs nothing and proves the split is a pure
// placement decision (it computes the identical bytes the plain host forward does).
func (s *Session) glmDsaMatKernel() matKernel {
	host := matKernel(residentKernel{s.M})
	device := host
	if s.Backend != nil {
		device = backendKernel{s}
	}
	if s.CPUOffloadExperts {
		// The predicate is GRADED by ExpertSpillLayers (#5612, expert_spill_placement.go): unset (the
		// default) it is exactly hostOffloadWeight, so this line is byte-for-byte the ungraded split
		// over ROUTED experts; set to N it spills only the first N MoE layers' routed experts and
		// leaves the rest on the device, where R0's ring (#5611) bounds what they may hold resident.
		// The always-on shared expert is pinned DEVICE-resident on both arms (#1304): it is never a
		// host-offload candidate, so it is never streamed as a miss.
		return splitKernel{host: host, device: device, onHost: s.expertSpillOnHost()}
	}
	return device
}
