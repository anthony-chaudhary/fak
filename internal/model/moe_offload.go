package model

import "strings"

// moe_offload.go — the CPU-offload hybrid matKernel (the llama.cpp `--n-cpu-moe` equivalent).
//
// A 753B MoE model's parameter count is dominated by its experts: in GLM-5.2 the per-layer
// expert tensors (mlp.experts.<e>.{gate,up,down}_proj) are the overwhelming bulk of the
// weights, while the dense projections (attention q_a/q_b/kv_a/kv_b/o_proj, the learned-index
// projections, the MoE router) and the per-token activation FLOPs are tiny by comparison. At
// Q4_K_M the full model is ~424 GB — far past any single GPU's VRAM — but the host has 1007 GB
// of RAM. So the serve-able split is exactly llama.cpp's `--n-cpu-moe`: keep the EXPERT GEMMs
// resident in host RAM (run them on the CPU Q4_K/Q8 kernel) while the DENSE projections + router
// + attention run on the GPU device backend. The experts are sparse (only top-k fire per token)
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

// mul routes the named weight to the host or device sub-kernel by the predicate. The chosen
// sub-kernel computes it exactly as it would in an all-host / all-device forward, so the split is
// a pure placement decision over the SAME math.
func (k splitKernel) mul(name string, x any, out, in int) []float32 {
	if k.onHost(name) {
		return k.host.mul(name, x, out, in)
	}
	return k.device.mul(name, x, out, in)
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
// symmetric partner of sparseAttend above: isExpertWeight keeps the learned-index PROJECTIONS on the
// device under a `--n-cpu-moe` split (only the expert bulk goes host-resident), so the selection
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

// isExpertWeight is the default CPU-offload predicate: it selects the MoE expert and shared-expert
// projection weights — the parameter bulk a `--n-cpu-moe` split sends to host RAM — and nothing else.
// It deliberately does NOT match the router (mlp.gate.weight), the attention/MLA projections, the
// learned-index projections, or the LM head: those are the small, every-token dense GEMMs that stay
// on the device. The names are the canonical HF tensor names the GLM-DSA forward reads:
//
//	routed experts: model.layers.<l>.mlp.experts.<e>.{gate,up,down}_proj.weight  (expertName, moe.go)
//	shared experts: model.layers.<l>.mlp.shared_experts.{gate,up,down}_proj.weight (glmSharedExperts)
//
// so the substrings ".mlp.experts." and ".mlp.shared_experts." partition expert weights from every
// dense weight exactly (the router mlp.gate.weight contains neither). A bias tensor of an expert
// (…experts.<e>.gate_proj.bias) also matches, which is correct — its add belongs with its GEMM.
func isExpertWeight(name string) bool {
	return strings.Contains(name, ".mlp.experts.") || strings.Contains(name, ".mlp.shared_experts.")
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
		return splitKernel{host: host, device: device, onHost: isExpertWeight}
	}
	return device
}
