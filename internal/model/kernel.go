package model

import "github.com/anthony-chaudhary/fak/internal/compute"

const glmDsaLargeScratchPoolKeepBytes = 1 << 20

type largeScratchTrimmer interface {
	TrimLarge(maxKeepBytes int)
}

func trimLargeScratchPool(be compute.Backend) {
	if t, ok := be.(largeScratchTrimmer); ok {
		t.TrimLarge(glmDsaLargeScratchPoolKeepBytes)
	}
}

// matKernel selects which arithmetic a single-position block's weight matmuls run
// through, so the f32 decode block (tokenHidden) and its Q8 twin (tokenHiddenQ) share
// ONE block skeleton — blockStep — instead of two hand-copied transcriptions. prep
// converts a (just-normed) activation into the kernel's preferred operand form ONCE,
// and mul applies a named weight to that prepared operand. The f32 kernel's prep is
// the identity, so its instruction stream is byte-for-byte today's parMatRows path;
// the Q8 kernel's prep quantizes the activation a single time and the q/k/v (and
// gate/up) matmuls reuse it, exactly as the hand-copied tokenHiddenQ did — so neither
// reduction order moves and the bit-equality / Q8 gates both hold.
type matKernel interface {
	// prep turns a normed activation vector into the operand the kernel multiplies
	// against weights (an f32 slice for f32, a quantized vector for Q8). It is called
	// once per distinct activation; the returned handle feeds every mul that shares it.
	prep(x []float32) any
	// mul applies the named weight (out×in, row-major) to a prepared activation handle.
	mul(name string, x any, out, in int) []float32
}

// f32Kernel runs the single-position block in f32: prep is the identity and every
// matmul routes through parMatRows (the row-parallel, in-order fdot GEMV), preserving
// the exact reduction order the bit-exact f32 rungs (R2/R14) certify.
type f32Kernel struct{ m *Model }

func (k f32Kernel) prep(x []float32) any { return x }
func (k f32Kernel) mul(name string, x any, out, in int) []float32 {
	return parMatRows(k.m.tensor(name), x.([]float32), out, in)
}

// q8Kernel runs the single-position block in Q8_0 with an independent q8Vec for each
// prep and qMatRows against the prebuilt Q8_0 weight. The fresh operand allocation is
// intentional: it remains safe for nested callers such as MoE experts that need an
// earlier prepared operand after a later prep.
type q8Kernel struct{ m *Model }

func (k q8Kernel) prep(x []float32) any { return quantizeVecQ8(x) }
func (k q8Kernel) mul(name string, x any, out, in int) []float32 {
	return qMatRows(k.m.q8(name), x.(q8Vec))
}

// sessionQ8Kernel is the allocation-light Q8 fallback kernel for single-position block
// paths where prepared operands are consumed before the next prep. It reuses the same
// session-owned activation quant buffer as headQ and the fast PreNorm decode path.
type sessionQ8Kernel struct{ s *Session }

func (k sessionQ8Kernel) prep(x []float32) any { return k.s.quantizeVecQ8(x) }
func (k sessionQ8Kernel) mul(name string, x any, out, in int) []float32 {
	return qMatRows(k.s.M.q8(name), x.(q8Vec))
}

// residentKernel is the cacheless Forward kernel: prefer the exact f32 tensor when it
// exists, otherwise read the quant-on-load Q8_0 resident copy. That keeps ordinary
// Forward bit-identical while allowing lean models to run without re-inflating dropped
// projection weights.
type residentKernel struct{ m *Model }

func (k residentKernel) prep(x []float32) any { return x }
func (k residentKernel) mul(name string, x any, out, in int) []float32 {
	return k.m.residentMatRows(name, x.([]float32), out, in)
}

// backendKernel routes matKernel GEMMs through a compute.Backend (the GPU pure kernels) instead of
// the host residentMatRows. It is the GLM-MoE-DSA forward's device path (#86, partial): the heavy
// dense GEMMs — MoE/FFN experts + router, and the attention projections — run on the backend, while
// the DSA index-scoring + sparse-attention glue and the KV cache stay host-resident. The activation
// stays host-side (prep returns x); mul uploads the resident weight (q8 -> k_q8_gemm, pure; f32 ->
// the backend's f32 GEMM) and the activation, runs be.MatMul, and reads the result back to host, so
// it is a drop-in for residentMatRows in the host data flow. The device↔host copy per GEMM keeps the
// glue simple (correctness-first); a fully device-resident GLM-DSA forward is the next slice.
type backendKernel struct{ s *Session }

func (k backendKernel) prep(x []float32) any { return x }
func (k backendKernel) mul(name string, x any, out, in int) []float32 {
	s := k.s
	be := s.Backend
	xf := x.([]float32)
	if len(xf) != in {
		panic("model: backendKernel " + name + " activation length mismatch")
	}
	wt := s.glmDsaWeightHAL(name, out, in)
	xt := uploadHostF32Class(be, []int{in}, xf, compute.MemoryActivation, "glm-dsa-activation "+name)
	y := be.Read(be.MatMul(wt, xt))
	// The activation upload has a FRESH host pointer every token, so compute.uploadCache (which
	// shares only STABLE weight pointers) can never reuse it. Evict it now — otherwise a multi-token
	// Prefill accumulates one resident device buffer per position per GEMM until cudaMalloc fails
	// (the glmdsatput "cuda device allocation failed" on the 512-token prefill). The resident weight
	// wt stays cached. No-op on cpu-ref (Upload is identity, Free is a no-op).
	be.Free(xt)
	if len(y) != out {
		panic("model: backendKernel " + name + " result length mismatch")
	}
	return y
}

// dsaSparseKernel is the OPTIONAL device path for GLM-MoE-DSA's sparse attention compute
// (the per-head softmax(scale·q·k)·V over the host-selected keys). glmDsaAttendCached
// type-asserts its matKernel for it and falls back to the host loop when absent — so only a
// backendKernel whose compute.Backend implements compute.DSASparseBackend (the cpu-ref, and
// the cuda backend via k_dsa_sparse_attend) takes the device path; residentKernel and a
// backend without the capability leave the attention math byte-for-byte host-resident. The
// KEY SELECTION (index scores + top-k) is computed host-side and handed in via the gathered
// selK/selV, so the device's only divergence is the f32 reduction order over the same keys.
type dsaSparseKernel interface {
	// sparseAttend runs the device sparse attention over nSel gathered selected keys/values
	// for one query position. q is [nH*qkHead]; selK [nSel,nH*qkHead]; selV [nSel,nH*vHead].
	// It returns the [nH*vHead] attnConcat and ok=false (caller falls back to host) when the
	// backend does not advertise compute.DSASparseBackend.
	sparseAttend(q, selK, selV []float32, nSel, nH, qkHead, vHead int, scale float32) ([]float32, bool)
}

// sparseAttend routes GLM-DSA's sparse attention to the compute.Backend when it advertises
// DSASparseBackend: it uploads the query + the host-gathered selected K/V rows, runs the
// device op (q8/f32 -> k_dsa_sparse_attend on the cuda backend, the pure GPU kernel), and
// reads the [nH*vHead] result back. A backend without the capability returns ok=false so the
// caller keeps the host loop. Like backendKernel.mul this copies host↔device per call
// (correctness-first); a fully device-resident DSA cache is the next slice.
func (k backendKernel) sparseAttend(q, selK, selV []float32, nSel, nH, qkHead, vHead int, scale float32) ([]float32, bool) {
	be := k.s.Backend
	sb, ok := be.(compute.DSASparseBackend)
	if !ok {
		return nil, false
	}
	qt := uploadHostF32Class(be, []int{nH * qkHead}, q, compute.MemoryActivation, "glm-dsa-sparse-query")
	kt := uploadHostF32Class(be, []int{nSel * nH * qkHead}, selK, compute.MemoryActivation, "glm-dsa-sparse-selected-k")
	vt := uploadHostF32Class(be, []int{nSel * nH * vHead}, selV, compute.MemoryActivation, "glm-dsa-sparse-selected-v")
	out := be.Read(sb.DSASparseAttend(qt, kt, vt, nSel, nH, qkHead, vHead, scale))
	// Per-call gathered operands with fresh host pointers (see backendKernel.mul) — evict them so a
	// multi-token Prefill does not accumulate one resident device copy per position. Read fenced.
	be.Free(qt)
	be.Free(kt)
	be.Free(vt)
	trimLargeScratchPool(be)
	if len(out) != nH*vHead {
		panic("model: backendKernel sparseAttend result length mismatch")
	}
	return out, true
}

// dsaIndexKernel is the OPTIONAL device path for GLM-MoE-DSA's learned-indexer score + top-k
// SELECTION — the last GLM-5.2 compute that stays host-resident even after the dense projections
// AND the sparse-attention compute already run on the kernel. glmDsaIndexStep (the per-token DECODE
// hot path, via decodeBandGLMDsa) type-asserts its matKernel for it and falls back to the host f64
// score+top-k loop when absent — so only a backendKernel whose compute.Backend implements
// compute.DSAIndexBackend (the cpu-ref, and the cuda backend via k_dsa_index_score + k_dsa_index_topk)
// takes the device path. Because the device accumulates the score dot in f64 (matching the host
// reduction over the same losslessly-widened-f32 operands) and selects with the identical total
// order, the selected key SET is bit-identical CPU↔device — the selection-stability boundary that
// previously kept this host-side is satisfied, not bypassed. (The whole-sequence Prefill index
// helper glmDsaTopKIndicesNormed runs host-only and is the labeled residual for a later slice.)
type dsaIndexKernel interface {
	// indexSelect scores nKeys cached index-keys against one query's nH index-heads and returns the
	// top-k selected key positions for the query at queryPos. indexQ is [nH*indexDim]; indexK is
	// [nKeys*indexDim]; weights is [nH] (already scaled). It returns ok=false (caller falls back to
	// the host loop) when the backend does not advertise compute.DSAIndexBackend.
	indexSelect(indexQ, indexK, weights []float32, nKeys, nH, indexDim, queryPos, topK int, scale float32) ([]int, bool)
}

// indexSelect routes GLM-DSA's indexer score + top-k to the compute.Backend when it advertises
// DSAIndexBackend: it uploads the per-head query, the cached index-keys, and the per-head weights,
// runs the device op (k_dsa_index_score + k_dsa_index_topk on the cuda backend — f64-accumulated
// scoring, on-device selection), and returns the selected positions. A backend without the
// capability returns ok=false so the caller keeps the host f64 loop. The selection is bit-identical
// to the host because the device scores in f64 and uses the same total order.
func (k backendKernel) indexSelect(indexQ, indexK, weights []float32, nKeys, nH, indexDim, queryPos, topK int, scale float32) ([]int, bool) {
	be := k.s.Backend
	ib, ok := be.(compute.DSAIndexBackend)
	if !ok {
		return nil, false
	}
	if nKeys <= 0 || topK <= 0 || len(indexQ) != nH*indexDim || len(indexK) != nKeys*indexDim || len(weights) != nH {
		return nil, false
	}
	qt := uploadHostF32Class(be, []int{nH * indexDim}, indexQ, compute.MemoryActivation, "glm-dsa-index-query")
	kt := uploadHostF32Class(be, []int{nKeys * indexDim}, indexK, compute.MemoryActivation, "glm-dsa-index-keys")
	wt := uploadHostF32Class(be, []int{nH}, weights, compute.MemoryActivation, "glm-dsa-index-weights")
	sel := ib.DSAIndexSelect(qt, kt, wt, nKeys, nH, indexDim, queryPos, topK, scale)
	// Per-call operands with fresh host pointers — evict so a multi-token Prefill does not accumulate
	// a resident device copy per position. DSAIndexSelect returns a host []int, so it has fenced.
	be.Free(qt)
	be.Free(kt)
	be.Free(wt)
	return sel, true
}

// glmDsaWeightHAL uploads a resident GLM projection weight to the backend (cached in halW), mirroring
// residentMatRows's dtype dispatch on the upload side: a q8-resident weight uploads codes+scales
// (k_q8_gemm, pure — needs cuda.go uploadQ8Resident), an f32 weight uploads f32 (backend f32 GEMM).
func (s *Session) glmDsaWeightHAL(name string, out, in int) compute.Tensor {
	if qt, ok := s.M.q8w[name]; ok {
		return s.weightHALQ8(name, qt)
	}
	if qt, ok := s.M.q4kw[name]; ok {
		return s.weightHALQ4K(name, qt) // raw Q4_K super-blocks -> k_q4k_gemm (cuda) / cpu-ref Q4_K
	}
	if s.M.has(name) {
		return s.weightHAL(name)
	}
	panic("model: glmDsaWeightHAL missing resident weight " + name + " (no f32/q8/q4_k residency)")
}

// residentMatRows reads a projection by name from either the ordinary f32 store
// or the memory-lean Q8 store. It is used by architecture-specific reference
// paths such as GLM-MoE-DSA, where the cacheless/session math is still CPU f32
// but large projection weights may have been quantized at load and dropped from
// raw f32 residency.
func (m *Model) residentMatRows(name string, x []float32, out, in int) []float32 {
	if m.has(name) {
		return matRows(m.tensor(name), x, out, in)
	}
	if qt := m.q8w[name]; qt != nil {
		if qt.out != out || qt.in != in {
			panic("model: q8 tensor shape mismatch: " + name)
		}
		return qMatRows(qt, quantizeVecQ8(x))
	}
	if qt := m.q4w[name]; qt != nil {
		if qt.out != out || qt.in != in {
			panic("model: int4 tensor shape mismatch: " + name)
		}
		return q4MatRows(qt, x)
	}
	if qt := m.q4kw[name]; qt != nil {
		if qt.out != out || qt.in != in {
			// Print stored-vs-requested so a residency/config-dim bug names its operands
			// (e.g. a MoE expert stored [moe_intermediate,H] but the forward asked
			// [intermediate,H] because expert_feed_forward_length was not read into
			// cfg.MoEIntermediateSize). The bare name alone can't distinguish those.
			panic("model: resident Q4_K tensor shape mismatch: " + name +
				" stored=[" + itoa(qt.out) + "," + itoa(qt.in) + "]" +
				" requested=[" + itoa(out) + "," + itoa(in) + "]")
		}
		return q4kMatRows(qt, x)
	}
	if qt := m.kqw[name]; qt != nil {
		if qt.out != out || qt.in != in {
			panic("model: resident k-quant tensor shape mismatch: " + name +
				" stored=[" + itoa(qt.out) + "," + itoa(qt.in) + "]" +
				" requested=[" + itoa(out) + "," + itoa(in) + "]")
		}
		return kQuantMatRows(qt, x)
	}
	if qt := m.gptqw[name]; qt != nil {
		if qt.out != out || qt.in != in {
			panic("model: GPTQ tensor shape mismatch: " + name)
		}
		return gptqMatRows(qt, x)
	}
	panic("model: missing tensor " + name)
}
