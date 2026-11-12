package model

// Chunkwise Gated-DeltaNet state/readout schedule adapted from vLLM at
// 5c8ae7c74aaa99d1f7fbeea942fd7282e3413810 (itself adapted from SGLang).
// Copyright contributors to the vLLM project.
// SPDX-License-Identifier: Apache-2.0
// See THIRD_PARTY_NOTICES.md and qwen35_gdn_chunked_host.README.md.
//
// This portable F32 adaptation is not vLLM's BF16/AMX kernel. It preserves
// Qwen's rectangular state, decay-before-delta, and distinct query readout.
// Chunkwise reduction is deliberately not a bitwise replacement for decode.

import "math"

const (
	qwen35GDNHostChunk                 = 64
	qwen35GDNHostMaxTokens             = 4096
	qwen35GDNHostMaxHeads              = 64
	qwen35GDNHostMaxWidth              = 256
	qwen35GDNHostMaxWorkers            = 16
	qwen35GDNHostMaxPayloadBytes int64 = 64 << 20
)

// qwen35GDNHostMode is an invocation-local diagnostic restriction, not a
// process-global selector. Only the typed build qualification may enable the
// candidate. The scalar mode allows an identical batched-path oracle without
// changing projection, convolution, attention, or unrelated qualification tags.
type qwen35GDNHostMode uint8

const (
	qwen35GDNHostBuildSelected qwen35GDNHostMode = iota
	qwen35GDNHostForceScalar
)

func qwen35GDNHostModeAllows(mode qwen35GDNHostMode) bool {
	return mode == qwen35GDNHostBuildSelected
}

type qwen35GDNHostInput struct {
	tokens, keyHeads, valueHeads, keyWidth, valueWidth int
	q, k, conv, a, b, aExp, dtBias                     []float32
	state                                              [][]float32
}

type qwen35GDNHostResult struct {
	core  []float32
	state [][]float32
}

// qwen35GDNHostPlan refuses before allocation/indexing. The bound counts all
// candidate float payload (output, final state, worker scratch), not the already
// live projection panels, allocator headers, or the runtime's goroutine stacks.
func qwen35GDNHostPlan(in qwen35GDNHostInput, workers int) (int, int, bool) {
	p, nk, nv, kd, vd := in.tokens, in.keyHeads, in.valueHeads, in.keyWidth, in.valueWidth
	if p < qwen35GDNHostChunk || p > qwen35GDNHostMaxTokens ||
		nk < 1 || nv < 1 || nk > nv || nv > qwen35GDNHostMaxHeads || nv%nk != 0 ||
		kd < 1 || vd < 1 || kd > qwen35GDNHostMaxWidth || vd > qwen35GDNHostMaxWidth {
		return 0, 0, false
	}
	// Products are bounded above before multiplication, including on 32-bit Go.
	keyDim, valDim := nk*kd, nv*vd
	convDim := 2*keyDim + valDim
	if len(in.q) != p*keyDim || len(in.k) != p*keyDim ||
		len(in.conv) != p*convDim || len(in.a) != p*nv || len(in.b) != p*nv ||
		len(in.aExp) != nv || len(in.dtBias) != nv || len(in.state) != nv {
		return 0, 0, false
	}
	for _, st := range in.state {
		if len(st) != kd*vd {
			return 0, 0, false
		}
	}
	if workers < 1 {
		workers = 1
	}
	if workers > nv {
		workers = nv
	}
	if workers > qwen35GDNHostMaxWorkers {
		workers = qwen35GDNHostMaxWorkers
	}
	const c = qwen35GDNHostChunk
	// decay/KK/QK, packed Q/K, K*S0/Q*S0/Vnew, gate/beta/prefix.
	scratch := 3*c*c + 2*c*kd + 3*c*vd + 3*c
	floats := int64(p)*int64(valDim) + int64(nv)*int64(kd)*int64(vd) + int64(workers)*int64(scratch)
	if floats > qwen35GDNHostMaxPayloadBytes/4 {
		return 0, 0, false
	}
	return workers, scratch, true
}

func qwen35GDNHostFinite(xs []float32) bool {
	for _, x := range xs {
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
			return false
		}
	}
	return true
}

// qwen35GDNHostCompute never changes any borrowed input. All heads/chunks join
// before return. False means no candidate state or output was published, even
// if an intermediate overflow was discovered in the last chunk of the last head.
func qwen35GDNHostCompute(in qwen35GDNHostInput, workers int) (qwen35GDNHostResult, bool) {
	workers, scratchSize, ok := qwen35GDNHostPlan(in, workers)
	if !ok {
		return qwen35GDNHostResult{}, false
	}
	for _, xs := range [][]float32{in.q, in.k, in.conv, in.a, in.b, in.aExp, in.dtBias} {
		if !qwen35GDNHostFinite(xs) {
			return qwen35GDNHostResult{}, false
		}
	}
	for h, st := range in.state {
		if in.aExp[h] < 0 || !qwen35GDNHostFinite(st) {
			return qwen35GDNHostResult{}, false
		}
	}
	nv, kd, vd := in.valueHeads, in.keyWidth, in.valueWidth
	out := qwen35GDNHostResult{core: make([]float32, in.tokens*nv*vd), state: make([][]float32, nv)}
	states := make([]float32, nv*kd*vd)
	for h := 0; h < nv; h++ {
		out.state[h] = states[h*kd*vd : (h+1)*kd*vd]
		copy(out.state[h], in.state[h])
	}
	// One workspace per explicitly bounded worker, reused across all its heads
	// and chunks. A worker only writes its own output/state and failure slot.
	work := make([]float32, workers*scratchSize)
	failed := make([]bool, workers)
	parFor(workers, workers, func(lo, hi int) {
		for w := lo; w < hi; w++ {
			scratch := work[w*scratchSize : (w+1)*scratchSize]
			for h := w; h < nv; h += workers {
				if !qwen35GDNHostHead(in, h, out.core, out.state[h], scratch) {
					failed[w] = true
					break
				}
			}
		}
	})
	for _, bad := range failed {
		if bad {
			return qwen35GDNHostResult{}, false
		}
	}
	return out, true
}

// qwen35GDNHostHead applies the donor's causal triangular solve, readout, and
// final-state update. Fusing the inverse's application avoids materializing W/U:
// Vnew_i = beta_i*(V_i - prefix_i*K_i*S0
//   - sum_{j<i} decay(i,j)*(K_i*K_j)*Vnew_j).
//
// Then O_i=prefix_i*Q_i*S0+sum_{j<=i}decay(i,j)*(Q_i*K_j)*Vnew_j.
// State is updated once per chunk. It is PRIVATE until the entire call succeeds.
func qwen35GDNHostHead(in qwen35GDNHostInput, head int, out, state, scratch []float32) bool {
	const c = qwen35GDNHostChunk
	p, kd, vd := in.tokens, in.keyWidth, in.valueWidth
	keyDim, valDim := in.keyHeads*kd, in.valueHeads*vd
	convDim := 2*keyDim + valDim
	kh := head / (in.valueHeads / in.keyHeads)
	take := func(n int) []float32 { x := scratch[:n]; scratch = scratch[n:]; return x }
	decay, kk, qk := take(c*c), take(c*c), take(c*c)
	q, k := take(c*kd), take(c*kd)
	ks, qs, u := take(c*vd), take(c*vd), take(c*vd)
	gate, beta, prefix := take(c), take(c), take(c)
	for base := 0; base < p; base += c {
		n := p - base
		if n > c {
			n = c
		}
		for i := 0; i < n; i++ {
			t := base + i
			copy(q[i*kd:(i+1)*kd], in.q[t*keyDim+kh*kd:t*keyDim+(kh+1)*kd])
			copy(k[i*kd:(i+1)*kd], in.k[t*keyDim+kh*kd:t*keyDim+(kh+1)*kd])
			beta[i] = sigmoidf(in.b[t*in.valueHeads+head])
			dt := softplus(in.a[t*in.valueHeads+head] + in.dtBias[head])
			gate[i] = float32(math.Exp(float64(-in.aExp[head] * dt)))
			if !qwen35GDNHostFinite(gate[i:i+1]) || !qwen35GDNHostFinite(beta[i:i+1]) ||
				gate[i] < 0 || gate[i] > 1 || beta[i] < 0 || beta[i] > 1 {
				return false
			}
			prefix[i] = gate[i]
			if i != 0 {
				prefix[i] *= prefix[i-1]
			}
			// Products of the EXISTING rounded scalar decay include exact-zero
			// resets without log(0), division by a vanished prefix, or 0/0.
			decay[i*c+i] = 1
			for j := i - 1; j >= 0; j-- {
				decay[i*c+j] = decay[i*c+j+1] * gate[j+1]
			}
		}
		// Pack once; two lower-triangular Gram panels reuse each key row.
		for i := 0; i < n; i++ {
			for j := 0; j <= i; j++ {
				var a, b float32
				for d := 0; d < kd; d++ {
					a = gdnFMA32(k[i*kd+d], k[j*kd+d], a)
					b = gdnFMA32(q[i*kd+d], k[j*kd+d], b)
				}
				kk[i*c+j], qk[i*c+j] = a, b
			}
			for d := 0; d < vd; d++ {
				var a, b float32
				for z := 0; z < kd; z++ {
					a = gdnFMA32(k[i*kd+z], state[z*vd+d], a)
					b = gdnFMA32(q[i*kd+z], state[z*vd+d], b)
				}
				ks[i*vd+d], qs[i*vd+d] = a, b
			}
		}
		// Forward-substitute the unit-lower-triangular system. No future token
		// can contribute, including in the final partial chunk.
		for i := 0; i < n; i++ {
			v := in.conv[(base+i)*convDim+2*keyDim+head*vd : (base+i)*convDim+2*keyDim+(head+1)*vd]
			for d := 0; d < vd; d++ {
				x := beta[i] * (v[d] - prefix[i]*ks[i*vd+d])
				for j := 0; j < i; j++ {
					coef := beta[i] * decay[i*c+j] * kk[i*c+j]
					x = gdnFMA32(-coef, u[j*vd+d], x)
				}
				u[i*vd+d] = x
			}
		}
		for i := 0; i < n; i++ {
			dst := out[(base+i)*valDim+head*vd : (base+i)*valDim+(head+1)*vd]
			for d := 0; d < vd; d++ {
				x := prefix[i] * qs[i*vd+d]
				for j := 0; j <= i; j++ {
					x = gdnFMA32(decay[i*c+j]*qk[i*c+j], u[j*vd+d], x)
				}
				dst[d] = x
			}
			if !qwen35GDNHostFinite(dst) {
				return false
			}
		}
		for z := 0; z < kd; z++ {
			for d := 0; d < vd; d++ {
				x := prefix[n-1] * state[z*vd+d]
				for j := 0; j < n; j++ {
					x = gdnFMA32(decay[(n-1)*c+j]*k[j*kd+z], u[j*vd+d], x)
				}
				state[z*vd+d] = x
			}
		}
		if !qwen35GDNHostFinite(state) {
			return false
		}
	}
	return true
}

// Only the ordinary builtin host producers grant this provenance. Callback,
// Metal/HAL, Prism, tapped/captured and vectorized-decode opt-ins retain their
// existing route. The default build constant folds this selection away.
func (s *Session) qwen35GDNHostSelected(ordinary bool, vectorized bool, tokens int) bool {
	return qwen35GDNChunkedHostQualification && ordinary && !vectorized &&
		tokens >= qwen35GDNHostChunk && s != nil && s.M != nil &&
		!s.Metal && !s.MetalQ4K && s.Backend == nil && s.M.prism == nil &&
		!s.captureTargetHidden && !s.M.Cfg.EnableResidualHook && s.activeTap() == nil
}
