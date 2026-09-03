package model

import (
	"fmt"
)

// StepGLM5NextKDAHead performs the delta update for a single head matrix S (shape Dk x Dv):
// 1. Decay: S = alpha * S
// 2. Predict: vHat_j = sum_i S_{i, j} * k_i
// 3. Error delta: delta_j = v_j - vHat_j
// 4. Update: S_{i, j} += beta * k_i * delta_j
// 5. Output: o_j = sum_i S_{i, j} * q_i
func StepGLM5NextKDAHead(S []float32, Dk, Dv int, q, k, v []float32, alpha, beta float32) []float32 {
	if len(S) != Dk*Dv {
		panic(fmt.Sprintf("model: KDA head state size %d != Dk*Dv (%d*%d)", len(S), Dk, Dv))
	}
	if len(q) != Dk || len(k) != Dk || len(v) != Dv {
		panic(fmt.Sprintf("model: KDA head vector sizes mismatch (q=%d, k=%d, v=%d; want Dk=%d, Dv=%d)", len(q), len(k), len(v), Dk, Dv))
	}

	// 1. In-place decay
	if alpha != 1.0 {
		for idx := range S {
			S[idx] *= alpha
		}
	}

	// 2. Predict vHat = S^T * k: vHat[j] = sum_i S[i*Dv + j] * k[i]
	vHat := make([]float32, Dv)
	for i := 0; i < Dk; i++ {
		ki := k[i]
		if ki == 0 {
			continue
		}
		rowOff := i * Dv
		for j := 0; j < Dv; j++ {
			vHat[j] += S[rowOff+j] * ki
		}
	}

	// 3. Error delta = v - vHat
	delta := make([]float32, Dv)
	for j := 0; j < Dv; j++ {
		delta[j] = v[j] - vHat[j]
	}

	// 4. Update S += beta * (k (outer) delta)
	if beta != 0 {
		for i := 0; i < Dk; i++ {
			betaKi := beta * k[i]
			if betaKi == 0 {
				continue
			}
			rowOff := i * Dv
			for j := 0; j < Dv; j++ {
				S[rowOff+j] += betaKi * delta[j]
			}
		}
	}

	// 5. Query output o = S^T * q: o[j] = sum_i S[i*Dv + j] * q[i]
	out := make([]float32, Dv)
	for i := 0; i < Dk; i++ {
		qi := q[i]
		if qi == 0 {
			continue
		}
		rowOff := i * Dv
		for j := 0; j < Dv; j++ {
			out[j] += S[rowOff+j] * qi
		}
	}

	return out
}

// StepGLM5NextKDALayer performs the delta update across all heads of a KDA layer.
// Q, K are [NumHeads * Dk], V is [NumHeads * Dv].
// alpha, beta are [NumHeads].
// Returns concatenated out [NumHeads * Dv].
func StepGLM5NextKDALayer(st *GLM5NextKDALayerState, Q, K, V []float32, alpha, beta []float32) []float32 {
	numHeads := st.NumHeads
	dk := st.HeadDim
	dv := st.HeadDim

	out := make([]float32, numHeads*dv)
	for h := 0; h < numHeads; h++ {
		sHead := st.HeadMatrix(h)
		qh := Q[h*dk : (h+1)*dk]
		kh := K[h*dk : (h+1)*dk]
		vh := V[h*dv : (h+1)*dv]

		a := float32(1.0)
		if h < len(alpha) {
			a = alpha[h]
		}
		b := float32(1.0)
		if h < len(beta) {
			b = beta[h]
		}

		oh := StepGLM5NextKDAHead(sHead, dk, dv, qh, kh, vh, a, b)
		copy(out[h*dv:(h+1)*dv], oh)
	}
	return out
}
