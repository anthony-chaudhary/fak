package model

// qwen35_chunked.go — the batched-prefill twin of the Gated-DeltaNet token mixer (issue #443,
// acceptance box 1: "a chunked or batched Gated-DeltaNet prefill path with f32 parity against
// the scalar reference").
//
// The lever, stated precisely. linearAttnSeq (qwen35.go) projects every prefill token through
// the five linear-attention weights (in_proj_qkv / z / b / a and out_proj) with a SEPARATE
// per-token GEMV — the same GEMV-per-token waste batch.go diagnoses for the dense layers, where
// each weight row is streamed from memory once PER TOKEN and thrown away. This path hoists those
// projections into full-sequence matMulBatch GEMMs: each weight row is read once and reused across
// all P tokens, raising arithmetic intensity from GEMV's 0.5 flop/byte toward compute-bound. The
// depthwise conv1d, the gated delta-rule recurrence, and the per-head gated RMSNorm are kept as
// the EXACT scalar math of linearAttnSeq — only WHICH tokens share a weight load changes.
//
// The bit-identity contract (f32). matMulBatch's row [t,o] is, by construction, bit-for-bit equal
// to the per-token residentMatRows it replaces (same fdot, same i-order — see parallel.go and the
// matMulBatch doc). So on an f32 model linearAttnSeqBatched is bit-IDENTICAL to linearAttnSeq, and
// the witness TestQwen35LinearAttnBatchedMatchesScalar pins Float32bits equality layer-by-layer and
// end-to-end through Forward. A Q8-resident model (the GGUF lean path) falls back to the per-token
// resident GEMV inside residentMatMulBatch — still bit-exact to the scalar path, just not yet
// batched; the batched-Q8 tile GEMM is the separate box-2 slice the issue scopes out of this pass.

import (
	"math"
	"os"
)

// gdnBatchedPrefill is the opt-in gate (issue #443, box 3: a hybrid model opts into the accelerated
// prefill path only AFTER the witness passes). It defaults OFF so the trunk forward path is
// unchanged; FAK_GDN_BATCHED=1 routes the linear-attention prefill through linearAttnSeqBatched,
// which the parity witness certifies bit-identical to the scalar reference on the f32 path.
var gdnBatchedPrefill = initGDNBatchedPrefill()

func initGDNBatchedPrefill() bool {
	switch os.Getenv("FAK_GDN_BATCHED") {
	case "1", "true", "True", "TRUE", "on", "ON":
		return true
	default:
		return false
	}
}

// residentMatMulBatch is the batched form of residentMatRows: it applies the named weight to a
// [P, in] activation panel and returns the [P, out] row-major result. For an f32-resident weight
// it runs ONE matMulBatch GEMM (each weight row reused across all P tokens); for a quant-resident
// weight it falls back to the per-token resident GEMV (correctness-first — batched Q8/Q4 is the
// separate device slice). Either way Y[t] is bit-for-bit the per-token residentMatRows(name, X[t]).
func (m *Model) residentMatMulBatch(name string, X []float32, out, in, P int) []float32 {
	if m.has(name) {
		return matMulBatch(m.tensor(name), X, out, in, P)
	}
	Y := make([]float32, P*out)
	for t := 0; t < P; t++ {
		copy(Y[t*out:(t+1)*out], m.residentMatRows(name, X[t*in:(t+1)*in], out, in))
	}
	return Y
}

// linearAttnSeqBatched mirrors linearAttnSeq exactly, save that the five projection GEMVs are
// hoisted into full-sequence matMulBatch GEMMs (via residentMatMulBatch). The conv1d, the gated
// delta-rule recurrence, and the gated RMSNorm are byte-for-byte the scalar math, so the result is
// bit-identical to linearAttnSeq on the f32 path.
func (m *Model) linearAttnSeqBatched(l int, xn [][]float32) [][]float32 {
	cfg := m.Cfg
	H := cfg.HiddenSize
	nK := cfg.LinearNumKeyHeads
	nV := cfg.LinearNumValueHeads
	kHd := cfg.LinearKeyHeadDim
	vHd := cfg.LinearValueHeadDim
	keyDim := nK * kHd
	valDim := nV * vHd
	convDim := 2*keyDim + valDim
	K := cfg.LinearConvKernelDim
	seq := len(xn)
	eps := float32(cfg.RMSNormEps)
	p := func(s string) string { return layerName(l, s) }

	if seq == 0 {
		return nil
	}

	// Pack the per-token normalized inputs into one [seq, H] panel, then run each input
	// projection as a SINGLE batched GEMM instead of seq separate GEMVs.
	xPanel := make([]float32, seq*H)
	for t := range xn {
		copy(xPanel[t*H:(t+1)*H], xn[t])
	}
	mixedFlat := m.residentMatMulBatch(p("linear_attn.in_proj_qkv.weight"), xPanel, convDim, H, seq)
	zFlat := m.residentMatMulBatch(p("linear_attn.in_proj_z.weight"), xPanel, valDim, H, seq)
	bFlat := m.residentMatMulBatch(p("linear_attn.in_proj_b.weight"), xPanel, nV, H, seq)
	aFlat := m.residentMatMulBatch(p("linear_attn.in_proj_a.weight"), xPanel, nV, H, seq)

	aLog := m.tensor(p("linear_attn.A_log"))     // [nV]
	dtBias := m.tensor(p("linear_attn.dt_bias")) // [nV]
	normW := m.tensor(p("linear_attn.norm.weight"))
	conv := m.tensor(p("linear_attn.conv1d.weight")) // [convDim*K] depthwise (no bias)

	// Per-position views into the batched projections + the per-head decay g and gate beta.
	mixed := make([][]float32, seq)
	zAll := make([][]float32, seq)
	gDecay := make([][]float32, seq)
	beta := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		mixed[t] = mixedFlat[t*convDim : (t+1)*convDim]
		zAll[t] = zFlat[t*valDim : (t+1)*valDim]
		bvec := bFlat[t*nV : (t+1)*nV]
		avec := aFlat[t*nV : (t+1)*nV]
		g := make([]float32, nV)
		bt := make([]float32, nV)
		for h := 0; h < nV; h++ {
			bt[h] = sigmoidf(bvec[h])
			a := float32(math.Exp(float64(aLog[h]))) // A = exp(A_log)
			dt := softplus(avec[h] + dtBias[h])
			g[h] = float32(math.Exp(float64(-a * dt))) // exp(g) state decay
		}
		gDecay[t] = g
		beta[t] = bt
	}

	// Causal depthwise conv1d (kernel K, no bias, left-padded) + SiLU over each channel.
	convOut := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		row := make([]float32, convDim)
		for c := 0; c < convDim; c++ {
			var acc float32
			cb := c * K
			for j := 0; j < K; j++ {
				if ti := t - (K - 1) + j; ti >= 0 {
					acc += conv[cb+j] * mixed[ti][c]
				}
			}
			row[c] = silu(acc)
		}
		convOut[t] = row
	}

	// Recurrent gated delta rule — identical scalar math to linearAttnSeq.
	scale := float32(1.0 / math.Sqrt(float64(kHd)))
	repeat := nV / nK
	state := make([][]float32, nV)
	for h := range state {
		state[h] = make([]float32, kHd*vHd)
	}
	core := make([][]float32, seq)
	qNorm := make([]float32, keyDim)
	kNorm := make([]float32, keyDim)
	kvmem := make([]float32, vHd)
	delta := make([]float32, vHd)
	for t := 0; t < seq; t++ {
		q := convOut[t][0:keyDim]
		k := convOut[t][keyDim : 2*keyDim]
		for h := 0; h < nK; h++ {
			l2normInto(qNorm[h*kHd:(h+1)*kHd], q[h*kHd:(h+1)*kHd], 1e-6)
			l2normInto(kNorm[h*kHd:(h+1)*kHd], k[h*kHd:(h+1)*kHd], 1e-6)
			for i := h * kHd; i < (h+1)*kHd; i++ {
				qNorm[i] *= scale
			}
		}
		v := convOut[t][2*keyDim : 2*keyDim+valDim]
		out := make([]float32, valDim)
		for h := 0; h < nV; h++ {
			kh := h / repeat
			qn := qNorm[kh*kHd : (kh+1)*kHd]
			kn := kNorm[kh*kHd : (kh+1)*kHd]
			vh := v[h*vHd : (h+1)*vHd]
			g := gDecay[t][h]
			bt := beta[t][h]
			st := state[h]
			for i := range st {
				st[i] *= g
			}
			for d := range kvmem {
				kvmem[d] = 0
			}
			for i := 0; i < kHd; i++ {
				ki := kn[i]
				base := i * vHd
				for d := 0; d < vHd; d++ {
					kvmem[d] += st[base+d] * ki
				}
			}
			for d := 0; d < vHd; d++ {
				delta[d] = (vh[d] - kvmem[d]) * bt
			}
			od := out[h*vHd : (h+1)*vHd]
			for i := 0; i < kHd; i++ {
				ki := kn[i]
				qi := qn[i]
				base := i * vHd
				for d := 0; d < vHd; d++ {
					st[base+d] += ki * delta[d]
					od[d] += st[base+d] * qi
				}
			}
		}
		core[t] = out
	}

	// Per-head gated RMSNorm of the readout, then the output projection as one batched GEMM.
	coreFlat := make([]float32, seq*valDim)
	for t := 0; t < seq; t++ {
		for h := 0; h < nV; h++ {
			rmsNormGatedInPlace(core[t][h*vHd:(h+1)*vHd], normW, zAll[t][h*vHd:(h+1)*vHd], eps)
		}
		copy(coreFlat[t*valDim:(t+1)*valDim], core[t])
	}
	outFlat := m.residentMatMulBatch(p("linear_attn.out_proj.weight"), coreFlat, H, valDim, seq)
	out := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		out[t] = outFlat[t*H : (t+1)*H]
	}
	return out
}
