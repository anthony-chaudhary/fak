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
	"fmt"
	"math"
	"os"

	"github.com/anthony-chaudhary/fak/internal/compute"
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
// with no active LoRA it runs ONE matMulBatch GEMM (each weight row reused across all P tokens);
// a quant-resident weight (Q8/int4/Q4_K/k-quant/GPTQ), or an active LoRA adapter whose per-row
// low-rank delta the batched f32 kernel does not model, falls back to the per-token resident GEMV
// (correctness-first — batched Q8/Q4 is the separate device slice). Either way Y[t] is bit-for-bit
// the per-token residentMatRows(name, X[t]), so callers stay bit-identical whatever the format.
func (m *Model) residentMatMulBatch(name string, X []float32, out, in, P int) []float32 {
	if m.has(name) && m.lora == nil {
		return matMulBatch(m.tensor(name), X, out, in, P)
	}
	Y := make([]float32, P*out)
	for t := 0; t < P; t++ {
		copy(Y[t*out:(t+1)*out], m.residentMatRows(name, X[t*in:(t+1)*in], out, in))
	}
	return Y
}

// linearAttnSeqBatched is the cacheless wrapper over linearAttnSeqBatchedStateful,
// preserving the zero-state prefill contract for whole sequences.
func (m *Model) linearAttnSeqBatched(l int, xn [][]float32) [][]float32 {
	out, _ := m.linearAttnSeqBatchedStateful(l, xn, nil)
	return out
}

// linearAttnSeqBatchedStateful runs batched-projection Gated-DeltaNet token mixing for a sequence
// of new rows, resuming from and updating the caller-owned persistent linearAttnLayerState when non-nil.
// When st is nil, state is initialized from zero (pure cacheless prefill).
// Before mutation, it validates geometry against model configuration and refuses malformed states.
func (m *Model) linearAttnSeqBatchedStateful(l int, xn [][]float32, st *linearAttnLayerState) ([][]float32, error) {
	cfg := m.Cfg
	H := cfg.HiddenSize
	nK, nV, kHd, vHd, keyDim, valDim, convDim := cfg.linearAttnDims()
	K := cfg.LinearConvKernelDim
	seq := len(xn)
	eps := float32(cfg.RMSNormEps)
	p := func(s string) string { return layerName(l, s) }

	if seq == 0 {
		return nil, nil
	}

	// Validate caller-owned state geometry before mutation.
	if st != nil {
		if len(st.recurrent) != 0 {
			if len(st.recurrent) != nV {
				return nil, fmt.Errorf("model: linearAttnSeqBatchedStateful invalid recurrent head count %d, want %d", len(st.recurrent), nV)
			}
			for h := 0; h < nV; h++ {
				if len(st.recurrent[h]) != kHd*vHd {
					return nil, fmt.Errorf("model: linearAttnSeqBatchedStateful invalid recurrent state size %d at head %d, want %d", len(st.recurrent[h]), h, kHd*vHd)
				}
			}
		}
		for i, r := range st.conv {
			if len(r) != convDim {
				return nil, fmt.Errorf("model: linearAttnSeqBatchedStateful invalid conv row size %d at index %d, want %d", len(r), i, convDim)
			}
		}
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

	// Causal depthwise conv1d (kernel K, no bias) + SiLU over each channel.
	// When st is present and holds prior conv rows, history is read from st.conv.
	var convOut [][]float32
	if st == nil {
		if tiledOut, _, _, err := compute.TiledConvConcatForwardSlices(mixed, conv, convDim, K, nil); err == nil {
			convOut = tiledOut
		}
	}
	if convOut == nil {
		convOut = make([][]float32, seq)
		for t := 0; t < seq; t++ {
			row := make([]float32, convDim)
			for c := 0; c < convDim; c++ {
				var acc float32
				cb := c * K
				for j := 0; j < K; j++ {
					ti := t - (K - 1) + j
					if ti >= 0 {
						acc += conv[cb+j] * mixed[ti][c]
					} else if st != nil {
						idx := len(st.conv) + ti
						if idx >= 0 && idx < len(st.conv) {
							acc += conv[cb+j] * st.conv[idx][c]
						}
					}
				}
				row[c] = silu(acc)
			}
			convOut[t] = row
		}
	}

	// If st is provided, advance st.conv with the new mixed rows (copying to prevent aliasing).
	if st != nil {
		for t := 0; t < seq; t++ {
			st.pushConvRow(mixed[t], K-1)
		}
	}

	// Recurrent gated delta rule — identical scalar math to linearAttnSeq.
	scale := float32(1.0 / math.Sqrt(float64(kHd)))
	repeat := nV / nK

	var state [][]float32
	if st != nil {
		if len(st.recurrent) == 0 {
			*st = newLinearAttnLayerState(cfg)
		}
		state = st.recurrent
	} else {
		state = make([][]float32, nV)
		for h := range state {
			state[h] = make([]float32, kHd*vHd)
		}
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
			stHead := state[h]
			od := out[h*vHd : (h+1)*vHd]
			VectorizedHeadStep(stHead, qn, kn, vh, bt, g, od, kvmem, delta)
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
	return out, nil
}
