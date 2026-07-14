// Command gdn-quant-length-sensitivity is the host-runnable, device-independent arm of issue
// #4273 — Qwen3.6-27B GGUF degenerates into a verbatim repetition loop on ~1.3k-token prompts
// while short prompts stay coherent. It is the length-axis sibling of gdn-divergence-sensitivity
// (which answers the *token-3 correctness* drift on the depth axis).
//
// The phenomenon (#4273, from the on-device real-artifact witnesses q36rawrepro / q36safe18 / q36tokenloop7):
// the completion opens coherent and grounded, then degrades ~40 tokens into decode and cycles to
// the token cap. It has been narrowed, by real-artifact witnesses, to LONG-CONTEXT REAL QUANTIZED
// inference — and specifically NOT to: the tokenizer/chat-template (HF tokenization byte-identical),
// generic f32 Gated-DeltaNet (GDN) recurrence (the Ornith oracle matches HF greedy at short and
// 300-token horizons), batched-vs-token-loop prefill (both collapse identically), recurrent state
// carry (the long-horizon self-consistency guard is green), or universal model corruption (`2+2=4`
// decodes cleanly at short context). Short coherent + long collapse + f32-fine + quant-fails is the
// signature this program measures.
//
// The variable it isolates: WEIGHT QUANTIZATION of the GDN projections. In the real loader
// (internal/model/safetensors_quant.go:isQuantWeight) all five linear_attn matmul weights —
// in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj — are quantized to Q8_0 at compute time
// in BOTH failing paths: the default lean-Q8 path quantizes everything, and the resident-Q4_K path
// refuses linear_attn from raw-Q4_K residency (quant_q4k.go:ResidentQ4KEligible returns false for
// .linear_attn.) then re-quantizes the dequant'd f32 to Q8 via the builder. That is exactly why the
// collapse is quant-INDEPENDENT (Q8 and Q4_K both fail): the GDN path runs Q8 in both. The coherent
// Ornith oracle, by contrast, is full f32. So the one difference between "coherent" and "collapses"
// on the GDN projections is weight quantization — the perturbation this program applies.
//
// It answers a falsifiable question with no Mac, no GPU, and no 27B artifact:
//
//	Does GDN-projection weight-quantization error, fed through the delta-rule recurrent scan,
//	COMPOUND with decode length — i.e. does the relative hidden divergence rho = ||Δh||/||h||
//	between the f32 and the Q8/Q4_K run GROW with the number of carried positions P, reaching a
//	decode-destabilizing magnitude by the ~1757-token failure horizon?
//
// Two outcomes, both useful and honest:
//   - rho grows with P and reaches a large magnitude (>= rho* ~ 0.09, the order at which the argmax
//     routinely moves) near P~1757: weight-quant magnitude, compounded over the recurrence, is a
//     SUFFICIENT mechanism for the length-onset collapse -> the fix direction is to keep the GDN
//     projections (and/or the recurrent state) in higher precision, as the reference runtimes do.
//   - rho stays small and roughly FLAT in P (like the f16-state bracket's ~3e-5 in the sibling):
//     weight-quant magnitude ALONE cannot explain #4273 -> the quantized long-context path has an
//     ALGORITHMIC defect (dequant/scale/tensor-mapping), not a precision-magnitude one, and the
//     on-artifact early-logit comparison vs llama.cpp/HF is the decisive next witness.
//
// The GDN layer math is copied verbatim from internal/model/metal_prefill_hybrid_core.go:202-246
// (the prefill twin of qwen35.go:linearAttnStep) via the sibling gdn-divergence-sensitivity, so the
// recurrence is faithful; only the projection weights are round-tripped through Q8_0 / Q4_K before
// the f32 matmul. Weight round-trip (dequant back to f32, then f32 matmul) models the WEIGHT-quant
// contribution; the real kernel additionally quantizes the activation vector per matmul, so the real
// path's total quant error is >= what this measures — this rho is a conservative lower bound on the
// quant-induced divergence.
//
// What this does NOT do: it does not run llama.cpp, does not load the 27B artifact, uses
// seeded-random (not trained) weights so it CANNOT reproduce the trained repetition attractor
// itself, and does not claim to have found a bug or a fix. Its claim is about the ORDER OF MAGNITUDE
// and the P-SLOPE of quant-induced recurrent divergence — the device-independent bound the current
// #4273 hypothesis demands. The on-artifact witness remains the honest `not yet`.
//
// Run:
//
//	go run ./experiments/qwen36/gdn-quant-length-sensitivity                    # human table
//	go run ./experiments/qwen36/gdn-quant-length-sensitivity -json              # machine result
//	go run ./experiments/qwen36/gdn-quant-length-sensitivity -positions 16,128  # quick smoke
//	go run ./experiments/qwen36/gdn-quant-length-sensitivity -hidden 5120       # real-H confirm (slow)
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
)

// Qwen3.6-27B Gated-DeltaNet layer dims (the 48 linear_attn layers). H (hidden) is a flag: rho is
// relative (||Δh||/||h||), so it is ~independent of the matmul fan-in H, which only drives cost —
// the recurrence dynamics that #4273 lives in depend on nV/kHd/vHd/K and the carried length, all
// real here. -hidden 5120 reproduces the true H for a (slow) confirmatory run.
const (
	nK  = 16  // LinearNumKeyHeads
	nV  = 48  // LinearNumValueHeads
	kHd = 128 // LinearKeyHeadDim
	vHd = 128 // LinearValueHeadDim
	K   = 4   // ssm.conv_kernel (Qwen3-Next)
)

// hidden is set from -hidden in main; keyDim/valDim/convDim below are H-independent.
var hidden = 1024

// aLogMean is the mean of the per-head A_log draw, set from -alog. It controls the recurrent decay
// g = exp(-exp(A_log)*dt): mean -2 -> g~0.88 (effective memory ~8 positions); mean -5 -> g~0.99
// (long memory, ~100+ positions). It is the load-bearing knob for whether per-step error can
// compound across a long decode, so the sweep MUST include a near-1-decay run to be honest.
var aLogMean = -2.0

const (
	keyDim  = nK * kHd          // 2048
	valDim  = nV * vHd          // 6144
	convDim = 2*keyDim + valDim // 10240
)

func silu(x float32) float32     { return x / (1 + float32(math.Exp(float64(-x)))) }
func sigmoidf(x float32) float32 { return 1 / (1 + float32(math.Exp(float64(-x)))) }
func softplus(x float32) float32 { return float32(math.Log1p(math.Exp(float64(x)))) }

// l2normInto reproduces qwen35.go:l2normInto — sum (not mean) of squares, eps inside sqrt.
func l2normInto(dst, src []float32, eps float32) {
	var ss float32
	for _, v := range src {
		ss += v * v
	}
	inv := 1.0 / float32(math.Sqrt(float64(ss+eps)))
	for i := range src {
		dst[i] = src[i] * inv
	}
}

// rmsnormGain1p reproduces the (1+w) RMSNorm used by every non-GDN-readout norm in qwen35.
func rmsnormGain1p(dst, src, w []float32, eps float32) {
	var ss float32
	for _, v := range src {
		ss += v * v
	}
	inv := 1.0 / float32(math.Sqrt(float64(ss/float32(len(src)))+float64(eps)))
	for i := range src {
		dst[i] = src[i] * inv * (1 + w[i])
	}
}

// rmsNormGatedInPlace reproduces qwen35.go:rmsNormGatedInPlace — the GDN readout's gated norm with
// PLAIN (not 1+w) weight and a silu(gate) multiply.
func rmsNormGatedInPlace(x, w, gate []float32, eps float32) {
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	inv := 1.0 / float32(math.Sqrt(float64(ss/float32(len(x)))+float64(eps)))
	for i := range x {
		x[i] = w[i] * (x[i] * inv) * silu(gate[i])
	}
}

func depthwiseCausalSilu(dst, src, weights []float32, steps, channels, kernel int) {
	for t := 0; t < steps; t++ {
		outRow := dst[t*channels : (t+1)*channels]
		for c := 0; c < channels; c++ {
			var acc float32
			base := c * kernel
			for j := 0; j < kernel; j++ {
				source := t + j - (kernel - 1)
				if source >= 0 {
					acc += weights[base+j] * src[source*channels+c]
				}
			}
			outRow[c] = silu(acc)
		}
	}
}

func parMatmul(Y, X, W []float32, P, outDim, inDim int) {
	workers := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	chunk := (outDim + workers - 1) / workers
	for w := 0; w < workers; w++ {
		lo := w * chunk
		hi := lo + chunk
		if hi > outDim {
			hi = outDim
		}
		if lo >= hi {
			break
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			for t := 0; t < P; t++ {
				xr := X[t*inDim : (t+1)*inDim]
				yr := Y[t*outDim : (t+1)*outDim]
				for i := lo; i < hi; i++ {
					wr := W[i*inDim : (i+1)*inDim]
					var acc float32
					for j := 0; j < inDim; j++ {
						acc += wr[j] * xr[j]
					}
					yr[i] = acc
				}
			}
		}(lo, hi)
	}
	wg.Wait()
}

// layerWeights holds one GDN layer's parameters. The five projection matrices (wqkv, wz, wb, wa,
// wOut) are the isQuantWeight tensors that the real loader stores as Q8; the control tensors (wIn,
// conv, aLog, dtB, normW) take the dequant->f32 path in the loader and are kept f32 here too.
type layerWeights struct {
	wIn   []float32 // input_layernorm.weight        [H]      (f32 in loader)
	wqkv  []float32 // in_proj_qkv                    [convDim, H]  (Q8)
	wz    []float32 // in_proj_z                      [valDim, H]   (Q8)
	wb    []float32 // in_proj_b                      [nV, H]       (Q8)
	wa    []float32 // in_proj_a                      [nV, H]       (Q8)
	conv  []float32 // conv1d.weight                  [convDim, K]  (f32 in loader)
	aLog  []float32 // A_log                          [nV]          (f32)
	dtB   []float32 // dt_bias                        [nV]          (f32)
	normW []float32 // linear_attn.norm.weight        [vHd]         (f32)
	wOut  []float32 // out_proj                       [H, valDim]   (Q8)
}

func newLayerWeights() *layerWeights {
	return &layerWeights{
		wIn:   make([]float32, hidden),
		wqkv:  make([]float32, convDim*hidden),
		wz:    make([]float32, valDim*hidden),
		wb:    make([]float32, nV*hidden),
		wa:    make([]float32, nV*hidden),
		conv:  make([]float32, convDim*K),
		aLog:  make([]float32, nV),
		dtB:   make([]float32, nV),
		normW: make([]float32, vHd),
		wOut:  make([]float32, hidden*valDim),
	}
}

// fill draws each weight from a per-layer seeded normal with 1/sqrt(fan_in) scaling, so the
// projections behave like trained weights (bounded activations) without the artifact. Identical
// draw to gdn-divergence-sensitivity so the two experiments share a residual regime.
func (lw *layerWeights) fill(layer int) {
	r := rand.New(rand.NewSource(int64(0x9E3779B9 ^ layer)))
	gauss := func(s []float32, scale float32) {
		for i := range s {
			s[i] = float32(r.NormFloat64()) * scale
		}
	}
	gauss(lw.wIn, 0.02)
	gauss(lw.wqkv, float32(1.0/math.Sqrt(float64(hidden))))
	gauss(lw.wz, float32(1.0/math.Sqrt(float64(hidden))))
	gauss(lw.wb, float32(1.0/math.Sqrt(float64(hidden))))
	gauss(lw.wa, float32(1.0/math.Sqrt(float64(hidden))))
	gauss(lw.conv, 0.5)
	gauss(lw.wOut, float32(1.0/math.Sqrt(float64(valDim))))
	for h := 0; h < nV; h++ {
		lw.aLog[h] = float32(r.NormFloat64())*0.5 + float32(aLogMean) // decay strength; see aLogMean
		lw.dtB[h] = float32(r.NormFloat64()) * 0.2
	}
	for i := range lw.normW {
		lw.normW[i] = float32(r.NormFloat64()) * 0.02
	}
}

// quantMode selects the weight-quantization applied to the GDN projections in the test run.
type quantMode int

const (
	modeQ8  quantMode = iota // per-row, per-32 block, symmetric int8 (GGML Q8_0; quant.go:quantizeQ8)
	modeQ4K                  // per-row, per-32 affine 4-bit sub-block (GGML Q4_K sub-block; min/max)
)

// q8RowRoundTrip round-trips w[out,in] through Q8_0 (block=32, symmetric, d=amax/127) and returns
// the dequantized f32 — bit-faithful to internal/model/quant.go:quantizeQ8 + its dequant.
func q8RowRoundTrip(w []float32, out, in int) []float32 {
	const blk = 32
	dq := make([]float32, len(w))
	nblk := in / blk
	tail := in % blk // in is a multiple of 32 for real GDN dims; handle a stray tail faithlessly-but-safely
	for o := 0; o < out; o++ {
		row := w[o*in : o*in+in]
		dst := dq[o*in : o*in+in]
		for b := 0; b < nblk; b++ {
			block := row[b*blk : b*blk+blk]
			var amax float32
			for _, v := range block {
				a := v
				if a < 0 {
					a = -a
				}
				if a > amax {
					amax = a
				}
			}
			d := amax / 127
			for i := 0; i < blk; i++ {
				if d == 0 {
					dst[b*blk+i] = 0
					continue
				}
				q := int32(math.Round(float64(block[i] / d)))
				if q > 127 {
					q = 127
				} else if q < -127 {
					q = -127
				}
				dst[b*blk+i] = float32(q) * d
			}
		}
		for i := nblk * blk; i < nblk*blk+tail; i++ {
			dst[i] = row[i]
		}
	}
	return dq
}

// q4kRowRoundTrip round-trips w[out,in] through a per-row, per-32 affine 4-bit sub-block scheme
// (min/max, scale=(max-min)/15, 4-bit code, dequant q*scale+min). This models the GGML Q4_K
// sub-block (quant_q4k.go: 8 sub-blocks of 32 per 256 super-block); the real Q4_K additionally
// quantizes the per-sub-block (scale,min) pair to 6 bits, which only INCREASES error, so this is a
// lower bound on the Q4_K contribution.
func q4kRowRoundTrip(w []float32, out, in int) []float32 {
	const sub = 32
	dq := make([]float32, len(w))
	nsub := in / sub
	tail := in % sub
	for o := 0; o < out; o++ {
		row := w[o*in : o*in+in]
		dst := dq[o*in : o*in+in]
		for b := 0; b < nsub; b++ {
			block := row[b*sub : b*sub+sub]
			mn, mx := block[0], block[0]
			for _, v := range block {
				if v < mn {
					mn = v
				}
				if v > mx {
					mx = v
				}
			}
			scale := (mx - mn) / 15
			for i := 0; i < sub; i++ {
				if scale == 0 {
					dst[b*sub+i] = mn
					continue
				}
				q := int32(math.Round(float64((block[i] - mn) / scale)))
				if q > 15 {
					q = 15
				} else if q < 0 {
					q = 0
				}
				dst[b*sub+i] = float32(q)*scale + mn
			}
		}
		for i := nsub * sub; i < nsub*sub+tail; i++ {
			dst[i] = row[i]
		}
	}
	return dq
}

// quantizeProjections returns a copy of lw with the five isQuantWeight projection matrices
// round-tripped through the given quant mode and the control tensors shared (f32) — exactly the
// tensor split the real loader applies.
func quantizeProjections(lw *layerWeights, mode quantMode) *layerWeights {
	rt := q8RowRoundTrip
	if mode == modeQ4K {
		rt = q4kRowRoundTrip
	}
	q := *lw // copy the slice headers; the control tensors are shared read-only
	q.wqkv = rt(lw.wqkv, convDim, hidden)
	q.wz = rt(lw.wz, valDim, hidden)
	q.wb = rt(lw.wb, nV, hidden)
	q.wa = rt(lw.wa, nV, hidden)
	q.wOut = rt(lw.wOut, hidden, valDim)
	return &q
}

// gdnLayer runs one linear_attn (GDN) layer's forward over P tokens and returns its output [P,H]
// (to be added to the residual). Verbatim metal_prefill_hybrid_core.go body; recurrent + conv state
// start at zero (fresh-prefill precondition — prefill on this path is itself a token loop, so this
// is also the decode carry). No mode branch: the ONLY difference between the two runs is which
// (quantized-or-not) weights are passed in.
func gdnLayer(lw *layerWeights, X []float32, P int, eps float32) []float32 {
	Xn := make([]float32, P*hidden)
	for t := 0; t < P; t++ {
		rmsnormGain1p(Xn[t*hidden:(t+1)*hidden], X[t*hidden:(t+1)*hidden], lw.wIn, eps)
	}

	mixed := make([]float32, P*convDim)
	zAll := make([]float32, P*valDim)
	bvec := make([]float32, P*nV)
	avec := make([]float32, P*nV)
	parMatmul(mixed, Xn, lw.wqkv, P, convDim, hidden)
	parMatmul(zAll, Xn, lw.wz, P, valDim, hidden)
	parMatmul(bvec, Xn, lw.wb, P, nV, hidden)
	parMatmul(avec, Xn, lw.wa, P, nV, hidden)

	convOut := make([]float32, P*convDim)
	depthwiseCausalSilu(convOut, mixed, lw.conv, P, convDim, K)

	scale := float32(1.0 / math.Sqrt(float64(kHd)))
	repeat := nV / nK
	qNormAll := make([]float32, P*keyDim)
	kNormAll := make([]float32, P*keyDim)
	for t := 0; t < P; t++ {
		row := convOut[t*convDim : (t+1)*convDim]
		q := row[0:keyDim]
		k := row[keyDim : 2*keyDim]
		qNorm := qNormAll[t*keyDim : (t+1)*keyDim]
		kNorm := kNormAll[t*keyDim : (t+1)*keyDim]
		for h := 0; h < nK; h++ {
			l2normInto(qNorm[h*kHd:(h+1)*kHd], q[h*kHd:(h+1)*kHd], 1e-6)
			l2normInto(kNorm[h*kHd:(h+1)*kHd], k[h*kHd:(h+1)*kHd], 1e-6)
			for i := h * kHd; i < (h+1)*kHd; i++ {
				qNorm[i] *= scale
			}
		}
	}

	aExp := make([]float32, nV)
	for h := 0; h < nV; h++ {
		aExp[h] = float32(math.Exp(float64(lw.aLog[h])))
	}

	core := make([]float32, P*valDim)
	var wg sync.WaitGroup
	workers := runtime.GOMAXPROCS(0)
	chunk := (nV + workers - 1) / workers
	for wk := 0; wk < workers; wk++ {
		hlo := wk * chunk
		hhi := hlo + chunk
		if hhi > nV {
			hhi = nV
		}
		if hlo >= hhi {
			break
		}
		wg.Add(1)
		go func(hlo, hhi int) {
			defer wg.Done()
			st := make([]float32, kHd*vHd)
			kvmem := make([]float32, vHd)
			delta := make([]float32, vHd)
			for h := hlo; h < hhi; h++ {
				for i := range st {
					st[i] = 0
				}
				kh := h / repeat
				a := aExp[h]
				dtB := lw.dtB[h]
				for t := 0; t < P; t++ {
					row := convOut[t*convDim : (t+1)*convDim]
					qn := qNormAll[t*keyDim+kh*kHd : t*keyDim+(kh+1)*kHd]
					kn := kNormAll[t*keyDim+kh*kHd : t*keyDim+(kh+1)*kHd]
					vh := row[2*keyDim+h*vHd : 2*keyDim+(h+1)*vHd]
					bt := sigmoidf(bvec[t*nV+h])
					dt := softplus(avec[t*nV+h] + dtB)
					g := float32(math.Exp(float64(-a * dt)))
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
					od := core[t*valDim+h*vHd : t*valDim+(h+1)*vHd]
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
			}
		}(hlo, hhi)
	}
	wg.Wait()

	for t := 0; t < P; t++ {
		for h := 0; h < nV; h++ {
			rmsNormGatedInPlace(
				core[t*valDim+h*vHd:t*valDim+(h+1)*vHd],
				lw.normW,
				zAll[t*valDim+h*vHd:t*valDim+(h+1)*vHd],
				eps,
			)
		}
	}

	o := make([]float32, P*hidden)
	parMatmul(o, core, lw.wOut, P, hidden, valDim)
	return o
}

// runStackForLength runs the 48-GDN-layer residual stack over P positions TWICE — a reference run
// with f32 weights and a test run with the same weights but the projections quantized (mode) — and
// returns the relative hidden divergence rho of the LAST token after the whole stack, plus the
// per-layer rho curve. Both runs share identical inputs and identical (pre-quantization) weights, so
// the only source of divergence is projection weight quantization.
func runStackForLength(P, layers int, mode quantMode) (perLayerRho []float64, finalRho float64) {
	eps := float32(1e-6)
	X := make([]float32, P*hidden)
	Y := make([]float32, P*hidden)
	r := rand.New(rand.NewSource(0xC0FFEE))
	for i := range X {
		v := float32(r.NormFloat64())
		X[i] = v
		Y[i] = v
	}
	lw := newLayerWeights()
	last := (P - 1) * hidden
	for l := 0; l < layers; l++ {
		lw.fill(l)
		qw := quantizeProjections(lw, mode)
		oRef := gdnLayer(lw, X, P, eps)
		oTst := gdnLayer(qw, Y, P, eps)
		for i := range X {
			X[i] += oRef[i]
			Y[i] += oTst[i]
		}
		perLayerRho = append(perLayerRho, relDiv(X[last:last+hidden], Y[last:last+hidden]))
	}
	finalRho = relDiv(X[last:last+hidden], Y[last:last+hidden])
	return
}

// relDiv = ||a-b|| / ||a||.
func relDiv(a, b []float32) float64 {
	var num, den float64
	for i := range a {
		d := float64(a[i] - b[i])
		num += d * d
		den += float64(a[i]) * float64(a[i])
	}
	if den == 0 {
		return 0
	}
	return math.Sqrt(num) / math.Sqrt(den)
}

type lengthPoint struct {
	Positions int     `json:"positions"`
	FinalRho  float64 `json:"final_rho_last_token"`
	ImpliedDL float64 `json:"implied_logit_delta_at_logit_scale_20"`
}

type modeResult struct {
	Mode         string        `json:"mode"`
	Points       []lengthPoint `json:"points"`
	GrowthRatio  float64       `json:"rho_growth_ratio_maxP_over_minP"`
	MaxRho       float64       `json:"max_rho"`
	Destabilizes bool          `json:"reaches_destabilizing_rho_at_max_P"`
}

type result struct {
	Schema     string       `json:"schema"`
	Issue      string       `json:"issue"`
	Host       string       `json:"host"`
	Model      string       `json:"model"`
	Hidden     int          `json:"hidden"`
	ALogMean   float64      `json:"a_log_mean_decay_knob"`
	Layers     int          `json:"gdn_layers"`
	Positions  []int        `json:"positions_swept"`
	LogitScale float64      `json:"assumed_logit_scale"`
	RhoStar    float64      `json:"rho_destabilize_threshold"`
	Modes      []modeResult `json:"modes"`
	Verdict    string       `json:"verdict"`
}

func parsePositions(s string) ([]int, error) {
	var out []int
	for _, p := range strings.Split(s, ",") {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		n, err := strconv.Atoi(p)
		if err != nil {
			return nil, err
		}
		if n < 1 {
			return nil, fmt.Errorf("position %d must be >= 1", n)
		}
		out = append(out, n)
	}
	sort.Ints(out)
	if len(out) == 0 {
		return nil, fmt.Errorf("no positions parsed")
	}
	return out, nil
}

func main() {
	asJSON := flag.Bool("json", false, "emit machine-readable JSON")
	positionsFlag := flag.String("positions", "16,128,512,1757", "comma-separated carried-position counts to sweep (the ~1757 failure horizon by default)")
	layers := flag.Int("layers", 48, "number of GDN layers in the residual stack")
	hiddenFlag := flag.Int("hidden", 1024, "hidden size H (rho is ~H-independent; 5120 is the real 27B H, slower)")
	alogFlag := flag.Float64("alog", -2.0, "mean of the per-head A_log draw (decay strength): -2 -> g~0.88 (~8-pos memory); -5 -> g~0.99 (long memory)")
	flag.Parse()
	aLogMean = *alogFlag

	if *hiddenFlag%256 != 0 {
		fmt.Fprintf(os.Stderr, "hidden must be a multiple of 256 (Q8_0/Q4_K block alignment); got %d\n", *hiddenFlag)
		os.Exit(2)
	}
	hidden = *hiddenFlag

	positions, err := parsePositions(*positionsFlag)
	if err != nil {
		fmt.Fprintf(os.Stderr, "bad -positions: %v\n", err)
		os.Exit(2)
	}

	const logitScale = 20.0           // |top logit| order from the witnessed 27B decode
	const rhoStar = 1.75 / logitScale // ~0.0875: the relative hidden move at which a near-tie argmax flips

	res := result{
		Schema:     "qwen36-gdn-quant-length-sensitivity/v1",
		Issue:      "#4273 (long-context quantized repetition collapse); length sibling of gdn-divergence-sensitivity",
		Host:       "windows/amd64 (CGO_ENABLED=0); device-independent, no Mac / no GPU / no 27B artifact",
		Model:      fmt.Sprintf("Qwen3.6-27B GDN shapes (H=%d nK=%d nV=%d kHd=%d vHd=%d K=%d), projections round-tripped Q8_0 / Q4_K", hidden, nK, nV, kHd, vHd, K),
		Hidden:     hidden,
		ALogMean:   aLogMean,
		Layers:     *layers,
		Positions:  positions,
		LogitScale: logitScale,
		RhoStar:    rhoStar,
	}

	modes := []struct {
		m    quantMode
		name string
	}{
		{modeQ8, "Q8_0 (per-32 symmetric int8; the compute dtype of the GDN projections in BOTH failing paths)"},
		{modeQ4K, "Q4_K (per-32 affine 4-bit sub-block; lower bound — real Q4_K also quantizes the sub-scales)"},
	}

	for _, md := range modes {
		mr := modeResult{Mode: md.name}
		for _, P := range positions {
			_, final := runStackForLength(P, *layers, md.m)
			mr.Points = append(mr.Points, lengthPoint{Positions: P, FinalRho: final, ImpliedDL: final * logitScale})
			if final > mr.MaxRho {
				mr.MaxRho = final
			}
		}
		first := mr.Points[0].FinalRho
		lastPt := mr.Points[len(mr.Points)-1]
		if first > 0 {
			mr.GrowthRatio = lastPt.FinalRho / first
		}
		mr.Destabilizes = lastPt.FinalRho >= rhoStar
		res.Modes = append(res.Modes, mr)
	}

	// Verdict from whether either mode's rho GROWS with P and REACHES the destabilizing order at the
	// failure horizon. Both outcomes route the investigation (see file header).
	anyDestab, anyGrowth := false, false
	for _, m := range res.Modes {
		if m.Destabilizes {
			anyDestab = true
		}
		if m.GrowthRatio >= 3 {
			anyGrowth = true
		}
	}
	switch {
	case anyDestab:
		res.Verdict = "SUFFICIENT: GDN-projection weight-quantization error, compounded through the delta-rule " +
			"recurrence, reaches a decode-destabilizing relative hidden divergence (rho >= rho*) by the ~1757-token " +
			"horizon. Weight-quant magnitude is a plausible mechanism for #4273's length-onset collapse -> fix " +
			"direction: keep the GDN projections (and/or recurrent state) in higher precision, as llama.cpp/HF do."
	case anyGrowth:
		res.Verdict = "PARTIAL: quant-induced rho GROWS materially with decode length (>=3x from min to max P) but " +
			"stays below the near-tie flip order at the horizon. Compounding is real and length-dependent (consistent " +
			"with #4273's onset) but its magnitude alone is sub-threshold on random weights; a trained repetition " +
			"attractor could amplify it. The on-artifact early-logit comparison remains the decisive witness."
	default:
		res.Verdict = "INSUFFICIENT: quant-induced rho stays small and roughly FLAT in decode length (no material " +
			"compounding). Weight-quantization MAGNITUDE alone cannot explain #4273 -> the quantized long-context path " +
			"has an ALGORITHMIC defect (dequant/scale/tensor-mapping), not a precision-magnitude one. The decisive " +
			"next witness is the on-artifact early token/logit comparison vs llama.cpp/HF on the real 27B Q4_K_M."
	}

	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(res)
		return
	}

	fmt.Printf("Qwen3.6-27B GDN weight-quant / decode-length sensitivity — #4273 (device-independent)\n")
	fmt.Printf("  shapes: H=%d nK=%d nV=%d kHd=%d vHd=%d convDim=%d valDim=%d K=%d, %d GDN layers\n",
		hidden, nK, nV, kHd, vHd, convDim, valDim, K, *layers)
	fmt.Printf("  swept carried positions: %v   (rho* to flip a near-tie ~ %.4f at |logit|~%.0f)\n\n", positions, rhoStar, logitScale)
	for _, m := range res.Modes {
		fmt.Printf("  mode: %s\n", m.Mode)
		fmt.Printf("    %-10s %-16s %-16s\n", "positions", "final_rho", "implied|Δlogit|")
		for _, p := range m.Points {
			fmt.Printf("    %-10d %-16.4e %-16.4e\n", p.Positions, p.FinalRho, p.ImpliedDL)
		}
		fmt.Printf("    growth rho(maxP)/rho(minP) = %.2fx ; max rho = %.4e ; destabilizes at horizon: %v\n\n",
			m.GrowthRatio, m.MaxRho, m.Destabilizes)
	}
	fmt.Printf("  VERDICT: %s\n", res.Verdict)
}
