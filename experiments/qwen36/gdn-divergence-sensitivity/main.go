// Command gdn-divergence-sensitivity is the host-runnable, device-independent arm of the
// Qwen3.6-27B *correctness* parity blocker — the "token-3 drift" — described in
// experiments/qwen36/token3-drift-investigation-2026-06-28.md (§4 third bullet, §5 step 2).
//
// The phenomenon: on the fixed 22-token ChatML prompt, greedy decode, fak and llama.cpp
// agree for two tokens (`248068, 198`) then disagree on the third — fak picks `8160`
// (logit 23.18, 2nd-place `90700` at 21.43, a ~1.75-logit near-tie) while llama.cpp picks
// `90700`. The arch math is bit-exact vs HF on the tiny fixture, and the drift survives the
// Q8->q4_k change on BOTH engines, so it is localized to a kernel-numerics divergence at 27B
// scale on the hybrid Gated-DeltaNet (GDN) path. Hypothesis H1 (the top-ranked one) is that
// the delta-rule recurrent scan's fixed serial reduction order rounds differently from
// llama.cpp's kernel, and — because the scan is the only STATEFUL op — that per-step
// difference COMPOUNDS across tokens (state carry) and layers (residual stream) until it tips
// the near-tie at token 3.
//
// This program supplies the DEVICE-INDEPENDENT measurement that hypothesis demands: it runs a
// faithful 48-GDN-layer residual stack TWICE, where the two runs are bit-identical in every op
// EXCEPT the reduction order (or accumulation precision) of the recurrent scan, and measures
// how fast the resulting hidden-state divergence compounds with depth and tokens. The two runs
// share the same (seeded-random) weights and inputs, so the ONLY source of divergence is the
// scan numerics — exactly the "same math, different rounding" H1 posits. The recurrence math is
// copied verbatim from internal/model/metal_prefill_hybrid_core.go:202-246 (the prefill twin of
// qwen35.go:linearAttnStep); only the i-loop direction (mode "reorder") or a per-step f16 state
// round-trip (mode "f16state") differs between the two runs.
//
// It answers a falsifiable question with no Mac and no 27B artifact:
//
//	Is a reduction-order-only (or f16-state) numeric difference, compounded over 48 GDN
//	layers and ~24 carried positions, LARGE ENOUGH on its own to flip a ~1.75-logit
//	near-tie at the token-3 decode step?
//
// If the measured final relative hidden divergence rho >= rho* (~ margin / |logit| ~ 1.75/20
// ~ 0.09), pure rounding suffices and the fix is to match llama.cpp's reduction order. If
// rho << rho* even in the f16-state bracket, then numerics-as-rounding CANNOT explain the
// flip — the token-3 divergence is anomalously large and an algorithmic/ordering mismatch
// (not mere accumulation) is implicated, which sharpens the per-layer probe's job (find the
// layer where cosine drops ANOMALOUSLY, not merely below 1).
//
// What this does NOT do: it does not run llama.cpp, does not load the 27B artifact, and does
// not claim to have found the diverging (layer, op). Those are the Mac/artifact-gated steps
// (token3-drift-investigation §5 steps 4-5). This is the host-independent sensitivity bound.
//
// Real Qwen3.6-27B GDN dims (H=5120; LinearNumKeyHeads 16, LinearNumValueHeads 48,
// LinearKeyHeadDim 128, LinearValueHeadDim 128, ssm.conv_kernel 4) match the in-tree fixture
// internal/model/quant_q4k_resident_test.go and the sibling gdn-recurrence-bench.
//
// Run:
//
//	go run ./experiments/qwen36/gdn-divergence-sensitivity                 # human table, default modes
//	go run ./experiments/qwen36/gdn-divergence-sensitivity -json           # machine result (result.json)
//	go run ./experiments/qwen36/gdn-divergence-sensitivity -tokens 8 -layers 12   # quick smoke
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sync"
)

// Qwen3.6-27B Gated-DeltaNet layer dims (the 48 linear_attn layers).
const (
	H   = 5120 // hidden size
	nK  = 16   // LinearNumKeyHeads
	nV  = 48   // LinearNumValueHeads
	kHd = 128  // LinearKeyHeadDim
	vHd = 128  // LinearValueHeadDim
	K   = 4    // ssm.conv_kernel (Qwen3-Next)
)

const (
	keyDim  = nK * kHd          // 2048
	valDim  = nV * vHd          // 6144
	convDim = 2*keyDim + valDim // 10240
)

func silu(x float32) float32     { return x / (1 + float32(math.Exp(float64(-x)))) }
func sigmoidf(x float32) float32 { return 1 / (1 + float32(math.Exp(float64(-x)))) }
func softplus(x float32) float32 { return float32(math.Log1p(math.Exp(float64(x)))) }

// l2normInto reproduces qwen35.go:l2normInto — sum (not mean) of squares, eps inside sqrt.
// NOTE: the prefill path uses 1e-6 eps; this is identical between the two runs so it is not a
// divergence source here (it cancels). Kept verbatim for faithfulness.
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

// rmsNormGatedInPlace reproduces qwen35.go:rmsNormGatedInPlace — the GDN readout's gated norm
// with PLAIN (not 1+w) weight and a silu(gate) multiply. Identical between the two runs.
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

// quantF16 round-trips a float32 through IEEE-754 half precision (round-to-nearest-even,
// normal range; tiny values flush toward zero). Models a kernel that stores the GDN state in
// f16 — the realistic worst case for the recurrent accumulator.
func quantF16(x float32) float32 {
	b := math.Float32bits(x)
	sign := b & 0x80000000
	exp := int32((b>>23)&0xFF) - 127 // unbiased
	mant := b & 0x7FFFFF
	if exp == 128 { // Inf/NaN
		return x
	}
	if exp > 15 { // overflow -> saturate to max f16 (~65504), keep sign
		return math.Float32frombits(sign | math.Float32bits(65504))
	}
	if exp < -14 { // subnormal/zero in f16 -> flush to zero (state lives well above this)
		return math.Float32frombits(sign)
	}
	// Round the 23-bit mantissa to 10 bits, round-to-nearest-even.
	const drop = 23 - 10
	half := uint32(1) << (drop - 1)
	rounded := mant + half
	if rounded&((1<<drop)-1) == half { // exactly halfway -> round to even
		rounded &^= 1 << drop
	}
	// Carry into exponent if mantissa overflowed.
	newExp := exp
	if rounded&(1<<23) != 0 {
		rounded = 0
		newExp++
		if newExp > 15 {
			return math.Float32frombits(sign | math.Float32bits(65504))
		}
	}
	rounded &^= (1 << drop) - 1
	out := sign | (uint32(newExp+127) << 23) | (rounded & 0x7FFFFF)
	return math.Float32frombits(out)
}

// ---- parallel f32 GEMM: Y[P,out] = X[P,in] * W[out,in]^T ----

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

// layerWeights holds one GDN layer's parameters, refilled per layer from a layer-seeded PRNG
// (reused buffers — the same shapes for all layers — to avoid 48x large allocations).
type layerWeights struct {
	wIn   []float32 // input_layernorm.weight        [H]
	wqkv  []float32 // in_proj_qkv                    [convDim, H]
	wz    []float32 // in_proj_z                      [valDim, H]
	wb    []float32 // in_proj_b                      [nV, H]
	wa    []float32 // in_proj_a                      [nV, H]
	conv  []float32 // conv1d.weight                  [convDim, K]
	aLog  []float32 // A_log                          [nV]
	dtB   []float32 // dt_bias                        [nV]
	normW []float32 // linear_attn.norm.weight        [valDim/nV = vHd] per-head; stored [vHd]
	wOut  []float32 // out_proj                       [H, valDim]
}

func newLayerWeights() *layerWeights {
	return &layerWeights{
		wIn:   make([]float32, H),
		wqkv:  make([]float32, convDim*H),
		wz:    make([]float32, valDim*H),
		wb:    make([]float32, nV*H),
		wa:    make([]float32, nV*H),
		conv:  make([]float32, convDim*K),
		aLog:  make([]float32, nV),
		dtB:   make([]float32, nV),
		normW: make([]float32, vHd),
		wOut:  make([]float32, H*valDim),
	}
}

// fill draws each weight matrix from a per-layer seeded normal with a small scale, so the
// projections behave like trained weights (bounded activations) without needing the artifact.
func (lw *layerWeights) fill(layer int) {
	r := rand.New(rand.NewSource(int64(0x9E3779B9 ^ layer)))
	gauss := func(s []float32, scale float32) {
		for i := range s {
			s[i] = float32(r.NormFloat64()) * scale
		}
	}
	// 1/sqrt(fan_in) scaling keeps projection outputs O(1).
	gauss(lw.wIn, 0.02) // norm gain delta around 0 -> (1+w) ~ 1
	gauss(lw.wqkv, float32(1.0/math.Sqrt(float64(H))))
	gauss(lw.wz, float32(1.0/math.Sqrt(float64(H))))
	gauss(lw.wb, float32(1.0/math.Sqrt(float64(H))))
	gauss(lw.wa, float32(1.0/math.Sqrt(float64(H))))
	gauss(lw.conv, 0.5)
	gauss(lw.wOut, float32(1.0/math.Sqrt(float64(valDim))))
	for h := 0; h < nV; h++ {
		lw.aLog[h] = float32(r.NormFloat64())*0.5 - 2.0 // A_log ~ -2 -> exp ~ 0.13
		lw.dtB[h] = float32(r.NormFloat64()) * 0.2
	}
	for i := range lw.normW {
		lw.normW[i] = float32(r.NormFloat64()) * 0.02
	}
}

// scanMode selects how the recurrent reductions round.
type scanMode int

const (
	modeForward  scanMode = iota // i ascending (the trunk's serial order)
	modeReverse                  // i descending (a different, math-equivalent reduction order)
	modeF16state                 // forward order, but state round-tripped through f16 each step
)

// gdnLayer runs one linear_attn (GDN) layer's forward over P tokens and returns its output
// [P,H] (to be added to the residual). It is the verbatim metal_prefill_hybrid_core.go body,
// parameterized only by `mode` for the recurrent scan's numerics. Recurrent + conv state start
// at zero (fresh prefill precondition).
func gdnLayer(lw *layerWeights, X []float32, P int, mode scanMode, eps float32) []float32 {
	// input RMSNorm (1+w), identical across modes.
	Xn := make([]float32, P*H)
	for t := 0; t < P; t++ {
		rmsnormGain1p(Xn[t*H:(t+1)*H], X[t*H:(t+1)*H], lw.wIn, eps)
	}

	mixed := make([]float32, P*convDim)
	zAll := make([]float32, P*valDim)
	bvec := make([]float32, P*nV)
	avec := make([]float32, P*nV)
	parMatmul(mixed, Xn, lw.wqkv, P, convDim, H)
	parMatmul(zAll, Xn, lw.wz, P, valDim, H)
	parMatmul(bvec, Xn, lw.wb, P, nV, H)
	parMatmul(avec, Xn, lw.wa, P, nV, H)

	// causal depthwise conv1d + SiLU (verbatim core.go:145-169), fresh-prefill (no history).
	convOut := make([]float32, P*convDim)
	for t := 0; t < P; t++ {
		outRow := convOut[t*convDim : (t+1)*convDim]
		for c := 0; c < convDim; c++ {
			var acc float32
			cb := c * K
			for j := 0; j < K; j++ {
				src := t + j - (K - 1)
				if src < 0 {
					continue
				}
				acc += lw.conv[cb+j] * mixed[src*convDim+c]
			}
			outRow[c] = silu(acc)
		}
	}

	// q/k per-head L2-norm + 1/sqrt(kHd) query scale (verbatim core.go:186-201).
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

	// delta-rule recurrent scan (verbatim core.go:202-246) — the ONLY mode-dependent op.
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
					if mode == modeF16state {
						for i := range st {
							st[i] = quantF16(st[i])
						}
					}
					for d := range kvmem {
						kvmem[d] = 0
					}
					accumulate(kvmem, st, kn, mode) // kvmem[d] = sum_i st[i*vHd+d]*kn[i]
					for d := 0; d < vHd; d++ {
						delta[d] = (vh[d] - kvmem[d]) * bt
					}
					od := core[t*valDim+h*vHd : t*valDim+(h+1)*vHd]
					readout(od, st, kn, qn, delta, mode) // st += k(x)delta; od += sum_i st[i]*qn[i]
					if mode == modeF16state {
						for i := range st {
							st[i] = quantF16(st[i])
						}
					}
				}
			}
		}(hlo, hhi)
	}
	wg.Wait()

	// gated RMSNorm readout (verbatim core.go:247-258), identical across modes.
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

	o := make([]float32, P*H)
	parMatmul(o, core, lw.wOut, P, H, valDim)
	return o
}

// accumulate computes kvmem[d] = sum_i st[i*vHd+d]*kn[i] in the given reduction order.
func accumulate(kvmem, st, kn []float32, mode scanMode) {
	if mode == modeReverse {
		for i := kHd - 1; i >= 0; i-- {
			ki := kn[i]
			base := i * vHd
			for d := 0; d < vHd; d++ {
				kvmem[d] += st[base+d] * ki
			}
		}
		return
	}
	for i := 0; i < kHd; i++ {
		ki := kn[i]
		base := i * vHd
		for d := 0; d < vHd; d++ {
			kvmem[d] += st[base+d] * ki
		}
	}
}

// readout applies the state update st[i*vHd+d] += kn[i]*delta[d] (order-independent: disjoint
// blocks) and the readout reduction od[d] += sum_i st[i*vHd+d]*qn[i] in the given order.
func readout(od, st, kn, qn, delta []float32, mode scanMode) {
	if mode == modeReverse {
		for i := kHd - 1; i >= 0; i-- {
			ki := kn[i]
			qi := qn[i]
			base := i * vHd
			for d := 0; d < vHd; d++ {
				st[base+d] += ki * delta[d]
				od[d] += st[base+d] * qi
			}
		}
		return
	}
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

// runStackTraced records the relative divergence of the last token's hidden
// state against a reference run after EACH layer (the compounding curve). It runs the reference
// (forward) and the test (mode) in lockstep so per-layer snapshots align.
func runStackTraced(P, layers int, mode scanMode) (perLayerRho []float64, finalRho float64) {
	eps := float32(1e-6)
	X := make([]float32, P*H)
	Y := make([]float32, P*H)
	r := rand.New(rand.NewSource(0xC0FFEE))
	for i := range X {
		v := float32(r.NormFloat64())
		X[i] = v
		Y[i] = v
	}
	lwRef := newLayerWeights()
	lwTst := newLayerWeights()
	last := (P - 1) * H
	for l := 0; l < layers; l++ {
		lwRef.fill(l)
		lwTst.fill(l)
		oRef := gdnLayer(lwRef, X, P, modeForward, eps)
		oTst := gdnLayer(lwTst, Y, P, mode, eps)
		for i := range X {
			X[i] += oRef[i]
			Y[i] += oTst[i]
		}
		perLayerRho = append(perLayerRho, relDiv(X[last:last+H], Y[last:last+H]))
	}
	finalRho = relDiv(X[last:last+H], Y[last:last+H])
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

type modeResult struct {
	Mode              string    `json:"mode"`
	PerLayerRho       []float64 `json:"per_layer_rho_last_token"`
	FinalRho          float64   `json:"final_rho_last_token"`
	ImpliedLogitDelta float64   `json:"implied_logit_delta_at_logit_scale_20"`
	FlipsNearTie      bool      `json:"flips_175_logit_near_tie"`
}

type result struct {
	Schema        string       `json:"schema"`
	Issue         string       `json:"issue"`
	Host          string       `json:"host"`
	Model         string       `json:"model"`
	Tokens        int          `json:"tokens_carried"`
	Layers        int          `json:"gdn_layers"`
	LogitScale    float64      `json:"assumed_logit_scale"`
	NearTieMargin float64      `json:"observed_near_tie_margin_logits"`
	RhoThreshold  float64      `json:"rho_threshold_to_flip"`
	Modes         []modeResult `json:"modes"`
	Verdict       string       `json:"verdict"`
}

func main() {
	asJSON := flag.Bool("json", false, "emit machine-readable JSON")
	tokens := flag.Int("tokens", 24, "positions carried before the measured (token-3) step (22 prompt + 2 agreed)")
	layers := flag.Int("layers", 48, "number of GDN layers in the residual stack")
	flag.Parse()

	const logitScale = 20.0 // |top logit| from the witnessed token-3 data (fak 23.18 / 21.43)
	const nearTie = 1.75    // observed fak margin between {8160, 90700}
	rhoStar := nearTie / logitScale

	modes := []struct {
		m    scanMode
		name string
	}{
		{modeReverse, "reorder (reverse-i reduction; pure rounding, models a different SIMD/threadgroup order)"},
		{modeF16state, "f16state (forward order + f16 state round-trip each step; models f16 state storage)"},
	}

	res := result{
		Schema:        "qwen36-gdn-divergence-sensitivity/v1",
		Issue:         "token-3 drift (correctness parity); see token3-drift-investigation-2026-06-28.md",
		Host:          "windows/amd64 (CGO_ENABLED=0); device-independent, no Mac / no GPU / no 27B artifact",
		Model:         "Qwen3.6-27B q4_k_m GDN shapes (H=5120 nK=16 nV=48 kHd=vHd=128 K=4)",
		Tokens:        *tokens,
		Layers:        *layers,
		LogitScale:    logitScale,
		NearTieMargin: nearTie,
		RhoThreshold:  rhoStar,
	}

	for _, md := range modes {
		perLayer, final := runStackTraced(*tokens, *layers, md.m)
		implied := final * logitScale
		res.Modes = append(res.Modes, modeResult{
			Mode:              md.name,
			PerLayerRho:       perLayer,
			FinalRho:          final,
			ImpliedLogitDelta: implied,
			FlipsNearTie:      implied >= nearTie,
		})
	}

	// Verdict from the strongest bracket (f16state).
	strongest := res.Modes[len(res.Modes)-1]
	if strongest.FlipsNearTie {
		res.Verdict = "numerics-as-rounding is SUFFICIENT: even/at-least one modeled kernel-numerics " +
			"difference compounds to >= the observed 1.75-logit near-tie over the GDN stack, so the token-3 " +
			"flip is consistent with pure accumulation/precision divergence -> match llama.cpp's scan " +
			"reduction order / state dtype. Confirm the actual (layer,op) with the Mac per-layer probe."
	} else {
		res.Verdict = "numerics-as-rounding is INSUFFICIENT on its own: even the f16-state bracket stays " +
			"far below the rho needed to flip the 1.75-logit near-tie, so accumulated reduction-order/precision " +
			"divergence ALONE cannot explain token-3. The flip implies an ANOMALOUS (algorithmic/ordering) " +
			"divergence, not mere rounding -> the per-layer probe must find where cosine drops anomalously " +
			"(a real op mismatch), and a 1-ULP-floor threshold would miss it."
	}

	if *asJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(res)
		return
	}

	fmt.Printf("Qwen3.6-27B GDN reduction-order / precision sensitivity — token-3 drift (H1), device-independent\n")
	fmt.Printf("  shapes: H=%d nK=%d nV=%d kHd=%d vHd=%d convDim=%d valDim=%d K=%d\n", H, nK, nV, kHd, vHd, convDim, valDim, K)
	fmt.Printf("  stack:  %d GDN layers, %d carried positions (22 prompt + 2 agreed, predicting token 3)\n", *layers, *tokens)
	fmt.Printf("  near-tie: observed fak margin %.2f logits at |logit|~%.0f -> need rho >= %.4f (%.2f%%) of ||hidden|| to flip\n\n",
		nearTie, logitScale, rhoStar, rhoStar*100)
	for _, m := range res.Modes {
		fmt.Printf("  mode: %s\n", m.Mode)
		fmt.Printf("    per-layer relative hidden divergence (last token), layers 1..%d:\n      ", *layers)
		for i, v := range m.PerLayerRho {
			fmt.Printf("%.2e ", v)
			if (i+1)%8 == 0 {
				fmt.Printf("\n      ")
			}
		}
		fmt.Printf("\n    final rho = %.4e  -> implied |Δlogit| ~ %.4e  (flips 1.75 near-tie: %v)\n\n",
			m.FinalRho, m.ImpliedLogitDelta, m.FlipsNearTie)
	}
	fmt.Printf("  VERDICT: %s\n", res.Verdict)
}
