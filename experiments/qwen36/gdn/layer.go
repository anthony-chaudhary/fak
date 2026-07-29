package gdn

import (
	"math"
	"math/rand"
	"runtime"
	"sync"
)

// LayerWeights holds one GDN layer's parameters. The five projection matrices (Wqkv,
// Wz, Wb, Wa, WOut) are the tensors internal/model/safetensors_quant.go:isQuantWeight
// stores as Q8 in the real loader; the control tensors (WIn, Conv, ALog, DtB, NormW)
// take the dequant->f32 path there and are f32 here too. Buffers are reused across
// layers (the shapes are layer-invariant) to avoid 48x large allocations.
type LayerWeights struct {
	Hidden int       // hidden size H these buffers were sized for
	WIn    []float32 // input_layernorm.weight  [H]            (f32 in the loader)
	Wqkv   []float32 // in_proj_qkv             [ConvDim, H]   (Q8)
	Wz     []float32 // in_proj_z               [ValDim, H]    (Q8)
	Wb     []float32 // in_proj_b               [NV, H]        (Q8)
	Wa     []float32 // in_proj_a               [NV, H]        (Q8)
	Conv   []float32 // conv1d.weight           [ConvDim, K]   (f32)
	ALog   []float32 // A_log                   [NV]           (f32)
	DtB    []float32 // dt_bias                 [NV]           (f32)
	NormW  []float32 // linear_attn.norm.weight [VHd] per head (f32)
	WOut   []float32 // out_proj                [H, ValDim]    (Q8)
}

// NewLayerWeights allocates one reusable layer-weight set for hidden size H.
func NewLayerWeights(hidden int) *LayerWeights {
	return &LayerWeights{
		Hidden: hidden,
		WIn:    make([]float32, hidden),
		Wqkv:   make([]float32, ConvDim*hidden),
		Wz:     make([]float32, ValDim*hidden),
		Wb:     make([]float32, NV*hidden),
		Wa:     make([]float32, NV*hidden),
		Conv:   make([]float32, ConvDim*K),
		ALog:   make([]float32, NV),
		DtB:    make([]float32, NV),
		NormW:  make([]float32, VHd),
		WOut:   make([]float32, hidden*ValDim),
	}
}

// Fill draws each weight matrix from a per-layer seeded normal with 1/sqrt(fan_in)
// scaling, so the projections behave like trained weights (bounded activations) without
// needing the 27B artifact. The draw is a pure function of `layer`, so two weight sets
// filled with the same layer index are bit-identical.
//
// aLogMean is the mean of the per-head A_log draw and is an explicit PARAMETER rather
// than a baked-in constant because the two callers legitimately differ: the depth-axis
// study pins it at -2.0 (g~0.88, effective memory ~8 positions) while the length-axis
// study sweeps it from a flag, and a near-1 decay (mean -5 -> g~0.99) is exactly the
// regime in which per-step error can compound across a long decode.
func (lw *LayerWeights) Fill(layer int, aLogMean float64) {
	r := rand.New(rand.NewSource(int64(0x9E3779B9 ^ layer)))
	gauss := func(s []float32, scale float32) {
		for i := range s {
			s[i] = float32(r.NormFloat64()) * scale
		}
	}
	// 1/sqrt(fan_in) scaling keeps projection outputs O(1).
	gauss(lw.WIn, 0.02) // norm gain delta around 0 -> (1+w) ~ 1
	gauss(lw.Wqkv, float32(1.0/math.Sqrt(float64(lw.Hidden))))
	gauss(lw.Wz, float32(1.0/math.Sqrt(float64(lw.Hidden))))
	gauss(lw.Wb, float32(1.0/math.Sqrt(float64(lw.Hidden))))
	gauss(lw.Wa, float32(1.0/math.Sqrt(float64(lw.Hidden))))
	gauss(lw.Conv, 0.5)
	gauss(lw.WOut, float32(1.0/math.Sqrt(float64(ValDim))))
	for h := 0; h < NV; h++ {
		lw.ALog[h] = float32(r.NormFloat64())*0.5 + float32(aLogMean)
		lw.DtB[h] = float32(r.NormFloat64()) * 0.2
	}
	for i := range lw.NormW {
		lw.NormW[i] = float32(r.NormFloat64()) * 0.02
	}
}

// ScanMode selects how the delta-rule recurrent scan rounds. It is the ONLY axis along
// which Layer's arithmetic varies; everything else is bit-identical across modes.
type ScanMode int

const (
	// ModeForward is the trunk's serial order: i ascending, f32 state.
	ModeForward ScanMode = iota
	// ModeReverse walks i descending — the same math in a different, equally valid
	// reduction order, so it isolates pure rounding (a different SIMD/threadgroup order).
	ModeReverse
	// ModeF16State keeps the forward order but round-trips the recurrent state through
	// f16 on every step, modelling a kernel that stores the accumulator in half precision.
	ModeF16State
)

// Layer runs one linear_attn (GDN) layer's forward over P tokens and returns its output
// [P, lw.Hidden] (to be added to the residual). It is the verbatim
// metal_prefill_hybrid_core.go body, parameterized only by `mode` for the recurrent
// scan's numerics. Recurrent and conv state start at zero (fresh-prefill precondition —
// prefill on this path is itself a token loop, so this is also the decode carry).
func Layer(lw *LayerWeights, X []float32, P int, mode ScanMode, eps float32) []float32 {
	h := lw.Hidden

	// input RMSNorm (1+w), identical across modes.
	Xn := make([]float32, P*h)
	for t := 0; t < P; t++ {
		RMSNormGain1p(Xn[t*h:(t+1)*h], X[t*h:(t+1)*h], lw.WIn, eps)
	}

	mixed := make([]float32, P*ConvDim)
	zAll := make([]float32, P*ValDim)
	bvec := make([]float32, P*NV)
	avec := make([]float32, P*NV)
	ParMatmul(mixed, Xn, lw.Wqkv, P, ConvDim, h)
	ParMatmul(zAll, Xn, lw.Wz, P, ValDim, h)
	ParMatmul(bvec, Xn, lw.Wb, P, NV, h)
	ParMatmul(avec, Xn, lw.Wa, P, NV, h)

	// causal depthwise conv1d + SiLU (verbatim core.go:145-169), fresh-prefill (no history).
	convOut := make([]float32, P*ConvDim)
	DepthwiseCausalSilu(convOut, mixed, lw.Conv, P, ConvDim, K)

	// q/k per-head L2-norm + 1/sqrt(KHd) query scale (verbatim core.go:186-201).
	//
	// The query scale is kept because the trunk has it, but it is NOT behaviourally
	// load-bearing HERE: it multiplies the whole readout od, and the gated RMSNorm below
	// divides each head's readout by its own RMS, so any global query scale cancels. Measured
	// on the smallLayer fixture with the scale changed to 1/KHd: at eps=0 the layer output is
	// unchanged to 7 significant figures (norm 0.15710974 vs 0.15710976, pure f32 rounding),
	// and it only moves at eps=1e-6 (norm 0.1142 vs 0.0317) because at this fixture's scale
	// the readout magnitude sits near the eps floor. So no behavioural test in this package
	// can pin this constant without actually pinning an eps artifact — it is pinned by
	// inspection against core.go instead.
	scale := float32(1.0 / math.Sqrt(float64(KHd)))
	repeat := NV / NK
	qNormAll := make([]float32, P*KeyDim)
	kNormAll := make([]float32, P*KeyDim)
	for t := 0; t < P; t++ {
		row := convOut[t*ConvDim : (t+1)*ConvDim]
		q := row[0:KeyDim]
		k := row[KeyDim : 2*KeyDim]
		qNorm := qNormAll[t*KeyDim : (t+1)*KeyDim]
		kNorm := kNormAll[t*KeyDim : (t+1)*KeyDim]
		for hd := 0; hd < NK; hd++ {
			L2NormInto(qNorm[hd*KHd:(hd+1)*KHd], q[hd*KHd:(hd+1)*KHd], 1e-6)
			L2NormInto(kNorm[hd*KHd:(hd+1)*KHd], k[hd*KHd:(hd+1)*KHd], 1e-6)
			for i := hd * KHd; i < (hd+1)*KHd; i++ {
				qNorm[i] *= scale
			}
		}
	}

	aExp := make([]float32, NV)
	for hd := 0; hd < NV; hd++ {
		aExp[hd] = float32(math.Exp(float64(lw.ALog[hd])))
	}

	// delta-rule recurrent scan (verbatim core.go:202-246) — the ONLY mode-dependent op.
	core := make([]float32, P*ValDim)
	var wg sync.WaitGroup
	workers := runtime.GOMAXPROCS(0)
	chunk := (NV + workers - 1) / workers
	for wk := 0; wk < workers; wk++ {
		hlo := wk * chunk
		hhi := hlo + chunk
		if hhi > NV {
			hhi = NV
		}
		if hlo >= hhi {
			break
		}
		wg.Add(1)
		go func(hlo, hhi int) {
			defer wg.Done()
			st := make([]float32, KHd*VHd)
			kvmem := make([]float32, VHd)
			delta := make([]float32, VHd)
			for hd := hlo; hd < hhi; hd++ {
				for i := range st {
					st[i] = 0
				}
				kh := hd / repeat
				a := aExp[hd]
				dtB := lw.DtB[hd]
				for t := 0; t < P; t++ {
					row := convOut[t*ConvDim : (t+1)*ConvDim]
					qn := qNormAll[t*KeyDim+kh*KHd : t*KeyDim+(kh+1)*KHd]
					kn := kNormAll[t*KeyDim+kh*KHd : t*KeyDim+(kh+1)*KHd]
					vh := row[2*KeyDim+hd*VHd : 2*KeyDim+(hd+1)*VHd]
					bt := Sigmoidf(bvec[t*NV+hd])
					dt := Softplus(avec[t*NV+hd] + dtB)
					g := float32(math.Exp(float64(-a * dt)))
					for i := range st {
						st[i] *= g
					}
					if mode == ModeF16State {
						for i := range st {
							st[i] = QuantF16(st[i])
						}
					}
					for d := range kvmem {
						kvmem[d] = 0
					}
					accumulate(kvmem, st, kn, mode) // kvmem[d] = sum_i st[i*VHd+d]*kn[i]
					for d := 0; d < VHd; d++ {
						delta[d] = (vh[d] - kvmem[d]) * bt
					}
					od := core[t*ValDim+hd*VHd : t*ValDim+(hd+1)*VHd]
					readout(od, st, kn, qn, delta, mode) // st += k(x)delta; od += sum_i st[i]*qn[i]
					if mode == ModeF16State {
						for i := range st {
							st[i] = QuantF16(st[i])
						}
					}
				}
			}
		}(hlo, hhi)
	}
	wg.Wait()

	// gated RMSNorm readout (verbatim core.go:247-258), identical across modes.
	for t := 0; t < P; t++ {
		for hd := 0; hd < NV; hd++ {
			RMSNormGatedInPlace(
				core[t*ValDim+hd*VHd:t*ValDim+(hd+1)*VHd],
				lw.NormW,
				zAll[t*ValDim+hd*VHd:t*ValDim+(hd+1)*VHd],
				eps,
			)
		}
	}

	o := make([]float32, P*h)
	ParMatmul(o, core, lw.WOut, P, h, ValDim)
	return o
}

// keyOrder returns the first key-dim index and the stride that walk 0..KHd-1 in the
// reduction order `mode` implies: ascending for the trunk's serial order, descending for
// ModeReverse. Both reductions below drive their i-loop from it, which is what makes "the
// only difference between the modes is the ORDER of the same additions" a structural
// property of this file rather than a claim you have to verify by diffing two loop bodies.
func keyOrder(mode ScanMode) (first, stride int) {
	if mode == ModeReverse {
		return KHd - 1, -1
	}
	return 0, 1
}

// accumulate computes kvmem[d] = sum_i st[i*VHd+d]*kn[i] in the given reduction order.
func accumulate(kvmem, st, kn []float32, mode ScanMode) {
	i, stride := keyOrder(mode)
	for n := 0; n < KHd; n++ {
		ki := kn[i]
		base := i * VHd
		for d := 0; d < VHd; d++ {
			kvmem[d] += st[base+d] * ki
		}
		i += stride
	}
}

// readout applies the state update st[i*VHd+d] += kn[i]*delta[d] (order-independent:
// disjoint blocks) and the readout reduction od[d] += sum_i st[i*VHd+d]*qn[i] in the
// given order.
func readout(od, st, kn, qn, delta []float32, mode ScanMode) {
	i, stride := keyOrder(mode)
	for n := 0; n < KHd; n++ {
		ki := kn[i]
		qi := qn[i]
		base := i * VHd
		for d := 0; d < VHd; d++ {
			st[base+d] += ki * delta[d]
			od[d] += st[base+d] * qi
		}
		i += stride
	}
}
