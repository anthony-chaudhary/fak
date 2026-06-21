package model

import "sync"

// prefill_attn.go — the balanced, allocation-free batched causal GQA attention used by the
// prefill paths. It replaces the per-token parFor that profiling (FAK_QPROFILE) exposed as
// the second-largest prefill cost — ~27% of Q8 prefill, despite attention being ~25x fewer
// MACs than the projection GEMMs. Two structural defects caused that:
//
//  1. TRIANGULAR LOAD IMBALANCE. Causal token t attends over base+t+1 positions, so its
//     work grows linearly in t. A contiguous parFor(P) hands worker W-1 the high-t tokens,
//     which do ~P/chunk more work than worker 0's low-t tokens; every core then waits on
//     that one slow chunk. Splitting (token,head) work units ROUND-ROBIN over the flat
//     u=t*nH+h index instead gives every worker an even spread of token positions, so the
//     per-worker totals match to within one unit (perfect balance is unnecessary; the
//     residual is the single most-expensive (t,h) unit, ~microseconds).
//
//  2. PER-(token,head) ALLOCATION. The old loop did `scores := make([]float32, nPos)` inside
//     the (t,h) loop — P*nH (=2304 at P=256) heap slices per layer, 30 layers per prefill.
//     Here each worker keeps ONE scratch buffer of the max width and reslices it per unit.
//
// The math is byte-for-byte the old computation — same scoreDot, same softmax, same
// in-order V accumulation per output — so the Q8 correctness gate (argmax-exact vs the HF
// oracle, logit-cosine vs f32) and the f32 oracle rungs are unaffected; only WHICH core
// computes WHICH (t,h) output, and the scratch lifetime, change.

// attnPrefillInto computes batched causal GQA attention for a prefill panel of P tokens into
// attnOut ([P, nH*hd], assumed zeroed by the caller). Q is [P, nH*hd]; Kl/Vl are the full KV
// cache as flat [nPos, nKV*hd] (stride w=nKV*hd). base is the number of cached positions
// before this panel (0 for a fresh prefill), so token t attends over base+t+1 positions.
// scoreDot is the (q·k) kernel (fdot in the fast path, dot in the legacy A/B path). grp is
// the GQA group size (nH/nKV) — passed in so it matches cfg.GroupSize() exactly. W is the
// layer's sliding-window bound (-1 = full causal, the default): query t (absolute base+t)
// then attends only keys [j0, base+t], j0=max(0, base+t-W+1). The prefill cache is
// contiguous (pos[j]==j), so the index is the absolute position; W=-1 keeps j0=0 exactly,
// reducing every byte to the pre-SWA loop. attnCap is the optional Gemma2 attention
// score soft-cap; zero keeps Llama-family scores unchanged.
func attnPrefillInto(attnOut, Q, Kl, Vl []float32, P, base, nH, hd, w, grp, W int, scale, attnCap float32, scoreDot func(a, b []float32) float32) {
	units := P * nH
	maxPos := base + P // widest scores row in this panel

	work := func(wkr, nw int) {
		scores := make([]float32, maxPos) // one scratch per worker, resliced per unit
		for u := wkr; u < units; u += nw {
			t := u / nH
			h := u % nH
			nPos := base + t + 1
			j0 := windowLoContig(nPos, base+t, W)
			kvh := h / grp
			qh := Q[t*nH*hd+h*hd : t*nH*hd+(h+1)*hd]
			sc := scores[:nPos-j0]
			for j := j0; j < nPos; j++ {
				kh := Kl[j*w+kvh*hd : j*w+(kvh+1)*hd]
				sc[j-j0] = scoreDot(qh, kh) * scale
			}
			softcapInPlace(sc, attnCap)
			softmaxInPlace(sc)
			out := attnOut[t*nH*hd+h*hd : t*nH*hd+(h+1)*hd]
			for j := j0; j < nPos; j++ {
				vh := Vl[j*w+kvh*hd : j*w+(kvh+1)*hd]
				saxpy(out, vh, sc[j-j0])
			}
		}
	}

	nw := numWorkers
	if nw > units {
		nw = units
	}
	if nw <= 1 {
		work(0, 1)
		return
	}
	var wg sync.WaitGroup
	for k := 0; k < nw; k++ {
		wg.Add(1)
		go func(k int) {
			defer wg.Done()
			work(k, nw)
		}(k)
	}
	wg.Wait()
}

// saxpy does out += a*x over the full length of out (== len(x)). Split into 8 independent
// updates so the compiler can keep the lanes in flight (the per-output accumulation order is
// unchanged vs the old scalar `out[d] += wj*vh[d]` loop: each out[d] still receives exactly
// one a*x[d] term per call, in the same call sequence). hd is a multiple of 8 (64) here.
func saxpy(out, x []float32, a float32) {
	n := len(out)
	i := 0
	for ; i+8 <= n; i += 8 {
		out[i] += a * x[i]
		out[i+1] += a * x[i+1]
		out[i+2] += a * x[i+2]
		out[i+3] += a * x[i+3]
		out[i+4] += a * x[i+4]
		out[i+5] += a * x[i+5]
		out[i+6] += a * x[i+6]
		out[i+7] += a * x[i+7]
	}
	for ; i < n; i++ {
		out[i] += a * x[i]
	}
}

func saxpy3(out0, out1, out2, x []float32, a0, a1, a2 float32) {
	if saxpy3Fast(out0, out1, out2, x, a0, a1, a2) {
		return
	}
	saxpy3scalar(out0, out1, out2, x, a0, a1, a2)
}

func saxpy3scalar(out0, out1, out2, x []float32, a0, a1, a2 float32) {
	n := len(out0)
	i := 0
	for ; i+8 <= n; i += 8 {
		x0, x1, x2, x3 := x[i], x[i+1], x[i+2], x[i+3]
		x4, x5, x6, x7 := x[i+4], x[i+5], x[i+6], x[i+7]
		out0[i] += a0 * x0
		out1[i] += a1 * x0
		out2[i] += a2 * x0
		out0[i+1] += a0 * x1
		out1[i+1] += a1 * x1
		out2[i+1] += a2 * x1
		out0[i+2] += a0 * x2
		out1[i+2] += a1 * x2
		out2[i+2] += a2 * x2
		out0[i+3] += a0 * x3
		out1[i+3] += a1 * x3
		out2[i+3] += a2 * x3
		out0[i+4] += a0 * x4
		out1[i+4] += a1 * x4
		out2[i+4] += a2 * x4
		out0[i+5] += a0 * x5
		out1[i+5] += a1 * x5
		out2[i+5] += a2 * x5
		out0[i+6] += a0 * x6
		out1[i+6] += a1 * x6
		out2[i+6] += a2 * x6
		out0[i+7] += a0 * x7
		out1[i+7] += a1 * x7
		out2[i+7] += a2 * x7
	}
	for ; i < n; i++ {
		v := x[i]
		out0[i] += a0 * v
		out1[i] += a1 * v
		out2[i] += a2 * v
	}
}
