package model

// kimik31_kda_naive_test.go — the independent float64 transcription of the
// K3.1 KDA recurrence used as the parity oracle for the L5 host oracle. It is
// written straight from the documented formula list (llama.cpp master
// build_kda_layer projection/gate/readout order + build_delta_net_autoregressive
// decay/predict/delta/update/readout order), not from the oracle's code, and
// accumulates in float64 throughout so it doubles as the numerical reference.

import "math"

type naiveKimiK31KDAParams struct {
	nh, d, hid, kernel  int
	wq, wk, wv          [][]float64 // [dInner][hid]
	convQ, convK, convV [][]float64 // [dInner][kernel], tap 0 = oldest
	wfa                 [][]float64 // [d][hid]
	wfb                 [][]float64 // [dInner][d]
	dtbias              []float64
	aFolded             []float64
	wbeta               [][]float64 // [nh][hid]
	lb                  float64
	wgate               [][]float64 // [dInner][hid]
	onorm               []float64
	wo                  [][]float64 // [hid][dInner]
	eps                 float64
}

func naiveSig(v float64) float64 { return 1 / (1 + math.Exp(-v)) }

func naiveSoftplus(v float64) float64 {
	if v > 20 {
		return v
	}
	return math.Log1p(math.Exp(v))
}

func naiveSilu(v float64) float64 { return v / (1 + math.Exp(-v)) }

func naiveProj(w [][]float64, x []float64) []float64 {
	out := make([]float64, len(w))
	for i, row := range w {
		var s float64
		for j, wj := range row {
			s += wj * x[j]
		}
		out[i] = s
	}
	return out
}

// naiveL2 is FLA's l2norm: x * rsqrt(sum(x^2) + eps) (sum, not mean).
func naiveL2(x []float64, eps float64) []float64 {
	var ss float64
	for _, v := range x {
		ss += v * v
	}
	inv := 1 / math.Sqrt(ss+eps)
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = v * inv
	}
	return out
}

// naiveFromTiny reshapes the shared tiny fixture into 2-D float64 form.
func naiveFromTiny(p KimiK31KDAParams) naiveKimiK31KDAParams {
	res := func(rows, cols int, flat []float32) [][]float64 {
		m := make([][]float64, rows)
		for i := 0; i < rows; i++ {
			m[i] = make([]float64, cols)
			for j := 0; j < cols; j++ {
				m[i][j] = float64(flat[i*cols+j])
			}
		}
		return m
	}
	convReshape := func(f *GLM5NextKDAConvFilter) [][]float64 {
		m := make([][]float64, f.Dim)
		for i := 0; i < f.Dim; i++ {
			m[i] = make([]float64, f.FilterWidth)
			for tap := 0; tap < f.FilterWidth; tap++ {
				m[i][tap] = float64(f.Weight[i*f.FilterWidth+tap])
			}
		}
		return m
	}
	return naiveKimiK31KDAParams{
		nh: p.NumHeads, d: p.HeadDim, hid: p.HiddenSize, kernel: p.ConvKernel,
		wq:      res(tDI, tHid, p.Wq),
		wk:      res(tDI, tHid, p.Wk),
		wv:      res(tDI, tHid, p.Wv),
		convQ:   convReshape(p.ConvQ),
		convK:   convReshape(p.ConvK),
		convV:   convReshape(p.ConvV),
		wfa:     res(tD, tHid, p.WFA),
		wfb:     res(tDI, tD, p.WFB),
		dtbias:  f64s(p.DTBias),
		aFolded: f64s(p.AFolded),
		wbeta:   res(tNH, tHid, p.WBeta),
		lb:      float64(p.GateLowerBound),
		wgate:   res(tDI, tHid, p.WGate),
		onorm:   f64s(p.ONorm),
		wo:      res(tHid, tDI, p.WO),
		eps:     float64(p.RMSEps),
	}
}

// naiveKimiK31KDA runs the documented recurrence over toks and returns the
// per-token outputs plus the final per-head state matrices S[h][i][c].
func naiveKimiK31KDA(p naiveKimiK31KDAParams, toks [][]float64) ([][]float64, [][][]float64) {
	di := p.nh * p.d
	states := make([][][]float64, p.nh)
	for h := range states {
		states[h] = make([][]float64, p.d)
		for i := range states[h] {
			states[h][i] = make([]float64, p.d)
		}
	}
	histQ := make([][]float64, 0, p.kernel-1)
	histK := make([][]float64, 0, p.kernel-1)
	histV := make([][]float64, 0, p.kernel-1)

	// Causal depthwise conv over the window [hist..., current] then SiLU;
	// hist keeps the last kernel-1 PRE-conv raw rows (oldest first), left
	// zero-padded while fewer than kernel-1 tokens have been seen.
	conv := func(w [][]float64, raw []float64, hist *[][]float64) []float64 {
		win := make([][]float64, 0, p.kernel)
		for pad := len(*hist); pad < p.kernel-1; pad++ {
			win = append(win, make([]float64, di))
		}
		win = append(win, *hist...)
		win = append(win, raw)
		out := make([]float64, di)
		for ch := 0; ch < di; ch++ {
			var s float64
			for tap := 0; tap < p.kernel; tap++ {
				s += w[ch][tap] * win[tap][ch]
			}
			out[ch] = naiveSilu(s)
		}
		next := make([][]float64, 0, p.kernel-1)
		if len(*hist) > 0 {
			next = append(next, (*hist)[1:]...)
		}
		next = append(next, raw)
		*hist = next
		return out
	}

	outs := make([][]float64, 0, len(toks))
	for _, x := range toks {
		qraw := naiveProj(p.wq, x)
		kraw := naiveProj(p.wk, x)
		vraw := naiveProj(p.wv, x)
		qc := conv(p.convQ, qraw, &histQ)
		kc := conv(p.convK, kraw, &histK)
		vc := conv(p.convV, vraw, &histV)

		fa := naiveProj(p.wfa, x)
		fraw := naiveProj(p.wfb, fa)
		for i := range fraw {
			fraw[i] += p.dtbias[i]
		}
		betaraw := naiveProj(p.wbeta, x)
		gateraw := naiveProj(p.wgate, x)

		y := make([]float64, di)
		qScale := 1 / math.Sqrt(float64(p.d))
		for h := 0; h < p.nh; h++ {
			base := h * p.d
			ql := naiveL2(qc[base:base+p.d], p.eps)
			kl := naiveL2(kc[base:base+p.d], p.eps)
			vh := vc[base : base+p.d]
			for i := range ql {
				ql[i] *= qScale
			}

			alpha := make([]float64, p.d)
			if p.lb != 0 {
				expAlog := -p.aFolded[h]
				for c := 0; c < p.d; c++ {
					alpha[c] = math.Exp(p.lb * naiveSig(expAlog*fraw[base+c]))
				}
			} else {
				for c := 0; c < p.d; c++ {
					alpha[c] = math.Exp(p.aFolded[h] * naiveSoftplus(fraw[base+c]))
				}
			}
			beta := naiveSig(betaraw[h])

			S := states[h]
			for i := 0; i < p.d; i++ {
				for c := 0; c < p.d; c++ {
					S[i][c] *= alpha[c]
				}
			}
			kv := make([]float64, p.d)
			for i := 0; i < p.d; i++ {
				for c := 0; c < p.d; c++ {
					kv[c] += S[i][c] * kl[i]
				}
			}
			dl := make([]float64, p.d)
			for c := 0; c < p.d; c++ {
				dl[c] = beta * (vh[c] - kv[c])
			}
			for i := 0; i < p.d; i++ {
				for c := 0; c < p.d; c++ {
					S[i][c] += kl[i] * dl[c]
				}
			}
			o := make([]float64, p.d)
			for i := 0; i < p.d; i++ {
				for c := 0; c < p.d; c++ {
					o[c] += S[i][c] * ql[i]
				}
			}
			var ms float64
			for _, v := range o {
				ms += v * v
			}
			rms := math.Sqrt(ms/float64(p.d) + p.eps)
			for c := 0; c < p.d; c++ {
				y[base+c] = o[c] / rms * p.onorm[c] * naiveSig(gateraw[base+c])
			}
		}
		outs = append(outs, naiveProj(p.wo, y))
	}
	return outs, states
}
