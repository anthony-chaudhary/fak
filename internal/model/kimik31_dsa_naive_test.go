package model

// kimik31_dsa_naive_test.go — the independent float64 transcription of the
// Kimi-K3.1 Gated DSA recurrence (ticket 1611-kimi-k31 leaf L6 parity
// witness). It re-derives the whole layer from the pinned contract
// (kimik31_dsa.go header: llama.cpp kimi-k31.cpp build_dsa_layer) with its
// OWN matvec/norms/relu/top-k/softmax/sigmoid in float64 and serial
// accumulation — sharing no helper code with the oracle (not even
// kdaMatVecMul), so agreement within tolerance cross-checks the recurrence
// itself. It mirrors the oracle's pinned op order (projections serial; K
// absorption over the nope dim; indexer heads outer/dims inner with
// non-positive ReLU terms skipped; attention cells ascending; value
// projection inner product before the weighted sum). F32->F64 weight
// conversion is exact, so both sides consume identical numbers.
//
// No runnable reference dump exists yet (no staged K3.1 artifact — PF-11
// stays gated on staging, as the study records); this transcription is the
// contract pin the study's witness bar calls for in that case.

import (
	"math"
	"math/rand"
	"sort"
	"testing"
)

type naiveDSAGeom struct {
	NH, QL, KL, Nope, Rope, VD, Hid, IH, ID, TopK int
	Eps, IdxEps                                   float64
}

type naiveDSAWeights struct {
	WQa, WQaNorm, WQb      []float64
	WKvA, WKvANorm         []float64
	WKb, WVb, WO           []float64
	WGate                  []float64 // nil = no gate
	WIdxQ, WIdxK, WIdxProj []float64
	IdxKNormW, IdxKNormB   []float64
}

func f32sTo64s(s []float32) []float64 {
	if s == nil {
		return nil // preserve the optional-gate sentinel
	}
	out := make([]float64, len(s))
	for i, v := range s {
		out[i] = float64(v)
	}
	return out
}

func naiveDSAMatVec(w, x []float64, outDim, inDim int) []float64 {
	out := make([]float64, outDim)
	for o := 0; o < outDim; o++ {
		var s float64
		row := o * inDim
		for j := 0; j < inDim; j++ {
			s += w[row+j] * x[j]
		}
		out[o] = s
	}
	return out
}

func naiveDSADot(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func naiveDSARMSNorm(x, w []float64, eps float64) []float64 {
	var ss float64
	for _, v := range x {
		ss += v * v
	}
	inv := 1.0 / math.Sqrt(ss/float64(len(x))+eps)
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = v * inv * w[i]
	}
	return out
}

func naiveDSALayerNorm(x, w, b []float64, eps float64) []float64 {
	var mean float64
	for _, v := range x {
		mean += v
	}
	mean /= float64(len(x))
	var ss float64
	for _, v := range x {
		d := v - mean
		ss += d * d
	}
	inv := 1.0 / math.Sqrt(ss/float64(len(x))+eps)
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = (v-mean)*inv*w[i] + b[i]
	}
	return out
}

// naiveDSATopK ranks by (score desc, index asc) — the same documented
// deterministic tie-break as SelectKimiK31DSATopK.
func naiveDSATopK(scores []float64, topK int) []int {
	n := len(scores)
	if n == 0 || topK <= 0 {
		return nil
	}
	k := topK
	if k > n {
		k = n
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool {
		sa, sb := scores[idx[a]], scores[idx[b]]
		if sa != sb {
			return sa > sb
		}
		return idx[a] < idx[b]
	})
	sel := make([]int, k)
	copy(sel, idx[:k])
	sort.Ints(sel)
	return sel
}

func naiveDSASoftmax(s []float64) {
	m := s[0]
	for _, v := range s[1:] {
		if v > m {
			m = v
		}
	}
	var sum float64
	for i, v := range s {
		s[i] = math.Exp(v - m)
		sum += s[i]
	}
	for i := range s {
		s[i] /= sum
	}
}

func naiveDSASigmoid(z float64) float64 { return 1.0 / (1.0 + math.Exp(-z)) }

// naiveKimiK31DSASequence runs T tokens causally through the Gated DSA
// recurrence in float64; it returns per-token outputs ([T x Hidden]) and
// per-token selections.
func naiveKimiK31DSASequence(g naiveDSAGeom, w naiveDSAWeights, xSeq []float64, T int) ([][]float64, [][]int) {
	headK := g.Nope + g.Rope
	latentDim := g.KL + g.Rope
	kqScale := 1.0 / math.Sqrt(float64(headK))
	preScale := 1.0 / math.Sqrt(float64(g.IH*g.ID))
	latent, indexK := []float64{}, []float64{}
	outs := make([][]float64, T)
	sels := make([][]int, T)
	for t := 0; t < T; t++ {
		x := xSeq[t*g.Hid : (t+1)*g.Hid]

		qr := naiveDSARMSNorm(naiveDSAMatVec(w.WQa, x, g.QL, g.Hid), w.WQaNorm, g.Eps)
		qFull := naiveDSAMatVec(w.WQb, qr, g.NH*headK, g.QL)

		latRaw := naiveDSAMatVec(w.WKvA, x, latentDim, g.Hid)
		kvC := naiveDSARMSNorm(latRaw[:g.KL], w.WKvANorm, g.Eps)
		row := make([]float64, latentDim)
		copy(row, kvC)
		copy(row[g.KL:], latRaw[g.KL:])
		ik := naiveDSALayerNorm(naiveDSAMatVec(w.WIdxK, x, g.ID, g.Hid), w.IdxKNormW, w.IdxKNormB, g.IdxEps)
		latent = append(latent, row...)
		indexK = append(indexK, ik...)

		iq := naiveDSAMatVec(w.WIdxQ, qr, g.IH*g.ID, g.QL)
		iw := naiveDSAMatVec(w.WIdxProj, x, g.IH, g.Hid)
		for i := range iw {
			iw[i] *= preScale
		}
		nCells := len(latent) / latentDim
		scores := make([]float64, nCells)
		for tau := 0; tau < nCells; tau++ {
			ikRow := indexK[tau*g.ID : (tau+1)*g.ID]
			var s float64
			for h := 0; h < g.IH; h++ {
				d := naiveDSADot(iq[h*g.ID:(h+1)*g.ID], ikRow)
				if d > 0 {
					s += iw[h] * d
				}
			}
			scores[tau] = s
		}
		sel := naiveDSATopK(scores, g.TopK)
		sels[t] = sel

		out := make([]float64, g.NH*g.VD)
		for h := 0; h < g.NH; h++ {
			qBase := h * headK
			qNope := qFull[qBase : qBase+g.Nope]
			qPe := qFull[qBase+g.Nope : qBase+headK]
			qAbs := make([]float64, g.KL)
			for j := 0; j < g.KL; j++ {
				var s float64
				for i := 0; i < g.Nope; i++ {
					s += w.WKb[(h*g.Nope+i)*g.KL+j] * qNope[i]
				}
				qAbs[j] = s
			}
			wts := make([]float64, len(sel))
			for i, tau := range sel {
				rowOff := tau * latentDim
				s := naiveDSADot(qAbs, latent[rowOff:rowOff+g.KL]) +
					naiveDSADot(qPe, latent[rowOff+g.KL:rowOff+latentDim])
				wts[i] = s * kqScale
			}
			naiveDSASoftmax(wts)
			for v := 0; v < g.VD; v++ {
				var acc float64
				for i, tau := range sel {
					wi := wts[i]
					if wi == 0 {
						continue
					}
					rowOff := tau * latentDim
					var vp float64
					wRow := (h*g.VD + v) * g.KL
					for kv := 0; kv < g.KL; kv++ {
						vp += w.WVb[wRow+kv] * latent[rowOff+kv]
					}
					acc += wi * vp
				}
				out[h*g.VD+v] = acc
			}
		}
		if w.WGate != nil {
			gate := naiveDSAMatVec(w.WGate, x, g.NH*g.VD, g.Hid)
			for i := range out {
				out[i] *= naiveDSASigmoid(gate[i])
			}
		}
		outs[t] = naiveDSAMatVec(w.WO, out, g.Hid, g.NH*g.VD)
	}
	return outs, sels
}

// naiveDSAFromOracle converts a validated oracle param set into the naive
// transcription's geometry + weights (exact f32->f64).
func naiveDSAFromOracle(p *KimiK31DSAParams) (naiveDSAGeom, naiveDSAWeights) {
	g := naiveDSAGeom{
		NH: p.NumHeads, QL: p.QLoRA, KL: p.KVLoRA, Nope: p.QkNope, Rope: p.QkRope,
		VD: p.VDim, Hid: p.Hidden, IH: p.IdxHeads, ID: p.IdxDim, TopK: p.TopK,
		Eps: float64(p.RMSEps), IdxEps: float64(p.IdxEps),
	}
	w := naiveDSAWeights{
		WQa: f32sTo64s(p.WQa), WQaNorm: f32sTo64s(p.WQaNorm), WQb: f32sTo64s(p.WQb),
		WKvA: f32sTo64s(p.WKvA), WKvANorm: f32sTo64s(p.WKvANorm),
		WKb: f32sTo64s(p.WKb), WVb: f32sTo64s(p.WVb), WO: f32sTo64s(p.WO),
		WGate: f32sTo64s(p.WGate),
		WIdxQ: f32sTo64s(p.WIdxQ), WIdxK: f32sTo64s(p.WIdxK),
		WIdxProj:  f32sTo64s(p.WIdxProj),
		IdxKNormW: f32sTo64s(p.IdxKNormW), IdxKNormB: f32sTo64s(p.IdxKNormB),
	}
	return g, w
}

func TestKimiK31DSAStepMatchesNaiveTranscription(t *testing.T) {
	p := dsaTinyParams(1611, true)
	if err := ValidateKimiK31DSAParams(p); err != nil {
		t.Fatalf("fixture must validate: %v", err)
	}
	rnd := rand.New(rand.NewSource(1611))
	const T = 5
	xSeq := make([]float32, T*p.Hidden)
	for i := range xSeq {
		xSeq[i] = rnd.Float32()*2 - 1
	}

	st := NewKimiK31DSALayerState(T, p.KVLoRA+p.QkRope, p.IdxDim)
	got := ForwardKimiK31DSAPrefillSeq(st, p, xSeq, T)

	g, w := naiveDSAFromOracle(p)
	want, _ := naiveKimiK31DSASequence(g, w, f32sTo64s(xSeq), T)
	bad := 0
	for tok := 0; tok < T; tok++ {
		for c := 0; c < p.Hidden; c++ {
			a, b := got[tok*p.Hidden+c], want[tok][c]
			diff := math.Abs(float64(a) - b)
			if diff > 1e-5+1e-3*math.Abs(b) {
				t.Errorf("token %d ch %d: got %v, want %v (diff %v)", tok, c, a, b, diff)
				bad++
				if bad > 8 {
					t.Fatal("too many divergent elements")
				}
			}
		}
	}
}

// TestKimiK31DSASelectionAgreesAcrossPrecision pins the top-k selection at
// the fixture level: the float64 transcription's selection (well-separated
// boundary, asserted) must drive the oracle's shared-indexer entry point to
// the SAME output as the oracle's internal float32 selection.
func TestKimiK31DSASelectionAgreesAcrossPrecision(t *testing.T) {
	p := dsaTinyParams(1611, false)
	if err := ValidateKimiK31DSAParams(p); err != nil {
		t.Fatalf("fixture must validate: %v", err)
	}
	rnd := rand.New(rand.NewSource(1611))
	const T = 5
	xSeq := make([]float32, T*p.Hidden)
	for i := range xSeq {
		xSeq[i] = rnd.Float32()*2 - 1
	}

	g, w := naiveDSAFromOracle(p)
	_, sels := naiveKimiK31DSASequence(g, w, f32sTo64s(xSeq), T)

	latentDim := p.KVLoRA + p.QkRope
	stA := NewKimiK31DSALayerState(T, latentDim, p.IdxDim)
	ref := ForwardKimiK31DSAPrefillSeq(stA, p, xSeq, T)

	stB := NewKimiK31DSALayerState(T, latentDim, p.IdxDim)
	for tok := 0; tok < T; tok++ {
		gotTok := ForwardKimiK31DSAStepWithSelection(stB, p, xSeq[tok*p.Hidden:(tok+1)*p.Hidden], sels[tok])
		for c := 0; c < p.Hidden; c++ {
			if ref[tok*p.Hidden+c] != gotTok[c] {
				t.Fatalf("token %d ch %d: selection-driven output %v != internal %v (f32/f64 selections disagree)",
					tok, c, gotTok[c], ref[tok*p.Hidden+c])
			}
		}
	}
}

// TestKimiK31DSAKvBDualFormBitEqual proves the combined kv_b_proj slab and
// the split k_b/v_b halves are interchangeable through the real kernel:
// slicing the combined slab per head (the SGLang unflatten/split layout,
// head h block = rows [h*(nope+v), (h+1)*(nope+v)), first nope rows K, rest
// V) must reproduce the split-form outputs BIT-for-bit.
func TestKimiK31DSAKvBDualFormBitEqual(t *testing.T) {
	base := dsaTinyParams(99, true)
	if err := ValidateKimiK31DSAParams(base); err != nil {
		t.Fatalf("fixture must validate: %v", err)
	}
	nope, v, kl := base.QkNope, base.VDim, base.KVLoRA
	block := nope + v

	// Build the combined slab [NH*block x KL] from the split halves.
	slab := make([]float32, base.NumHeads*block*kl)
	for h := 0; h < base.NumHeads; h++ {
		copy(slab[(h*block)*kl:(h*block+nope)*kl], base.WKb[(h*nope)*kl:(h*nope+nope)*kl])
		copy(slab[(h*block+nope)*kl:(h*block+block)*kl], base.WVb[(h*v)*kl:(h*v+v)*kl])
	}
	// Split it back out, exactly as SGLang's post-load hook does.
	splitP := *base
	splitP.WKb = make([]float32, base.NumHeads*nope*kl)
	splitP.WVb = make([]float32, base.NumHeads*v*kl)
	for h := 0; h < base.NumHeads; h++ {
		copy(splitP.WKb[(h*nope)*kl:(h*nope+nope)*kl], slab[(h*block)*kl:(h*block+nope)*kl])
		copy(splitP.WVb[(h*v)*kl:(h*v+v)*kl], slab[(h*block+nope)*kl:(h*block+block)*kl])
	}
	if err := ValidateKimiK31DSAParams(&splitP); err != nil {
		t.Fatalf("split params must validate: %v", err)
	}

	rnd := rand.New(rand.NewSource(7))
	const T = 3
	xSeq := make([]float32, T*base.Hidden)
	for i := range xSeq {
		xSeq[i] = rnd.Float32()*2 - 1
	}
	latentDim := base.KVLoRA + base.QkRope
	stA := NewKimiK31DSALayerState(T, latentDim, base.IdxDim)
	outA := ForwardKimiK31DSAPrefillSeq(stA, base, xSeq, T)
	stB := NewKimiK31DSALayerState(T, latentDim, base.IdxDim)
	outB := ForwardKimiK31DSAPrefillSeq(stB, &splitP, xSeq, T)
	for i := range outA {
		if outA[i] != outB[i] {
			t.Fatalf("combined vs split kv-b diverged at %d: %v vs %v", i, outA[i], outB[i])
		}
	}
}
