package model

// kimik31_kda_test.go — fixed-tensor parity tests for the K3.1 KDA layer
// forward (ticket 1611-kimi-k31 leaf L5). No runnable reference dump is
// obtainable without a staged K3.1 artifact (PF-11), so the recurrence
// contract is pinned two ways: (1) hand-derived closed-form values for a
// single token from a zero state, and (2) an independent float64
// transcription of the documented llama.cpp recurrence (build_kda_layer +
// build_delta_net_autoregressive order, see kimik31_kda_naive_test.go) over
// a multi-token sequence, for both decay-gate forms.

import (
	"errors"
	"fmt"
	"math"
	"testing"
)

// Tiny geometry shared by the L5 tests: 2 heads x 2 dims, hidden 3, conv
// kernel 2 — small enough that every intermediate is hand-verifiable.
const (
	tNH   = 2
	tD    = 2
	tHid  = 3
	tKern = 2
	tDI   = tNH * tD // 4
)

// tinyKimiK31KDAParams builds the shared fixture. Weights are chosen so the
// projections have sparse structure (hand-computable intermediates):
//
//	qRaw = [x0, x1, x2, 0]      kRaw = [x2, 0, 0, 0]      vRaw = [x0+x1+x2, 0, 0, 0]
//	fa   = [x0, x1]             fRaw(pre-bias) = [fa0, fa1, fa0+fa1, 0]
//	betaRaw = [x0, x2]          gateRaw = [x1, x2, 0, 0]
//	ConvQ mixes the previous raw row (tap0=0.5) with the current (tap1=1.0);
//	ConvK/ConvV are identity-on-current (tap0=0).
func tinyKimiK31KDAParams(lb float32) KimiK31KDAParams {
	conv := func(tap0, tap1 float32) *GLM5NextKDAConvFilter {
		f := NewGLM5NextKDAConvFilter(tDI, tKern)
		for ch := 0; ch < tDI; ch++ {
			f.Weight[ch*tKern+0] = tap0
			f.Weight[ch*tKern+1] = tap1
		}
		return f
	}
	return KimiK31KDAParams{
		NumHeads:       tNH,
		HeadDim:        tD,
		ConvKernel:     tKern,
		HiddenSize:     tHid,
		Wq:             []float32{1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0},
		Wk:             []float32{0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		Wv:             []float32{1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		ConvQ:          conv(0.5, 1.0),
		ConvK:          conv(0, 1.0),
		ConvV:          conv(0, 1.0),
		WFA:            []float32{1, 0, 0, 0, 1, 0},
		WFB:            []float32{1, 0, 0, 1, 1, 1, 0, 0},
		DTBias:         []float32{0.5, -0.25, 0.1, 0},
		AFolded:        []float32{-0.5, -0.25},
		WBeta:          []float32{1, 0, 0, 0, 0, 1},
		GateLowerBound: lb,
		WGate:          []float32{0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0},
		ONorm:          []float32{1, 2},
		WO:             []float32{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0},
		RMSEps:         1e-5,
	}
}

func f64s(in []float32) []float64 {
	out := make([]float64, len(in))
	for i, v := range in {
		out[i] = float64(v)
	}
	return out
}

func f32s(in []float64) []float32 {
	out := make([]float32, len(in))
	for i, v := range in {
		out[i] = float32(v)
	}
	return out
}

func kimiK31KDAClose(t *testing.T, label string, got, want, relTol float64) {
	t.Helper()
	diff := math.Abs(got - want)
	scale := math.Max(1, math.Abs(want))
	if diff > relTol*scale {
		t.Fatalf("%s: got %v want %v (diff %g exceeds tol %g)", label, got, want, diff, relTol*scale)
	}
}

func TestKimiK31KDAWeightNames(t *testing.T) {
	// Pins the L2 canonical vocabulary byte-for-byte (layer 7, arbitrary).
	want := KimiK31KDAWeightNames{
		QProj:    "model.layers.7.self_attn.q_proj.weight",
		KProj:    "model.layers.7.self_attn.k_proj.weight",
		VProj:    "model.layers.7.self_attn.v_proj.weight",
		QConv:    "model.layers.7.self_attn.q_conv1d.weight",
		KConv:    "model.layers.7.self_attn.k_conv1d.weight",
		VConv:    "model.layers.7.self_attn.v_conv1d.weight",
		FAProj:   "model.layers.7.self_attn.f_a_proj.weight",
		FBProj:   "model.layers.7.self_attn.f_b_proj.weight",
		BetaProj: "model.layers.7.self_attn.b_proj.weight",
		ALog:     "model.layers.7.self_attn.A_log",
		DTBias:   "model.layers.7.self_attn.dt_bias",
		GateProj: "model.layers.7.self_attn.g_proj.weight",
		ONorm:    "model.layers.7.self_attn.o_norm",
		OProj:    "model.layers.7.self_attn.o_proj.weight",
	}
	if got := KimiK31KDAWeightNamesFor(7); got != want {
		t.Fatalf("weight names mismatch:\n got %+v\nwant %+v", got, want)
	}
}

func TestKimiK31KDAValidation(t *testing.T) {
	valid := tinyKimiK31KDAParams(-5.0)
	if err := ValidateKimiK31KDAParams(&valid); err != nil {
		t.Fatalf("valid params rejected: %v", err)
	}

	cases := []struct {
		name   string
		mutate func(*KimiK31KDAParams)
	}{
		{"nil", func(p *KimiK31KDAParams) { *p = KimiK31KDAParams{} }},
		{"bad-geometry", func(p *KimiK31KDAParams) { p.NumHeads = 0 }},
		{"rmseps-zero", func(p *KimiK31KDAParams) { p.RMSEps = 0 }},
		{"wq-short", func(p *KimiK31KDAParams) { p.Wq = p.Wq[:len(p.Wq)-1] }},
		{"wk-long", func(p *KimiK31KDAParams) { p.Wk = append(p.Wk, 0) }},
		{"wfa-short", func(p *KimiK31KDAParams) { p.WFA = p.WFA[:1] }},
		{"wfb-short", func(p *KimiK31KDAParams) { p.WFB = p.WFB[:3] }},
		{"dtbias-short", func(p *KimiK31KDAParams) { p.DTBias = p.DTBias[:2] }},
		{"afolded-short", func(p *KimiK31KDAParams) { p.AFolded = p.AFolded[:1] }},
		{"afolded-positive-domain", func(p *KimiK31KDAParams) { p.AFolded[0] = 0.5 }},
		{"wbeta-short", func(p *KimiK31KDAParams) { p.WBeta = p.WBeta[:2] }},
		{"wgate-short", func(p *KimiK31KDAParams) { p.WGate = p.WGate[:5] }},
		{"onorm-short", func(p *KimiK31KDAParams) { p.ONorm = p.ONorm[:1] }},
		{"wo-short", func(p *KimiK31KDAParams) { p.WO = p.WO[:7] }},
		{"convq-nil", func(p *KimiK31KDAParams) { p.ConvQ = nil }},
		{"convk-wrong-dim", func(p *KimiK31KDAParams) { p.ConvK.Dim = tDI + 1 }},
		{"convv-wrong-weight-len", func(p *KimiK31KDAParams) { p.ConvV.Weight = p.ConvV.Weight[:3] }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p := tinyKimiK31KDAParams(-5.0)
			tc.mutate(&p)
			err := ValidateKimiK31KDAParams(&p)
			if !errors.Is(err, ErrKimiK31KDAParamShape) {
				t.Fatalf("expected ErrKimiK31KDAParamShape, got %v", err)
			}
		})
	}
}

// TestKimiK31KDAStepClosedFormZeroState pins the single-token recurrence from
// a zero state against HAND-DERIVED closed forms (no transcription involved):
// with the fixture weights, token x=[1,2,3] leaves exactly one nonzero state
// entry S[head0][0][0] = 18*sigmoid(1)/sqrt(9+eps) and one nonzero output
// channel out[0] = sigmoid(2)*o0/sqrt(o0^2/2+eps) (gateRaw[0] = x1 = 2).
func TestKimiK31KDAStepClosedFormZeroState(t *testing.T) {
	const eps = 1e-5
	sig := func(v float64) float64 { return 1 / (1 + math.Exp(-v)) }
	silu := func(v float64) float64 { return v / (1 + math.Exp(-v)) }

	p := tinyKimiK31KDAParams(-5.0)
	st := NewGLM5NextKDALayerState(tNH, tD, tKern)
	out := ForwardKimiK31KDAStep(st, &p, []float32{1, 2, 3})

	// Hand derivation (head 0, channel 0; all other state entries stay 0).
	// With x=[1,2,3] and empty conv history:
	//   qRaw=[1,2,3,0]  kRaw=[3,0,0,0]  vRaw=[6,0,0,0]
	//   qc = silu(qRaw) = [s1,s2,s3,0]   kc = [s3,0,0,0]   vc = [s6,0,0,0]
	//     where s1=silu(1), s2=silu(2), s3=silu(3), s6=silu(6)
	//   khat[0] = s3/sqrt(s3^2+eps);  beta0 = sigmoid(x0) = sigmoid(1)
	//   zero state -> kv = 0 -> delta[0] = beta0*s6
	//   S[0][0] = khat[0]*delta[0] = s3*beta0*s6/sqrt(s3^2+eps)
	s1, s2 := silu(1), silu(2)
	s3, s6 := silu(3), silu(6)
	wantS00 := s3 * sig(1) * s6 / math.Sqrt(s3*s3+eps)
	kimiK31KDAClose(t, "S[head0][0][0]", float64(st.HeadMatrix(0)[0]), wantS00, 1e-5)
	for i := 1; i < tD*tD; i++ {
		kimiK31KDAClose(t, fmt.Sprintf("S[head0][%d] zero", i), float64(st.HeadMatrix(0)[i]), 0, 1e-9)
	}
	for i := 0; i < tD*tD; i++ {
		kimiK31KDAClose(t, fmt.Sprintf("S[head1][%d] zero", i), float64(st.HeadMatrix(1)[i]), 0, 1e-9)
	}

	// Readout: qhat[0] = s1/sqrt(s1^2+s2^2+eps)/sqrt(2); o0 = S[0][0]*qhat[0];
	// y0 = o0 * rsqrt(o0^2/2 + eps) * ONorm[0] * sigmoid(gateRaw[0]=x1=2).
	qhat0 := s1 / math.Sqrt(s1*s1+s2*s2+eps) / math.Sqrt(2)
	o0 := wantS00 * qhat0
	wantOut0 := sig(2) * o0 / math.Sqrt(o0*o0/2+eps)
	kimiK31KDAClose(t, "out[0]", float64(out[0]), wantOut0, 1e-5)
	kimiK31KDAClose(t, "out[1] zero", float64(out[1]), 0, 1e-9)
	kimiK31KDAClose(t, "out[2] zero", float64(out[2]), 0, 1e-9)
}

// TestKimiK31KDAStepMatchesNaiveTranscription runs a 3-token sequence through
// the oracle and compares every output and the final state matrices against
// the independent float64 transcription, for BOTH decay-gate forms
// (safe-gate lb=-5.0 and key-absent softplus lb=0).
func TestKimiK31KDAStepMatchesNaiveTranscription(t *testing.T) {
	toks := [][]float64{{1, 2, 3}, {0.5, -1, 0.25}, {-0.5, 0.25, 1}}
	for _, tc := range []struct {
		name string
		lb   float32
	}{{"safeGate", -5.0}, {"softplus", 0}} {
		t.Run(tc.name, func(t *testing.T) {
			p := tinyKimiK31KDAParams(tc.lb)
			st := NewGLM5NextKDALayerState(tNH, tD, tKern)
			refOuts, refStates := naiveKimiK31KDA(naiveFromTiny(p), toks)
			for ti, tok := range toks {
				out := ForwardKimiK31KDAStep(st, &p, f32s(tok))
				for j := range out {
					kimiK31KDAClose(t, fmt.Sprintf("token %d out[%d]", ti, j),
						float64(out[j]), refOuts[ti][j], 1e-4)
				}
			}
			for h := 0; h < tNH; h++ {
				S := st.HeadMatrix(h)
				for i := 0; i < tD; i++ {
					for c := 0; c < tD; c++ {
						kimiK31KDAClose(t, fmt.Sprintf("state head %d [%d][%d]", h, i, c),
							float64(S[i*tD+c]), refStates[h][i][c], 1e-4)
					}
				}
			}
		})
	}
}

// TestKimiK31KDAPrefillSeqEqualsSteps checks the prefill wrapper reproduces
// successive single-token steps bit-for-bit on identical states.
func TestKimiK31KDAPrefillSeqEqualsSteps(t *testing.T) {
	p := tinyKimiK31KDAParams(-5.0)
	seq := []float32{1, 2, 3, 0.5, -1, 0.25, -0.5, 0.25, 1}
	T := 3

	stA := NewGLM5NextKDALayerState(tNH, tD, tKern)
	got := ForwardKimiK31KDAPrefillSeq(stA, &p, seq, T)

	stB := NewGLM5NextKDALayerState(tNH, tD, tKern)
	var want []float32
	for t := 0; t < T; t++ {
		want = append(want, ForwardKimiK31KDAStep(stB, &p, seq[t*tHid:(t+1)*tHid])...)
	}
	if len(got) != len(want) {
		t.Fatalf("prefill length %d != step length %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("prefill out[%d] = %v, steps %v", i, got[i], want[i])
		}
	}
	for h := 0; h < tNH; h++ {
		a, b := stA.HeadMatrix(h), stB.HeadMatrix(h)
		for i := range a {
			if a[i] != b[i] {
				t.Fatalf("state diverged at head %d entry %d: %v vs %v", h, i, a[i], b[i])
			}
		}
	}
}

// TestKimiK31KDAConfigDispatch pins the layer_types tokens the L8 composite
// will switch on.
func TestKimiK31KDAConfigDispatch(t *testing.T) {
	cfg := Config{LayerTypes: []string{"kda", "dsa", "kda"}}
	if !cfg.IsKimiK31KDALayer(0) || !cfg.IsKimiK31KDALayer(2) {
		t.Fatal("layers 0 and 2 must classify as KDA")
	}
	if cfg.IsKimiK31KDALayer(1) || cfg.IsKimiK31KDALayer(3) || cfg.IsKimiK31KDALayer(-1) {
		t.Fatal("non-KDA / out-of-range layers must not classify as KDA")
	}
	if !cfg.IsKimiK31DSALayer(1) {
		t.Fatal("layer 1 must classify as DSA")
	}
	if cfg.IsKimiK31DSALayer(0) || cfg.IsKimiK31DSALayer(2) {
		t.Fatal("KDA layers must not classify as DSA")
	}
	empty := Config{}
	if empty.IsKimiK31KDALayer(0) || empty.IsKimiK31DSALayer(0) {
		t.Fatal("empty layer_types must classify no layer as KDA/DSA")
	}
}
