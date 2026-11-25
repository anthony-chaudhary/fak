package ggufload

import (
	"math"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func ink2SessTiny() *Inkling2LoopModel {
	hp, w := ink2SessTinyModel()
	return &Inkling2LoopModel{HP: hp, Weights: w}
}

func TestInkling2HostSessionGreedyMatchesReferenceGreedy(t *testing.T) {
	m := ink2SessTiny()
	prompt := []int{1, 4, 2, 7, 3}
	want, err := Inkling2LoopGreedy(m, prompt, 6)
	if err != nil {
		t.Fatal(err)
	}
	got, st, err := Inkling2LoopGreedyTimed(m, nil, prompt, 6)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) || st.DecodeSteps != 5 {
		t.Fatalf("timed greedy = %v (%d decode steps), want %v (5)", got, st.DecodeSteps, want)
	}
}

func TestInkling2SessionPrefillStepIsFreshPerRequest(t *testing.T) {
	m := ink2SessTiny()
	prompt := []int{1, 4, 2, 7, 3}
	want, err := Inkling2LoopGreedy(m, prompt, 5)
	if err != nil {
		t.Fatal(err)
	}
	for req := 0; req < 2; req++ {
		s, err := NewInkling2LoopSession(m, nil)
		if err != nil {
			t.Fatal(err)
		}
		next, err := s.Prefill(prompt)
		if err != nil {
			t.Fatal(err)
		}
		got := []int{next}
		for len(got) < len(want) {
			if next, err = s.Step(next); err != nil {
				t.Fatal(err)
			}
			got = append(got, next)
		}
		if err := s.Close(); err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(got, want) {
			t.Fatalf("request %d: session stream %v, want %v", req, got, want)
		}
	}
}

func TestInkling2SessionPrefillRefusesEmptyPrompt(t *testing.T) {
	s, err := NewInkling2LoopSession(ink2SessTiny(), nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Prefill(nil); err == nil {
		t.Fatal("empty prefill accepted")
	}
}

type ink2SessEmbedding struct{ dim int }

func (e ink2SessEmbedding) Row(token int, dst []float32) {
	for i := range dst {
		dst[i] = float32(math.Sin(float64(token*e.dim + i + 1)))
	}
}

type ink2SessExperts struct {
	n int
	m []*model.Inkling2LoopDenseMatrix
}

func (e *ink2SessExperts) Experts() int { return e.n }
func (e *ink2SessExperts) Rows() int    { return e.m[0].R }
func (e *ink2SessExperts) Cols() int    { return e.m[0].C }
func (e *ink2SessExperts) ExpertMatVec(x int, dst, v []float32) {
	e.m[x].MatVec(dst, v)
}

func ink2SessMat(rows, cols, seed int) *model.Inkling2LoopDenseMatrix {
	m := &model.Inkling2LoopDenseMatrix{R: rows, C: cols, Data: make([]float32, rows*cols)}
	for i := range m.Data {
		m.Data[i] = float32(math.Sin(float64(seed*7919+i)*0.37)) * 0.3
	}
	return m
}

func ink2SessVec(n, seed int, base float32) []float32 {
	v := make([]float32, n)
	for i := range v {
		v[i] = base + 0.1*float32(math.Cos(float64(seed*31+i)))
	}
	return v
}

func ink2SessTinyModel() (model.Inkling2LoopHParams, *model.Inkling2LoopWeights) {
	hp := model.Inkling2LoopHParams{
		Hidden: 8, Layers: 6, Heads: 2, KVHeads: 1, HeadDim: 4, VocabUnpadded: 10,
		SlidingWindow: 3, RelLength: 2, RelExtent: 5, ConvKernel: 4, AttnResBlock: 2,
		Experts: 4, ExpertsUsed: 2, ExpertWidth: 3, CoreStart: 2, CoreEnd: 4, Passes: 3, GateRank: 2,
		LocalLayer: []bool{true, false, true, false, true, false},
		NormEps:    1e-6, QKClip: 2, RelClip: 1, LogScaleFloor: 2, LogScaleAlpha: 0.1,
		SituBeta: 4, SituLinearBeta: 25, ExpertWeightsScale: 8, FinalLogitSoftcap: 30,
	}
	w := &model.Inkling2LoopWeights{
		TokEmbd: ink2SessEmbedding{dim: hp.Hidden}, TokNorm: ink2SessVec(8, 1, 1),
		Output: ink2SessMat(12, 8, 2), OutputNorm: ink2SessVec(8, 3, 1), OutputResQuery: ink2SessVec(8, 4, 0),
	}
	kv := hp.KVHeads * hp.HeadDim
	for il := 0; il < hp.Layers; il++ {
		s := 100 * (il + 1)
		lw := model.Inkling2LoopLayerWeights{
			Wq: ink2SessMat(8, 8, s+1), Wk: ink2SessMat(kv, 8, s+2), Wv: ink2SessMat(kv, 8, s+3),
			Wo: ink2SessMat(8, 8, s+4), Gate: ink2SessMat(8, 8, s+5), Rel: ink2SessMat(4, 8, s+6),
			SinkProj: ink2SessMat(2, 8, s+7),
			QClip:    ink2SessVec(8, s+8, 1), RelClip: ink2SessVec(4, s+9, 1),
			KConv: ink2SessVec(4*kv, s+10, 0), VConv: ink2SessVec(4*kv, s+11, 0),
			AttnPostConv: ink2SessVec(32, s+12, 0), FFNPostConv: ink2SessVec(32, s+13, 0),
			FFNResQuery: ink2SessVec(8, s+14, 0), FFNNorm: ink2SessVec(8, s+15, 1),
			ExpProbsBias: ink2SessVec(4, s+16, 0), Router: ink2SessMat(4, 8, s+17),
		}
		extent := hp.RelExtent
		if hp.LocalLayer[il] {
			extent = hp.SlidingWindow
			lw.AttnNorm = ink2SessVec(8, s+18, 1)
		} else {
			lw.QInpNorm, lw.KInpNorm, lw.VInpNorm = ink2SessVec(8, s+19, 1), ink2SessVec(8, s+20, 1), ink2SessVec(8, s+21, 1)
			lw.KConvProj, lw.VConvProj = ink2SessMat(4, 8, s+22), ink2SessMat(4, 8, s+23)
		}
		lw.RelBias = ink2SessVec(extent*hp.RelLength, s+24, 0)
		if il > 0 {
			n := 3
			if hp.LocalLayer[il] {
				n = 1
			}
			for j := 0; j < n; j++ {
				lw.AttnResQuery = append(lw.AttnResQuery, ink2SessVec(8, s+25+j, 0))
			}
		}
		gu, dn := &ink2SessExperts{n: 4}, &ink2SessExperts{n: 4}
		for e := 0; e < 4; e++ {
			gu.m = append(gu.m, ink2SessMat(6, 8, s+30+e))
			dn.m = append(dn.m, ink2SessMat(8, 3, s+40+e))
		}
		lw.GateUp, lw.Down = gu, dn
		w.Layers = append(w.Layers, lw)
	}
	for c := 0; c < hp.CoreEnd-hp.CoreStart; c++ {
		var cw model.Inkling2LoopControllerWeights
		for j := 0; j < 4; j++ {
			s := 1000*(c+1) + 10*j
			cw.MsgQuery[j] = ink2SessVec(8, s, 0)
			cw.Down[j] = ink2SessMat(2, 16, s+1)
			cw.Gain[j] = ink2SessMat(8, 2, s+2)
			cw.Forget[j] = ink2SessMat(8, 2, s+3)
		}
		w.Controllers = append(w.Controllers, cw)
	}
	return hp, w
}
