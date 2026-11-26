package model

// kimik31_kda_chunked_test.go — parity tests for the chunk-reformulated KDA
// prefill (ticket #14371 spine). Three pins:
//
//  1. chunkSize==1 is BIT-EXACT vs ForwardKimiK31KDAPrefillSeq (the length-1
//     chunk delegates to the oracle's own step, so the chunk plumbing —
//     projections, conv, gates, readout, output layout, state carry — is
//     proven exact, isolating any size>=2 mismatch to the reformulated math).
//  2. chunkSize in {2,4,8,16} matches the scan within the measured float32
//     re-association tolerance (the chunk form factors the per-token decay
//     out of the state sums; see the chunk invariant in kimik31_kda_chunked.go).
//  3. the chunked outputs also match the independent float64 naive
//     transcription (kimik31_kda_naive_test.go) on the tiny fixture, the same
//     oracle the scan itself is pinned against.
//
// MEASURED TOLERANCE (tiny 2x2 and synthetic 8x16 shapes, safe-gate and
// softplus forms, chunks 2/4/8/16/oversize): max relative output delta
// <= 1.32e-6 and max relative final-state delta <= 1.2e-7 against the scan
// (scale max(1,|want|) per element, the kimiK31KDAClose measure; the run's
// actual maxima are printed by the t.Logf in
// TestKimiK31KDAChunkedPrefillMatchesScan so the bound stays re-derivable).
// The test asserts 1e-4 / 1e-3 — two orders of headroom over the measured
// bound — so a future edit that genuinely breaks the reformulation trips the
// test while ordinary float32 re-association noise never does.

import (
	"math"
	"math/rand"
	"strconv"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestKimiK31KDAChunkedPrefillChunk1EqualsScan(t *testing.T) {
	t.Parallel()
	seq := []float32{1, 2, 3, 0.5, -1, 0.25, -0.5, 0.25, 1}
	const T = 3
	for _, tc := range []struct {
		name string
		lb   float32
	}{{"safeGate", -5.0}, {"softplus", 0}} {
		t.Run(tc.name, func(t *testing.T) {
			p := tinyKimiK31KDAParams(tc.lb)
			stA := NewGLM5NextKDALayerState(tNH, tD, tKern)
			got := ForwardKimiK31KDAChunkedPrefill(stA, &p, seq, T, 1)
			stB := NewGLM5NextKDALayerState(tNH, tD, tKern)
			want := ForwardKimiK31KDAPrefillSeq(stB, &p, seq, T)
			if len(got) != len(want) {
				t.Fatalf("chunked length %d != scan length %d", len(got), len(want))
			}
			for i := range got {
				if got[i] != want[i] {
					t.Fatalf("chunked out[%d] = %v, scan %v", i, got[i], want[i])
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
		})
	}
}

// chunkedMaxRel reports max_i |got-want| / max(1,|want|) — the same measure
// kimiK31KDAClose asserts with, aggregated over a slice.
func chunkedMaxRel(got, want []float32) float64 {
	if len(got) != len(want) {
		return math.Inf(1)
	}
	var worst float64
	for i := range got {
		diff := math.Abs(float64(got[i]) - float64(want[i]))
		scale := math.Max(1, math.Abs(float64(want[i])))
		if d := diff / scale; d > worst {
			worst = d
		}
	}
	return worst
}

// synthKimiK31KDAParams builds a deterministic pseudo-random KDA fixture at
// an arbitrary geometry (validated by the caller via ValidateKimiK31KDAParams)
// so the chunk reformulation is exercised beyond the hand-verifiable tiny
// shape without staging a K3.1 checkpoint.
func synthKimiK31KDAParams(nh, d, hid, kernel int, lb float32, seed int64) KimiK31KDAParams {
	rng := rand.New(rand.NewSource(seed))
	rnd := func(scale float32) float32 { return float32(rng.Float64()*2-1) * scale }
	dInner := nh * d
	mk := func(rows, cols int) []float32 {
		w := make([]float32, rows*cols)
		for i := range w {
			w[i] = rnd(0.5)
		}
		return w
	}
	conv := func() *GLM5NextKDAConvFilter {
		f := NewGLM5NextKDAConvFilter(dInner, kernel)
		for i := range f.Weight {
			f.Weight[i] = rnd(0.5)
		}
		return f
	}
	aFolded := make([]float32, nh)
	for h := range aFolded {
		aFolded[h] = -float32(0.25 + rng.Float64()*0.75)
	}
	onorm := make([]float32, d)
	for i := range onorm {
		onorm[i] = float32(0.5 + rng.Float64())
	}
	return KimiK31KDAParams{
		NumHeads:       nh,
		HeadDim:        d,
		ConvKernel:     kernel,
		HiddenSize:     hid,
		Wq:             mk(dInner, hid),
		Wk:             mk(dInner, hid),
		Wv:             mk(dInner, hid),
		ConvQ:          conv(),
		ConvK:          conv(),
		ConvV:          conv(),
		WFA:            mk(d, hid),
		WFB:            mk(dInner, d),
		DTBias:         mk(dInner, 1),
		AFolded:        aFolded,
		WBeta:          mk(nh, hid),
		GateLowerBound: lb,
		WGate:          mk(dInner, hid),
		ONorm:          onorm,
		WO:             mk(hid, dInner),
		RMSEps:         1e-5,
	}
}

func synthKimiK31KDASeq(hid, T int, seed int64) []float32 {
	rng := rand.New(rand.NewSource(seed))
	seq := make([]float32, T*hid)
	for i := range seq {
		seq[i] = float32(rng.Float64()*2 - 1)
	}
	return seq
}

// fak-test:runtime fast est=100ms lane=default
func TestKimiK31KDAChunkedPrefillMatchesScan(t *testing.T) {
	t.Parallel()
	const outTol = 1e-4
	const stateTol = 1e-3

	type shape struct {
		name               string
		nh, d, hid, kernel int
		seqSeed            int64
		sizes              []int
	}
	shapes := []shape{
		{name: "tiny", nh: tNH, d: tD, hid: tHid, kernel: tKern, seqSeed: 11, sizes: []int{7, 8}},
		{name: "synthetic-8x16", nh: 8, d: 16, hid: 32, kernel: 4, seqSeed: 23, sizes: []int{32, 33}},
	}

	var worstOut, worstState float64
	for _, sh := range shapes {
		for _, tc := range []struct {
			name string
			lb   float32
		}{{"safeGate", -5.0}, {"softplus", 0}} {
			for _, cs := range []int{2, 4, 8, 16, 64} {
				for _, T := range sh.sizes {
					t.Run(sh.name+"/"+tc.name+"/chunk"+strconv.Itoa(cs)+"/T"+strconv.Itoa(T), func(t *testing.T) {
						p := synthKimiK31KDAParams(sh.nh, sh.d, sh.hid, sh.kernel, tc.lb, sh.seqSeed)
						if err := ValidateKimiK31KDAParams(&p); err != nil {
							t.Fatalf("synthetic params invalid: %v", err)
						}
						seq := synthKimiK31KDASeq(sh.hid, T, sh.seqSeed+100)

						stA := NewGLM5NextKDALayerState(sh.nh, sh.d, sh.kernel)
						got := ForwardKimiK31KDAChunkedPrefill(stA, &p, seq, T, cs)
						stB := NewGLM5NextKDALayerState(sh.nh, sh.d, sh.kernel)
						want := ForwardKimiK31KDAPrefillSeq(stB, &p, seq, T)

						if dOut := chunkedMaxRel(got, want); dOut > outTol {
							t.Errorf("outputs rel delta %g exceeds %g", dOut, outTol)
						} else if dOut > worstOut {
							worstOut = dOut
						}
						dState := 0.0
						for h := 0; h < sh.nh; h++ {
							if d := chunkedMaxRel(stA.HeadMatrix(h), stB.HeadMatrix(h)); d > dState {
								dState = d
							}
						}
						if dState > stateTol {
							t.Errorf("final state rel delta %g exceeds %g", dState, stateTol)
						} else if dState > worstState {
							worstState = dState
						}
					})
				}
			}
		}
	}
	t.Logf("measured max rel: outputs %g, final state %g (asserted %g / %g)", worstOut, worstState, outTol, stateTol)
}

// fak-test:runtime fast est=10ms lane=default
func TestKimiK31KDAChunkedPrefillMatchesNaiveTranscription(t *testing.T) {
	t.Parallel()
	toks := [][]float64{{1, 2, 3}, {0.5, -1, 0.25}, {-0.5, 0.25, 1}, {0.75, 0.5, -0.5}, {-1, 1, 0.125}}
	T := len(toks)
	seq := make([]float32, 0, T*tHid)
	for _, tok := range toks {
		seq = append(seq, f32s(tok)...)
	}
	for _, tc := range []struct {
		name string
		lb   float32
	}{{"safeGate", -5.0}, {"softplus", 0}} {
		t.Run(tc.name, func(t *testing.T) {
			p := tinyKimiK31KDAParams(tc.lb)
			refOuts, _ := naiveKimiK31KDA(naiveFromTiny(p), toks)
			for _, cs := range []int{1, 2, 3, 4, 5} {
				st := NewGLM5NextKDALayerState(tNH, tD, tKern)
				got := ForwardKimiK31KDAChunkedPrefill(st, &p, seq, T, cs)
				for ti := range toks {
					if d := chunkedMaxRel(got[ti*tHid:(ti+1)*tHid], f32s(refOuts[ti])); d > 1e-4 {
						t.Errorf("chunk %d token %d rel delta %g exceeds 1e-4", cs, ti, d)
					}
				}
			}
		})
	}
}
