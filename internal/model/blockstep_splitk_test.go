package model

// blockstep_splitk_test.go — parity witness for blockStep's parallel decode attend
// (#13693). attnDecodeStep replaced the serial per-head loop; its exact regime must stay
// bit-identical to that loop at any worker count, and its split-K regime (chunks over
// the KV span plus a log-sum-exp merge) must stay within f32 rounding of it and decode
// the same greedy tokens.

import (
	"math"
	"testing"
)

// pinDecodeWorkers pins the worker budget for one test and restores it afterwards.
func pinDecodeWorkers(t *testing.T, n int) {
	t.Helper()
	workerBudgetMu.Lock()
	prevN, prevSrc := numWorkers, workerBudgetSource
	workerBudgetMu.Unlock()
	if err := SetWorkers(n); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		workerBudgetMu.Lock()
		numWorkers, workerBudgetSource = prevN, prevSrc
		workerBudgetMu.Unlock()
	})
}

// pinSplitK sets the split-K regime knobs for one test and restores them afterwards.
func pinSplitK(t *testing.T, enabled bool, minPos int) {
	t.Helper()
	prevOn, prevMin := decodeSplitKEnabled, decodeSplitKMinPos
	decodeSplitKEnabled, decodeSplitKMinPos = enabled, minPos
	t.Cleanup(func() { decodeSplitKEnabled, decodeSplitKMinPos = prevOn, prevMin })
}

// serialDecodeAttendRef is the pre-#13693 blockStep attend loop verbatim: one head at a
// time, the full row decoded per head on q8, scalar dot, softcap, sink-aware softmax,
// then saxpy over j in order.
func serialDecodeAttendRef(c *KVCache, l int, q []float32, lo, nPos, nH, hd, grp int, scale, softcap float32, sinks []float32) []float32 {
	w := c.kvStride()
	out := make([]float32, nH*hd)
	scores := make([]float32, nPos-lo)
	krow, vrow := make([]float32, w), make([]float32, w)
	for h := 0; h < nH; h++ {
		kvh := h / grp
		qh := q[h*hd : (h+1)*hd]
		for j := lo; j < nPos; j++ {
			var kh []float32
			if c.quantized() {
				c.decodeRowInto(l, j, true, krow)
				kh = krow[kvh*hd : (kvh+1)*hd]
			} else {
				kh = c.K[l][j*w+kvh*hd : j*w+(kvh+1)*hd]
			}
			scores[j-lo] = dot(qh, kh) * scale
		}
		softcapInPlace(scores, softcap)
		softmaxWithSink(scores, sinks, h)
		oh := out[h*hd : (h+1)*hd]
		for j := lo; j < nPos; j++ {
			var vh []float32
			if c.quantized() {
				c.decodeRowInto(l, j, false, vrow)
				vh = vrow[kvh*hd : (kvh+1)*hd]
			} else {
				vh = c.V[l][j*w+kvh*hd : j*w+(kvh+1)*hd]
			}
			saxpy(oh, vh, scores[j-lo])
		}
	}
	return out
}

func splitKTestCache(cfg Config, prec KVPrecision, nPos int, seed uint64) *KVCache {
	c := NewKVCacheWithPrecision(cfg, prec)
	w := cfg.NumKVHeads * cfg.HeadDim
	next := func() float32 {
		seed = seed*6364136223846793005 + 1442695040888963407
		return (float32(seed>>40)/float32(1<<24))*2 - 1
	}
	k, v := make([]float32, w), make([]float32, w)
	for j := 0; j < nPos; j++ {
		for i := range k {
			k[i], v[i] = next()*2, next()
		}
		c.appendKV(0, k, v)
	}
	return c
}

func TestBlockStepSplitKKernelParity(t *testing.T) {
	cfg := Config{HiddenSize: 768, NumLayers: 1, NumHeads: 12, NumKVHeads: 4, HeadDim: 64}
	const nPos = 1000
	nH, hd, grp, w := cfg.NumHeads, cfg.HeadDim, cfg.GroupSize(), cfg.NumKVHeads*cfg.HeadDim
	scale := float32(1 / math.Sqrt(float64(hd)))
	q := make([]float32, nH*hd)
	for i := range q {
		q[i] = float32(math.Sin(float64(i)*0.37)) * 3
	}
	sinks := make([]float32, nH)
	for h := range sinks {
		sinks[h] = float32(h%5) - 1
	}
	cases := []struct {
		name    string
		prec    KVPrecision
		lo      int
		softcap float32
		sinks   []float32
	}{
		{"f32", KVPrecisionFP32, 0, 0, nil},
		{"f32-window", KVPrecisionFP32, 371, 0, nil},
		{"f32-softcap-sinks", KVPrecisionFP32, 0, 4, sinks},
		{"q8", KVPrecisionQ8_0, 0, 0, nil},
		{"q8-window-sinks", KVPrecisionQ8_0, 129, 0, sinks},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := splitKTestCache(cfg, tc.prec, nPos, 0x13693)
			ref := serialDecodeAttendRef(c, 0, q, tc.lo, nPos, nH, hd, grp, scale, tc.softcap, tc.sinks)
			run := func(nw int, splitK bool) []float32 {
				pinSplitK(t, splitK, 16)
				out := make([]float32, nH*hd)
				var sc splitKDecodeScratch
				attnDecodeStep(out, q, nil, newDecodeKVRows(c, 0, w, hd), decodeAttnStepArgs{
					l: 0, lo: tc.lo, nPos: nPos, qpos: nPos - 1, nH: nH, hd: hd, grp: grp,
					scale: scale, softcap: tc.softcap, cfg: &cfg, sinks: tc.sinks, nw: nw,
				}, &sc)
				return out
			}
			// Exact regime: bit-identical to the serial loop, serial or parallel.
			for _, nw := range []int{1, 3, 8} {
				got := run(nw, false)
				for i := range ref {
					if got[i] != ref[i] {
						t.Fatalf("exact nw=%d: out[%d]=%v, serial ref %v (want bit-identical)", nw, i, got[i], ref[i])
					}
				}
			}
			// Split-K regime: within f32 rounding of the serial loop.
			got := run(8, true)
			var maxD, maxRef float64
			for i := range ref {
				maxD = math.Max(maxD, math.Abs(float64(got[i]-ref[i])))
				maxRef = math.Max(maxRef, math.Abs(float64(ref[i])))
			}
			if rel := maxD / maxRef; rel > 1e-5 {
				t.Fatalf("split-K drifted from serial loop: max|Δ|=%.3e rel=%.3e (want <= 1e-5)", maxD, rel)
			} else {
				t.Logf("split-K vs serial: max|Δ|=%.3e rel=%.3e", maxD, rel)
			}
		})
	}
}

// TestBlockStepSplitKDecodeParity: a 128-token greedy decode at a context long enough
// to take the split-K regime is token-identical to the exact (serial-arithmetic) decode.
//
// fak-test:runtime medium est=10s lane=default
func TestBlockStepSplitKDecodeParity(t *testing.T) {
	m := NewSynthetic(Config{
		HiddenSize:       128,
		NumLayers:        2,
		NumHeads:         6,
		NumKVHeads:       2,
		HeadDim:          32,
		IntermediateSize: 256,
		VocabSize:        96,
		RMSNormEps:       1e-5,
		RopeTheta:        10000,
		EOSTokenID:       95,
	})
	pinDecodeWorkers(t, 8)
	prompt := make([]int, 400) // 6 heads*400*32 > parThreshold: every step takes split-K
	for i := range prompt {
		prompt[i] = (i*7)%90 + 1
	}
	decode := func(splitK bool) ([]int, []float32) {
		pinSplitK(t, splitK, 32)
		s := m.NewSession()
		logits := s.Prefill(prompt)
		toks := make([]int, 0, 128)
		for i := 0; i < 128; i++ {
			tok := argmax(logits)
			toks = append(toks, tok)
			logits = s.Step(tok)
		}
		return toks, logits
	}
	refToks, refLogits := decode(false)
	gotToks, gotLogits := decode(true)
	for i := range refToks {
		if gotToks[i] != refToks[i] {
			t.Fatalf("split-K greedy decode diverged at step %d: got %d, serial %d", i, gotToks[i], refToks[i])
		}
	}
	var maxD float64
	for i := range refLogits {
		maxD = math.Max(maxD, math.Abs(float64(gotLogits[i]-refLogits[i])))
	}
	t.Logf("128 greedy tokens identical; final-step logits max|Δ|=%.3e", maxD)
}
