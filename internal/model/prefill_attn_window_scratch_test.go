package model

import (
	"math"
	"sync"
	"testing"
)

// Original full-prefix CPU implementation is the exact-output/allocation reference.
func fullScratchPrefillReference(attnOut, Q, Kl, Vl []float32, P, base, nH, hd, w, grp, W, layer int, scale, attnCap float32, scoreDot func(a, b []float32) float32, obs AttnObserver) {
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
			qh := packedHead(Q, t, nH*hd, h, hd)
			sc := scores[:nPos-j0]
			fillAttentionScores(sc, qh, Kl, j0, nPos, w, kvh, hd, scale, scoreDot)
			softcapInPlace(sc, attnCap)
			softmaxInPlace(sc)
			if obs != nil { // #852: prefill token t sits at absolute position base+t
				emitAttnRow(obs, layer, base+t, h, j0, sc)
			}
			out := packedHead(attnOut, t, nH*hd, h, hd)
			accumulateAttentionValues(out, Vl, sc, j0, nPos, w, kvh, hd)
		}
	}

	nw := currentWorkerCount()
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

func TestPrefillWindowScratchParityAndAllocation(t *testing.T) {
	old := NumWorkers()
	defer SetWorkers(old)
	for _, workers := range []int{1, 2} {
		if err := SetWorkers(workers); err != nil {
			t.Fatal(err)
		}
		for _, tc := range []struct{ base, rows, window int }{{7, 3, -1}, {0, 3, 32}, {65, 3, 8}, {65, 3, 1}, {1024, 2, 8}} {
			q := make([]float32, tc.rows*16)
			k := make([]float32, (tc.base+tc.rows)*8)
			v := make([]float32, len(k))
			for i := range q {
				q[i] = float32(i%11-5) / 16
			}
			for i := range k {
				k[i] = float32(i%13-6) / 16
				v[i] = float32(i%17-8) / 16
			}
			got, want := make([]float32, len(q)), make([]float32, len(q))
			run := func(reference bool) {
				out := got
				f := attnPrefillInto
				if reference {
					out = want
					f = fullScratchPrefillReference
				}
				clear(out)
				f(out, q, k, v, tc.rows, tc.base, 2, 8, 8, 2, tc.window, 0, 0.25, 0, dot, nil)
			}
			run(true)
			run(false)
			for i := range got {
				if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
					t.Fatalf("workers=%d case=%+v output=%d", workers, tc, i)
				}
			}
			if tc.base == 1024 || tc.window < 0 {
				before := testing.Benchmark(func(b *testing.B) {
					for i := 0; i < b.N; i++ {
						run(true)
					}
				}).AllocedBytesPerOp()
				after := testing.Benchmark(func(b *testing.B) {
					for i := 0; i < b.N; i++ {
						run(false)
					}
				}).AllocedBytesPerOp()
				t.Logf("engine=fak-native CPU workers=%d base=%d window=%d before=%d after=%d B/op exact=true", workers, tc.base, tc.window, before, after)
				if tc.base == 1024 && before-after < int64(workers*(tc.base+tc.rows-tc.window)*4) {
					t.Fatalf("insufficient allocation reduction %d -> %d", before, after)
				}
				if tc.window < 0 && after > before {
					t.Fatalf("causal allocation grew %d -> %d", before, after)
				}
			}
		}
	}
	// W=0 already panics on empty softmax; preserve rather than widen this patch.
	if err := SetWorkers(1); err != nil {
		t.Fatal(err)
	}
	for _, f := range []func([]float32, []float32, []float32, []float32, int, int, int, int, int, int, int, int, float32, float32, func([]float32, []float32) float32, AttnObserver){fullScratchPrefillReference, attnPrefillInto} {
		func() {
			defer func() {
				if recover() == nil {
					t.Error("zero-window panic changed")
				}
			}()
			f(make([]float32, 8), make([]float32, 8), make([]float32, 8), make([]float32, 8), 1, 0, 1, 8, 8, 1, 0, 0, 1, 0, dot, nil)
		}()
	}
}
