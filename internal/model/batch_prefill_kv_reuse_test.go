package model

import (
	"math"
	"testing"
)

func rectKVReuseFixture(quantized bool) (Config, []*KVCache, []int, []float32) {
	const B, P = 3, 8
	cfg := Config{NumLayers: 1, HiddenSize: 192, NumHeads: 6, NumKVHeads: 2, HeadDim: 32}
	base := []int{0, 5, 2}
	caches := make([]*KVCache, B)
	for b := range caches {
		c := NewKVCache(cfg)
		if quantized {
			c = NewKVCacheWithPrecision(cfg, KVPrecisionQ8_0)
		}
		rows, width := base[b]+P, cfg.NumKVHeads*cfg.HeadDim
		k, v := make([]float32, rows*width), make([]float32, rows*width)
		for i := range k {
			k[i] = float32((i+b*7)%19-9) / 17
			v[i] = float32((i+b*11)%23-11) / 13
		}
		c.appendBatchedKV(0, k, v, rows, width)
		caches[b] = c
	}
	q := make([]float32, B*P*cfg.NumHeads*cfg.HeadDim)
	for i := range q {
		q[i] = float32(i%29-14) / 31
	}
	return cfg, caches, base, q
}

func TestRectPrefillBuiltinKVReuse(t *testing.T) {
	oldWorkers, oldSource := currentWorkerBudget()
	oldFuse, oldSIMD := attnGQAFuse, attnFdot3SIMD
	t.Cleanup(func() {
		workerBudgetMu.Lock()
		numWorkers, workerBudgetSource = oldWorkers, oldSource
		workerBudgetMu.Unlock()
		attnGQAFuse, attnFdot3SIMD = oldFuse, oldSIMD
	})
	const P = 8
	for _, quantized := range []bool{false, true} {
		cfg, caches, base, q := rectKVReuseFixture(quantized)
		qBefore := append([]float32(nil), q...)
		var before [3][2][]float32
		for b, c := range caches {
			k, v := c.attentionRows(0)
			before[b] = [2][]float32{append([]float32(nil), k...), append([]float32(nil), v...)}
		}
		for _, workers := range []int{1, 4, 64} {
			if err := SetWorkers(workers); err != nil {
				t.Fatal(err)
			}
			for _, window := range []int{-1, 1, 4} {
				got, want := make([]float32, len(q)), make([]float32, len(q))
				attnPrefillMultiInto(want, q, caches, base, 0, P, cfg.NumHeads, cfg.HeadDim, 64, 3, window, 0.25, dot, nil)
				attnPrefillMultiBuiltinInto(got, q, caches, base, 0, P, cfg.NumHeads, cfg.HeadDim, 64, 3, window, 0.25, nil)
				for i := range got {
					if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
						t.Fatalf("plain: quantized=%v workers=%d window=%d index=%d", quantized, workers, window, i)
					}
				}
			}
			for _, fuse := range []bool{false, true} {
				attnGQAFuse = fuse
				for _, simd := range []bool{false, true} {
					attnFdot3SIMD = simd
					dot3 := fdot3scalar
					if simd && len(caches) >= attnFdot3SIMDMinBatch {
						dot3 = fdot3SIMD
					}
					for _, window := range []int{-1, 1, 4} {
						got, want := make([]float32, len(q)), make([]float32, len(q))
						attnPrefillMultiGQAInto(want, q, caches, base, 0, P, cfg.NumHeads, cfg.HeadDim, 64, 3, window, 0.25, fdot, dot3, nil)
						attnPrefillMultiGQABuiltinInto(got, q, caches, base, 0, P, cfg.NumHeads, cfg.HeadDim, 64, 3, window, 0.25, nil)
						for i := range got {
							if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
								t.Fatalf("quantized=%v workers=%d fuse=%v simd=%v window=%d index=%d", quantized, workers, fuse, simd, window, i)
							}
						}
					}
				}
			}
		}
		for i := range q {
			if math.Float32bits(q[i]) != math.Float32bits(qBefore[i]) {
				t.Fatal("query panel mutated")
			}
		}
		for b, c := range caches {
			k, v := c.attentionRows(0)
			for axis, values := range [2][]float32{k, v} {
				for i := range values {
					if math.Float32bits(values[i]) != math.Float32bits(before[b][axis][i]) {
						t.Fatal("cache mutated")
					}
				}
			}
		}
	}

	// A single worker should decode each cache once, not P*nKV times. Both
	// score scratch owners are warmed before counting; no timing assertion.
	if err := SetWorkers(1); err != nil {
		t.Fatal(err)
	}
	attnGQAFuse, attnFdot3SIMD = true, false
	cfg, caches, base, q := rectKVReuseFixture(true)
	out := make([]float32, len(q))
	var oldScratch, newScratch [][]float32
	oldRun := func() {
		clear(out)
		oldScratch = attnPrefillMultiGQAInto(out, q, caches, base, 0, P, cfg.NumHeads, cfg.HeadDim, 64, 3, -1, 0.25, fdot, fdot3scalar, oldScratch)
	}
	newRun := func() {
		clear(out)
		newScratch = attnPrefillMultiGQABuiltinInto(out, q, caches, base, 0, P, cfg.NumHeads, cfg.HeadDim, 64, 3, -1, 0.25, newScratch)
	}
	oldRun()
	newRun()
	beforeAllocs := testing.AllocsPerRun(1, oldRun)
	afterAllocs := testing.AllocsPerRun(1, newRun)
	// Every removed decode allocated at least K and V; the row temporary is
	// deliberately not counted in this lower bound.
	if beforeAllocs-afterAllocs < float64(2*len(caches)*(P*cfg.NumKVHeads-1)) {
		t.Fatalf("whole-layer decode allocations not removed: old=%g new=%g", beforeAllocs, afterAllocs)
	}
	oldScratch, newScratch = nil, nil
	oldRun = func() {
		clear(out)
		oldScratch = attnPrefillMultiInto(out, q, caches, base, 0, P, cfg.NumHeads, cfg.HeadDim, 64, 3, -1, 0.25, dot, oldScratch)
	}
	newRun = func() {
		clear(out)
		newScratch = attnPrefillMultiBuiltinInto(out, q, caches, base, 0, P, cfg.NumHeads, cfg.HeadDim, 64, 3, -1, 0.25, newScratch)
	}
	oldRun()
	newRun()
	beforeAllocs = testing.AllocsPerRun(1, oldRun)
	afterAllocs = testing.AllocsPerRun(1, newRun)
	if beforeAllocs-afterAllocs < float64(2*len(caches)*(P*cfg.NumHeads-1)) {
		t.Fatalf("plain whole-layer decode allocations not removed: old=%g new=%g", beforeAllocs, afterAllocs)
	}
}

func TestRectPrefillGenericCallbackKeepsFreshKVReads(t *testing.T) {
	oldWorkers, oldSource := currentWorkerBudget()
	t.Cleanup(func() {
		workerBudgetMu.Lock()
		numWorkers, workerBudgetSource = oldWorkers, oldSource
		workerBudgetMu.Unlock()
	})
	if err := SetWorkers(1); err != nil {
		t.Fatal(err)
	}
	const P, width = 2, 32
	cfg := Config{NumLayers: 1, HiddenSize: width, NumHeads: 1, NumKVHeads: 1, HeadDim: width}
	c := NewKVCacheWithPrecision(cfg, KVPrecisionQ8_0)
	replacement := NewKVCacheWithPrecision(cfg, KVPrecisionQ8_0)
	k, v, changed := make([]float32, P*width), make([]float32, P*width), make([]float32, P*width)
	for i := range v {
		v[i], changed[i] = 1, 3
	}
	c.appendBatchedKV(0, k, v, P, width)
	replacement.appendBatchedKV(0, k, changed, P, width)
	_, oldValues := c.attentionRows(0)
	_, newValues := replacement.attentionRows(0)
	original := c.vQ8[0]
	calls := 0
	score := func(q, key []float32) float32 {
		if calls == 0 {
			// Only the generic callback path permits this inter-unit mutation.
			c.vQ8[0] = replacement.vQ8[0]
		}
		calls++
		return fdot(q, key)
	}
	out, q := make([]float32, P*width), make([]float32, P*width)
	for _, gqa := range []bool{false, true} {
		c.vQ8[0], calls = original, 0
		clear(out)
		if gqa {
			attnPrefillMultiGQAInto(out, q, []*KVCache{c}, []int{0}, 0, P, 1, width, width, 1, -1, 1, score, nil, nil)
		} else {
			attnPrefillMultiInto(out, q, []*KVCache{c}, []int{0}, 0, P, 1, width, width, 1, -1, 1, score, nil)
		}
		for d := 0; d < width; d++ {
			want := float32(0.5 * newValues[d])
			want += 0.5 * newValues[width+d]
			if math.Float32bits(out[d]) != math.Float32bits(oldValues[d]) || math.Float32bits(out[width+d]) != math.Float32bits(want) {
				t.Fatal("generic callback was given builtin snapshot authority")
			}
		}
	}
}

func TestRectPrefillFreshRetainedPanelLoan(t *testing.T) {
	oldWorkers, oldSource := currentWorkerBudget()
	oldSIMD := attnFdot3SIMD
	t.Cleanup(func() {
		workerBudgetMu.Lock()
		numWorkers, workerBudgetSource = oldWorkers, oldSource
		workerBudgetMu.Unlock()
		attnFdot3SIMD = oldSIMD
	})
	attnFdot3SIMD = false
	const B, w, nH, hd = 3, 64, 6, 32
	cfg := Config{NumLayers: 1, HiddenSize: nH * hd, NumHeads: nH, NumKVHeads: 2, HeadDim: hd}
	for _, P := range []int{8, 193} { // The larger decode crosses parForRange's work threshold.
		caches := make([]*KVCache, B)
		base := make([]int, B)
		wantK, wantV := make([]float32, B*P*w), make([]float32, B*P*w)
		for b := range caches {
			c := NewKVCacheWithPrecision(cfg, KVPrecisionQ8_0)
			if b == B-1 {
				c = NewKVCache(cfg) // Mixed precision copies the f32 member too.
			}
			k, v := make([]float32, P*w), make([]float32, P*w)
			for i := range k {
				k[i], v[i] = float32((i+b)%23-11)/17, float32((i+3*b)%31-15)/19
			}
			c.appendBatchedKV(0, k, v, P, w)
			kk, vv := c.attentionRows(0)
			copy(wantK[b*P*w:], kk)
			copy(wantV[b*P*w:], vv)
			caches[b] = c
		}
		keyStorage, valueStorage := make([]float32, B*P*w+2), make([]float32, B*P*w+2)
		keys, values := keyStorage[1:len(keyStorage)-1], valueStorage[1:len(valueStorage)-1]
		keyStorage[0], keyStorage[len(keyStorage)-1] = 17, 19
		valueStorage[0], valueStorage[len(valueStorage)-1] = 23, 29
		for _, workers := range []int{1, 4} {
			if err := SetWorkers(workers); err != nil {
				t.Fatal(err)
			}
			for i := range keys {
				keys[i], values[i] = float32(math.NaN()), float32(math.NaN())
			}
			if !prepareRectPrefillKVLoan(keys, values, caches, base, 0, P, w) {
				t.Fatal("fresh retained panel declined")
			}
			for i := range keys {
				if math.Float32bits(keys[i]) != math.Float32bits(wantK[i]) || math.Float32bits(values[i]) != math.Float32bits(wantV[i]) {
					t.Fatalf("decoded loan bits differ at %d", i)
				}
			}
			if keyStorage[0] != 17 || keyStorage[len(keyStorage)-1] != 19 || valueStorage[0] != 23 || valueStorage[len(valueStorage)-1] != 29 {
				t.Fatal("loan boundary overwritten")
			}
			if P == 8 {
				q := make([]float32, B*P*nH*hd)
				for i := range q {
					q[i] = float32(i%11-5) / 7
				}
				got, want := make([]float32, len(q)), make([]float32, len(q))
				attnPrefillMultiGQAInto(want, q, caches, base, 0, P, nH, hd, w, 3, -1, 0.25, fdot, fdot3scalar, nil)
				attnPrefillMultiGQALoanedInto(got, q, keys, values, caches, base, 0, P, nH, hd, w, 3, -1, 0.25, nil)
				for i := range got {
					if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
						t.Fatalf("fresh loan attention bits differ at %d", i)
					}
				}
			}
		}
		decline := func(label string, keyDst []float32) {
			t.Helper()
			for i := range keys {
				keys[i], values[i] = 31, 37
			}
			if prepareRectPrefillKVLoan(keyDst, values, caches, base, 0, P, w) {
				t.Fatalf("accepted %s", label)
			}
			for i := range keys {
				if keys[i] != 31 || values[i] != 37 {
					t.Fatalf("partial loan write on %s", label)
				}
			}
		}
		decline("short loan with spare capacity", keys[:len(keys)-1])
		base[B-1] = 1
		decline("late prefix", keys)
		base[B-1] = 0
		scales := caches[1].vQ8[0].scales
		caches[1].vQ8[0].scales = nil
		decline("late malformed packed row", keys)
		caches[1].vQ8[0].scales = scales
		caches[0], caches[1] = caches[2], caches[2]
		decline("all-f32 no-copy route", keys)
	}
}
