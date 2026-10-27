package model

import (
	"math"
	"testing"
)

func TestPrefillGQA3LoanMatchesIndependentHeads(t *testing.T) {
	oldWorkers, oldSource := currentWorkerBudget()
	t.Cleanup(func() {
		workerBudgetMu.Lock()
		numWorkers, workerBudgetSource = oldWorkers, oldSource
		workerBudgetMu.Unlock()
	})
	for _, workers := range []int{1, 3, 4} {
		if err := SetWorkers(workers); err != nil {
			t.Fatal(err)
		}
		for _, grp := range []int{3, 4, 5, 6, 8} {
			for _, hd := range []int{7, 16} {
				for _, tc := range []struct {
					base, window int
					cap          float32
				}{{0, -1, 0}, {7, -1, 1.5}, {7, 4, 0}, {7, 1, 1.5}} {
					const P, nKV = 5, 2
					nH, w := nKV*grp, nKV*hd
					q := make([]float32, P*nH*hd)
					k := make([]float32, (tc.base+P)*w)
					v := make([]float32, len(k))
					for i := range q {
						q[i] = float32(i%23-11) / 19
					}
					for i := range k {
						k[i] = float32(i%17-8) / 13
						v[i] = float32(i%29-14) / 7
					}
					q[0], k[0], v[0] = float32(math.Copysign(0, -1)), 0, float32(math.Copysign(0, -1))
					qBefore := append([]float32(nil), q...)
					kBefore := append([]float32(nil), k...)
					vBefore := append([]float32(nil), v...)
					got, want := make([]float32, len(q)), make([]float32, len(q))
					for i := range got {
						got[i] = float32(i%5-2) / 11 // Preserve += semantics too.
					}
					copy(want, got)
					attnPrefillInto(want, q, k, v, P, tc.base, nH, hd, w, grp, tc.window, 0, 0.25, tc.cap, fdot, nil)
					maxPos := tc.base + P
					width := maxPos - windowLoContig(maxPos, maxPos-1, tc.window)
					need := 3 * workers * width
					loan := make([]float32, need+1)
					for i := range loan {
						loan[i] = 12345 // Every live score must be overwritten.
					}
					if !attnPrefillGQA3FDotInto(got, q, k, v, loan[:need], P, tc.base, nH, hd, w, grp, tc.window, 0.25, tc.cap, nil) {
						t.Fatalf("eligible loan declined: workers=%d grp=%d hd=%d case=%+v", workers, grp, hd, tc)
					}
					if loan[need] != 12345 {
						t.Fatal("wrote beyond loan length")
					}
					for _, pair := range [][2][]float32{{got, want}, {q, qBefore}, {k, kBefore}, {v, vBefore}} {
						for i := range pair[0] {
							if math.Float32bits(pair[0][i]) != math.Float32bits(pair[1][i]) {
								t.Fatalf("bits differ: workers=%d grp=%d hd=%d case=%+v index=%d", workers, grp, hd, tc, i)
							}
						}
					}
				}
			}
		}
	}
}

func TestPrefillGQA3LoanDeclinesWithoutWrites(t *testing.T) {
	oldWorkers, oldSource := currentWorkerBudget()
	t.Cleanup(func() {
		workerBudgetMu.Lock()
		numWorkers, workerBudgetSource = oldWorkers, oldSource
		workerBudgetMu.Unlock()
	})
	if err := SetWorkers(4); err != nil {
		t.Fatal(err)
	}
	const P, nH, hd, w, grp = 8, 6, 8, 16, 3
	q, k, v := make([]float32, P*nH*hd), make([]float32, P*w), make([]float32, P*w)
	out := make([]float32, len(q))
	loan := make([]float32, 3*4*P)
	for i := range out {
		out[i] = 17
	}
	for i := range loan {
		loan[i] = 19
	}
	observerCalls := 0
	obs := func(int, int, int, []int, []float32) { observerCalls++ }
	if attnPrefillGQA3FDotInto(out, q, k, v, loan, P, 0, nH, hd, w, grp, -1, 0.25, 0, obs) {
		t.Fatal("accepted an observer")
	}
	if observerCalls != 0 {
		t.Fatal("declined path emitted an observation")
	}
	maxInt := int(^uint(0) >> 1)
	for _, tc := range []struct {
		name                string
		rows, base, heads   int
		group, window, size int
	}{
		{"short length with spare capacity", P, 0, nH, grp, -1, len(loan) - 1},
		{"insufficient grouped units", 1, 0, nH, grp, -1, len(loan)},
		{"small GQA group", P, 0, nH, 2, -1, len(loan)},
		{"zero window", P, 0, nH, grp, 0, len(loan)},
		{"position overflow", P, maxInt, nH, grp, -1, len(loan)},
		{"unit overflow", maxInt, 0, nH, grp, -1, len(loan)},
		{"head width overflow", 1, 0, maxInt - maxInt%3, grp, -1, len(loan)},
	} {
		if attnPrefillGQA3FDotInto(out, q, k, v, loan[:tc.size], tc.rows, tc.base, tc.heads, hd, w, tc.group, tc.window, 0.25, 0, nil) {
			t.Fatalf("accepted %s", tc.name)
		}
		for _, pair := range []struct {
			values []float32
			want   float32
		}{{out, 17}, {loan, 19}} {
			for _, x := range pair.values {
				if x != pair.want {
					t.Fatalf("decline mutated storage: %s", tc.name)
				}
			}
		}
	}
}

func TestQwen35Q4KGQALoanRequiresBuiltinOwnership(t *testing.T) {
	oldWorkers, oldSource := currentWorkerBudget()
	t.Cleanup(func() {
		workerBudgetMu.Lock()
		numWorkers, workerBudgetSource = oldWorkers, oldSource
		workerBudgetMu.Unlock()
	})
	if err := SetWorkers(4); err != nil {
		t.Fatal(err)
	}
	const P, hd, nH, nKV = 8, 8, 6, 2
	const H, w = nH * hd, nKV * hd
	cfg := Config{HiddenSize: H, HeadDim: hd, NumHeads: nH, NumKVHeads: nKV, NumLayers: 1, RopeTheta: 10000}
	run := func(owned, metal, observed bool) (out, before, after []float32) {
		m := &Model{Cfg: cfg}
		if observed {
			m.attnObs = func(int, int, int, []int, []float32) {}
		}
		s := &Session{M: m, MetalQ4K: metal, Cache: &KVCache{K: make([][]float32, 1), Kraw: make([][]float32, 1), V: make([][]float32, 1)}}
		qf, k, v := make([]float32, P*2*H), make([]float32, P*w), make([]float32, P*w)
		for i := range qf {
			qf[i] = float32(i%31-15) / 19
		}
		for i := range k {
			k[i], v[i] = float32(i%17-8)/13, float32(i%29-14)/7
		}
		before = append([]float32(nil), qf...)
		group := func([]string, []float32, *q8Panel) [][]float32 { return [][]float32{qf, k, v} }
		project := func(_ string, x []float32, _ *q8Panel) []float32 { return append([]float32(nil), x...) }
		quantize := func([]float32, int, int, ...string) *q8Panel { return nil }
		if owned {
			// This fixture explicitly grants its fresh, separate result allocations.
			// The ordinary generic callback entry must never infer that authority.
			out = s.prefillQwen35FullAttnLayerQ4KOwnedQF(0, make([]float32, P*H), P, 0, project, group, quantize, true)
		} else {
			out = s.prefillQwen35FullAttnLayerQ4K(0, make([]float32, P*H), P, 0, project, group, quantize)
		}
		return out, before, qf
	}
	want, before, retained := run(false, false, false)
	for i := range before {
		if math.Float32bits(before[i]) != math.Float32bits(retained[i]) {
			t.Fatal("generic projection callback result was borrowed")
		}
	}
	got, before, loan := run(true, false, false)
	changed := false
	for i := range before {
		changed = changed || math.Float32bits(before[i]) != math.Float32bits(loan[i])
	}
	if !changed {
		t.Fatal("eligible builtin fixture never consumed its dead qf loan")
	}
	for i := range got {
		if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
			t.Fatalf("owned GQA loan changed output bits at %d", i)
		}
	}
	for _, metal := range []bool{false, true} {
		// An observer also forces the host reference when Metal is configured,
		// so this ownership witness never needs a GPU operation.
		got, before, retained = run(true, metal, true)
		for i := range before {
			if math.Float32bits(before[i]) != math.Float32bits(retained[i]) {
				t.Fatal("observed/Metal route borrowed qf")
			}
		}
		for i := range got {
			if math.Float32bits(got[i]) != math.Float32bits(want[i]) {
				t.Fatalf("declined route changed output bits at %d", i)
			}
		}
	}
}
