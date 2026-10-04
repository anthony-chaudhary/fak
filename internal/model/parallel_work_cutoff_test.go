package model

import (
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
)

// TestParForWorkCoversEveryIndexOnce pins parForWork's contract on both sides of
// parElemThreshold: every index in [0,n) is visited exactly once, and below the cutoff the body
// runs as ONE inline call body(0,n) (no pool dispatch), which is what makes the swap from parFor
// bit-identical for the per-index-independent prefill fan-outs (#13694).
func TestParForWorkCoversEveryIndexOnce(t *testing.T) {
	old := NumWorkers()
	defer SetWorkers(old)
	if err := SetWorkers(4); err != nil {
		t.Fatal(err)
	}
	for _, n := range []int{0, 1, 2, 17, 1000} {
		for _, side := range []struct {
			name  string
			elems int
		}{{"below", 1}, {"above", parElemThreshold}} {
			t.Run(fmt.Sprintf("n=%d/%s", n, side.name), func(t *testing.T) {
				hits := make([]atomic.Int32, n)
				var mu sync.Mutex
				var calls [][2]int
				parForWork(n, NumWorkers(), side.elems, func(lo, hi int) {
					mu.Lock()
					calls = append(calls, [2]int{lo, hi})
					mu.Unlock()
					for i := lo; i < hi; i++ {
						hits[i].Add(1)
					}
				})
				for i := range hits {
					if c := hits[i].Load(); c != 1 {
						t.Fatalf("index %d visited %d times, want 1", i, c)
					}
				}
				serial := n*side.elems < parElemThreshold
				if serial && (len(calls) != 1 || calls[0] != [2]int{0, n}) {
					t.Fatalf("below threshold: calls = %v, want exactly [[0 %d]] (inline)", calls, n)
				}
				if !serial && n >= 2 && len(calls) < 2 {
					t.Fatalf("above threshold: calls = %v, want the parFor chunking (>=2 chunks)", calls)
				}
			})
		}
	}
}

// BenchmarkPrefillElementwiseSerialVsParFor measures the per-layer elementwise fan-outs of a
// short-prompt prefill (RMSNorm over P rows of H, and a residual add over P*I) run serially vs
// through parFor, at Qwen2.5-7B widths (H=3584, I=18944). It is the evidence for
// parElemThreshold: below the cutoff the fork/join barrier costs more than the work it splits.
func BenchmarkPrefillElementwiseSerialVsParFor(b *testing.B) {
	const H, I = 3584, 18944
	workers := currentWorkerCount()
	for _, P := range []int{4, 17, 64, 256} {
		X := make([]float32, P*I)
		Y := make([]float32, P*I)
		wN := make([]float32, H)
		for i := range X {
			X[i] = float32(i%97) * 0.01
			Y[i] = float32(i%89) * 0.02
		}
		for i := range wN {
			wN[i] = 1
		}
		dst := make([]float32, P*H)
		norm := func(lo, hi int) {
			for t := lo; t < hi; t++ {
				rmsnormInto(dst[t*H:(t+1)*H], X[t*H:(t+1)*H], wN, 1e-6)
			}
		}
		add := func(lo, hi int) {
			for i := lo; i < hi; i++ {
				X[i] += Y[i]
			}
		}
		b.Run(fmt.Sprintf("rmsnorm/P=%d/elems=%d/serial", P, P*H), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				norm(0, P)
			}
		})
		b.Run(fmt.Sprintf("rmsnorm/P=%d/elems=%d/parFor", P, P*H), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				parFor(P, workers, norm)
			}
		})
		b.Run(fmt.Sprintf("add/P=%d/elems=%d/serial", P, P*I), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				add(0, P*I)
			}
		})
		b.Run(fmt.Sprintf("add/P=%d/elems=%d/parFor", P, P*I), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				parFor(P*I, workers, add)
			}
		})
	}
}
