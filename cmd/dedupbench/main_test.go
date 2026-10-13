package main

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/radixkv"
)

// TestDedupBenchFusionCoalescesAndMatchesFullPrefill pins the two properties the
// benchmark's headline depends on: the fusion arm REALLY coalesces (exactly one
// leader, N-1 followers) and every arm's decode logits are bit-identical to a
// fresh full prefill. Without the first, the ratio would be measuring N independent
// leaders, not fusion; without the second, a fast-but-wrong arm could post a win.
//
// fak-test:runtime medium est=6000ms
func TestDedupBenchFusionCoalescesAndMatchesFullPrefill(t *testing.T) {
	for _, procs := range []int{1, 2} {
		t.Run(fmt.Sprintf("procs_%d", procs), func(t *testing.T) {
			// A simultaneous admission is a registration contract, not a promise that
			// runnable goroutines execute before a short prefill on another scheduler P.
			previous := runtime.GOMAXPROCS(procs)
			t.Cleanup(func() { runtime.GOMAXPROCS(previous) })
			const n, prefix, suffix = 6, 96, 24
			m := model.NewSynthetic(model.Config{
				HiddenSize:        128,
				NumLayers:         4,
				NumHeads:          8,
				NumKVHeads:        4,
				HeadDim:           16,
				IntermediateSize:  256,
				VocabSize:         512,
				RMSNormEps:        1e-5,
				RopeTheta:         10000,
				TieWordEmbeddings: true,
				EOSTokenID:        -1,
			})
			vocab := 512
			shared := lcg(prefix, vocab, 7)
			prompts := make([][]int, n)
			for i := range prompts {
				prompts[i] = append(append([]int(nil), shared...), lcg(suffix, vocab, 100+uint64(i))...)
			}
			oracle := make([][]float32, n)
			for i, p := range prompts {
				s := m.NewSession()
				oracle[i] = s.Prefill(p)
			}

			fusion, ok := armFusion(m, prompts, oracle)
			if !ok {
				t.Fatalf("fusion arm failed its correctness/integrity gate: %+v", fusion)
			}
			if fusion.CoalescedLeaders != 1 || fusion.CoalescedFollowers != int64(n-1) {
				t.Fatalf("fusion coalescing = leaders %d followers %d, want 1/%d",
					fusion.CoalescedLeaders, fusion.CoalescedFollowers, n-1)
			}
			if !fusion.Correctness {
				t.Fatal("fusion arm not correctness-gated")
			}

			nba, ok := armWarmCacheNBA(m, prompts, oracle)
			if !ok || !nba.Correctness {
				t.Fatal("warm-cache NBA arm failed its correctness gate")
			}
			if nba.CoalescedLeaders != 0 || nba.CoalescedFollowers != 0 {
				t.Fatalf("warm-cache NBA reported coalescing counters %d/%d, want 0/0",
					nba.CoalescedLeaders, nba.CoalescedFollowers)
			}

			naive, ok := armNaive(m, prompts, oracle)
			if !ok || !naive.Correctness {
				t.Fatal("naive arm failed its correctness gate")
			}

			// Sanity on the token accounting: fusion and warm-cache do the SAME total prefill
			// tokens (one shared prefix + N suffixes), which is why the ratio is a wall-clock,
			// not a token-count, comparison.
			fusionWant := prefix + n*suffix
			if fusion.PrefillTok != fusionWant {
				t.Fatalf("fusion prefill tokens = %d, want %d", fusion.PrefillTok, fusionWant)
			}
			if nba.PrefillTok != fusion.PrefillTok {
				t.Fatalf("warm-cache prefill tokens = %d, want fusion total %d", nba.PrefillTok, fusion.PrefillTok)
			}
			if naive.PrefillTok != n*(prefix+suffix) {
				t.Fatalf("naive prefill tokens = %d, want %d", naive.PrefillTok, n*(prefix+suffix))
			}
		})
	}
}

// TestDedupBenchBitEqualRejectsDivergence makes the correctness gate non-vacuous.
func TestDedupBenchBitEqualRejectsDivergence(t *testing.T) {
	a := []float32{1, 2, 3}
	if bitEqual(a, []float32{1, 2, 3}) != true {
		t.Fatal("bitEqual rejected identical slices")
	}
	if bitEqual(a, []float32{1, 2, 4}) {
		t.Fatal("bitEqual accepted a divergent slice")
	}
	if bitEqual(a, []float32{1, 2}) {
		t.Fatal("bitEqual accepted a length mismatch")
	}
}

// A missing admission must fail within the caller's deadline, never hang a receipt.
func TestDedupBenchFusionAdmissionDeadline(t *testing.T) {
	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()
	if err := waitForFusionFollowers(ctx, radixkv.NewPrefixFlightGroup(nil), 1); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expired admission deadline = %v, want context.DeadlineExceeded", err)
	}
}
