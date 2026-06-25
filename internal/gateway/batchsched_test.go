package gateway

import (
	"math"
	"testing"
)

func TestDynamicBatchSizeClampsToPendingAndMax(t *testing.T) {
	p := BatchPolicy{MaxBatch: 8}
	for _, tt := range []struct {
		name    string
		pending int
		want    int
	}{
		{name: "empty queue still returns admissible minimum", pending: 0, want: 1},
		{name: "shallow queue uses pending", pending: 3, want: 3},
		{name: "deep queue clamps to max", pending: 20, want: 8},
	} {
		t.Run(tt.name, func(t *testing.T) {
			if got := p.DynamicBatchSize(tt.pending); got != tt.want {
				t.Fatalf("DynamicBatchSize(%d) = %d, want %d", tt.pending, got, tt.want)
			}
		})
	}

	if got := (BatchPolicy{MaxBatch: 0}).DynamicBatchSize(10); got != 1 {
		t.Fatalf("zero MaxBatch clamp = %d, want 1", got)
	}
}

func TestComposeBatchesCoversEveryRequestOnce(t *testing.T) {
	lengths := []int{32, 33, 65, 66, 67, 512, 513, 20, 21, 0, -4}
	p := DefaultBatchPolicy()

	batches := p.ComposeBatches(lengths)
	seen := make([]int, len(lengths))
	for _, b := range batches {
		if len(b) == 0 {
			t.Fatalf("empty batch emitted: %#v", batches)
		}
		for _, idx := range b {
			if idx < 0 || idx >= len(lengths) {
				t.Fatalf("batch index %d outside [0,%d): %#v", idx, len(lengths), batches)
			}
			seen[idx]++
		}
	}
	for i, n := range seen {
		if n != 1 {
			t.Fatalf("request %d scheduled %d times; batches=%#v", i, n, batches)
		}
	}
}

func TestBatchPaddingOverheadInvariant(t *testing.T) {
	lengths := []int{
		15, 16, 17, 18, 19,
		31, 32, 33, 34, 35,
		63, 64, 65, 66, 67,
		127, 128, 129, 130, 131,
		255, 256, 257, 258, 259,
		511, 512,
	}
	p := DefaultBatchPolicy()

	batches, stats := p.PlanBatches(lengths)
	if len(batches) == 0 {
		t.Fatal("expected at least one batch")
	}
	if stats.NumRequests != len(lengths) {
		t.Fatalf("NumRequests = %d, want %d", stats.NumRequests, len(lengths))
	}
	if stats.NumBatched == 0 {
		t.Fatalf("expected some requests to be batched; batches=%#v", batches)
	}
	if stats.WorstPadOverhead > p.MaxPadOverhead+1e-12 {
		t.Fatalf("worst padding overhead %.6f exceeds %.6f; batches=%#v", stats.WorstPadOverhead, p.MaxPadOverhead, batches)
	}
	if stats.AggregatePadOverhead > p.MaxPadOverhead+1e-12 {
		t.Fatalf("aggregate padding overhead %.6f exceeds %.6f; batches=%#v", stats.AggregatePadOverhead, p.MaxPadOverhead, batches)
	}
}

func TestComposeBatchesSplitsWhenPaddingWouldExceedPolicy(t *testing.T) {
	p := BatchPolicy{MaxBatch: 8, MaxPadOverhead: 0.10, MaxPromptLen: rectPrefillTokenCeiling}
	lengths := []int{100, 100, 100, 200}

	batches := p.ComposeBatches(lengths)
	if len(batches) != 2 {
		t.Fatalf("len(batches) = %d, want 2: %#v", len(batches), batches)
	}
	if got := PaddingOverhead([]int{100, 100, 100, 200}); got <= p.MaxPadOverhead {
		t.Fatalf("test fixture no longer exceeds padding policy: %.6f <= %.6f", got, p.MaxPadOverhead)
	}
	for _, b := range batches {
		ls := make([]int, len(b))
		for i, idx := range b {
			ls[i] = lengths[idx]
		}
		if got := PaddingOverhead(ls); got > p.MaxPadOverhead+1e-12 {
			t.Fatalf("batch %#v overhead %.6f exceeds %.6f", b, got, p.MaxPadOverhead)
		}
	}
}

func TestComposeBatchesSerializesInvalidAndOversizedPrompts(t *testing.T) {
	p := BatchPolicy{MaxBatch: 8, MaxPadOverhead: 0.10, MaxPromptLen: 64}
	lengths := []int{0, -1, 65, 32, 33}

	batches := p.ComposeBatches(lengths)
	for _, idx := range []int{0, 1, 2} {
		if !hasSingleton(batches, idx) {
			t.Fatalf("request %d should be a serial singleton; batches=%#v", idx, batches)
		}
	}
	if stats := StatsFor(lengths, batches); math.Abs(stats.BatchedFraction()-0.4) > 1e-12 {
		t.Fatalf("BatchedFraction = %.6f, want 0.4; stats=%#v batches=%#v", stats.BatchedFraction(), stats, batches)
	}
}

func hasSingleton(batches [][]int, idx int) bool {
	for _, b := range batches {
		if len(b) == 1 && b[0] == idx {
			return true
		}
	}
	return false
}
