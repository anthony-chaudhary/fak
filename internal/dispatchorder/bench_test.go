package dispatchorder

import (
	"fmt"
	"testing"
)

// BenchmarkDispatchOrder exercises dispatch ordering, supersede collapse,
// priority weighting, and tiebreaking in a loop.
func BenchmarkDispatchOrder(b *testing.B) {
	candidates := make([]Candidate, 0, 50)
	// Add multiple groups with duplicate keys to exercise supersede collapse.
	for i := 0; i < 20; i++ {
		candidates = append(candidates, Candidate{
			ID:          fmt.Sprintf("dup-%02d", i),
			Key:         fmt.Sprintf("key-%d", i%5),
			Priority:    (i % 3) * 100,
			CreatedUnix: int64(1_000_000 - 1000 + i*10),
			UpdatedUnix: int64(1_000_000 - 500 + i*20),
			Lane:        fmt.Sprintf("lane-%d", i%3),
			Tree:        []string{fmt.Sprintf("internal/pkg%d/**", i%4)},
		})
	}
	// Add candidates with distinct keys, attempted timestamps, and dependencies.
	for i := 20; i < 40; i++ {
		candidates = append(candidates, Candidate{
			ID:              fmt.Sprintf("cand-%02d", i),
			Key:             fmt.Sprintf("distinct-%02d", i),
			Priority:        (i % 4) * 100,
			CreatedUnix:     int64(1_000_000 - 2000 + i*5),
			UpdatedUnix:     int64(1_000_000 - 100 + i),
			LastAttemptUnix: int64(1_000_000 - 600 + i),
			Lane:            fmt.Sprintf("lane-%d", i%4),
			Tree:            []string{fmt.Sprintf("internal/leaf%d/**", i%3)},
		})
	}
	// Add collision-forcing candidates to exercise collision pricing.
	candidates = append(candidates,
		Candidate{
			ID:          "overlap-1",
			Key:         "overlap-key-1",
			Lane:        "shared-lane",
			Tree:        []string{"internal/overlap/**"},
			UpdatedUnix: 999_900,
		},
		Candidate{
			ID:          "overlap-2",
			Key:         "overlap-key-2",
			Lane:        "shared-lane",
			Tree:        []string{"internal/overlap/sub/**"},
			UpdatedUnix: 999_800,
		},
	)

	in := Input{
		Candidates:      candidates,
		NowUnix:         1_000_000,
		CooldownSeconds: 300,
		FinishFirst:     true,
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res := Plan(in)
		if len(res.Order) == 0 {
			b.Fatal("unexpected empty order result")
		}
	}
}

// TestBenchmarkDispatchOrderSanity verifies that the benchmark candidate input plans properly.
func TestBenchmarkDispatchOrderSanity(t *testing.T) {
	candidates := []Candidate{
		{ID: "c1", Key: "k1", UpdatedUnix: 100},
		{ID: "c2", Key: "k1", UpdatedUnix: 200},
		{ID: "c3", Key: "k2", UpdatedUnix: 150},
	}
	res := Plan(Input{Candidates: candidates, NowUnix: 1000})
	if res.KeepCount != 2 || res.SupersededCount != 1 {
		t.Fatalf("unexpected result: keep=%d, superseded=%d", res.KeepCount, res.SupersededCount)
	}
}
