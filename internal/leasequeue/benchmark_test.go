package leasequeue

import (
	"fmt"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchaging"
)

// BenchmarkTicketID measures stable ticket digest generation from actor, lane, and tree globs.
func BenchmarkTicketID(b *testing.B) {
	globs := []string{
		"internal/gateway/**",
		"internal/model/**",
		"docs/gateway/**",
		"cmd/fak/**",
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = TicketID("session:worker-42", "gateway", globs)
	}
}

// BenchmarkPrioritySorting measures priority sorting and anti-starvation candidate ordering.
func BenchmarkPrioritySorting(b *testing.B) {
	const now = nowFixture
	tax := testTax()
	holders := []Holder{{Lease: gatewayHolder("h1")}}
	tickets := []Ticket{
		operator("op-fresh", "session:op1", "gateway", now-5),
		operator("op-older", "session:op2", "gateway", now-60),
		waiter("loop-fresh", "loop:a", "gateway", now-120),
		waiter("loop-aging", "loop:b", "gateway", now-1800),
		waiter("loop-starved", "loop:c", "gateway", now-7*3600),
	}
	params := Params{
		NowUnix: now,
		Aging:   dispatchaging.DefaultParams(now),
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res := Plan(tickets, holders, tax, params)
		if res.Depth != 5 {
			b.Fatalf("depth = %d, want 5", res.Depth)
		}
	}
}

// BenchmarkPlan measures pure planning performance across different contention and queue topologies.
func BenchmarkPlan(b *testing.B) {
	const now = nowFixture
	tax := testTax()
	holders := []Holder{
		{Lease: gatewayHolder("h1"), ExpiresUnix: now + 300},
	}

	b.Run("ContendedArrivalOrder", func(b *testing.B) {
		tickets := make([]Ticket, 10)
		for i := range tickets {
			tickets[i] = waiter(fmt.Sprintf("w%02d", i), fmt.Sprintf("session:%d", i), "gateway", now-int64((10-i)*60))
		}
		params := Params{NowUnix: now}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			res := Plan(tickets, holders, tax, params)
			if res.Depth != 10 {
				b.Fatalf("depth = %d, want 10", res.Depth)
			}
		}
	})

	b.Run("ConservativeBackfill", func(b *testing.B) {
		tickets := make([]Ticket, 20)
		for i := range tickets {
			lane := "gateway"
			if i%2 == 0 {
				lane = "model"
			}
			tickets[i] = waiter(fmt.Sprintf("w%02d", i), fmt.Sprintf("session:%d", i), lane, now-int64((20-i)*60))
		}
		params := Params{NowUnix: now}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			res := Plan(tickets, holders, tax, params)
			if len(res.Granted) != 1 {
				b.Fatalf("granted = %d, want 1", len(res.Granted))
			}
		}
	})

	b.Run("DeepContention", func(b *testing.B) {
		tickets := make([]Ticket, 50)
		for i := range tickets {
			tickets[i] = waiter(fmt.Sprintf("w%02d", i), fmt.Sprintf("session:%d", i), "gateway", now-int64((50-i)*30))
		}
		params := Params{NowUnix: now}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			res := Plan(tickets, holders, tax, params)
			if res.Depth != 50 {
				b.Fatalf("depth = %d, want 50", res.Depth)
			}
		}
	})
}

// BenchmarkQueueEnqueue measures ticket minting and refresh in the durable journal.
func BenchmarkQueueEnqueue(b *testing.B) {
	b.Run("NewTicket", func(b *testing.B) {
		dir := b.TempDir()
		s := NewStore(dir)
		now := time.Unix(1_000_000, 0)
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			tk := Ticket{ID: fmt.Sprintf("tkt%08d", i), Actor: "session:worker", Lane: "gateway"}
			if _, err := s.Mint(tk, now); err != nil {
				b.Fatalf("mint: %v", err)
			}
		}
	})

	b.Run("RefreshExisting", func(b *testing.B) {
		dir := b.TempDir()
		s := NewStore(dir)
		now := time.Unix(1_000_000, 0)
		tk := Ticket{Actor: "session:worker", Lane: "gateway"}
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			now = now.Add(time.Second)
			if _, err := s.Mint(tk, now); err != nil {
				b.Fatalf("mint: %v", err)
			}
		}
	})
}

// BenchmarkQueuePeek measures listing live tickets from the journal and finding in planned results.
func BenchmarkQueuePeek(b *testing.B) {
	b.Run("StoreLive", func(b *testing.B) {
		dir := b.TempDir()
		s := NewStore(dir)
		now := time.Unix(1_000_000, 0)
		const ticketCount = 20
		for i := 0; i < ticketCount; i++ {
			tk := Ticket{
				ID:    fmt.Sprintf("live%04d", i),
				Actor: fmt.Sprintf("session:%d", i),
				Lane:  "gateway",
				Class: ClassLoop,
			}
			if _, err := s.Mint(tk, now); err != nil {
				b.Fatalf("setup mint: %v", err)
			}
		}

		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			live, err := s.Live(now)
			if err != nil {
				b.Fatalf("live: %v", err)
			}
			if len(live) != ticketCount {
				b.Fatalf("live count = %d, want %d", len(live), ticketCount)
			}
		}
	})

	b.Run("ResultFind", func(b *testing.B) {
		const now = nowFixture
		tax := testTax()
		holders := []Holder{{Lease: gatewayHolder("h1")}}
		tickets := make([]Ticket, 50)
		for i := range tickets {
			tickets[i] = waiter(fmt.Sprintf("w%02d", i), fmt.Sprintf("session:%d", i), "gateway", now-int64((50-i)*30))
		}
		res := Plan(tickets, holders, tax, Params{NowUnix: now})
		targetID := "w25"

		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			entry, ok := res.Find(targetID)
			if !ok || entry.ID != targetID {
				b.Fatalf("failed to find %s", targetID)
			}
		}
	})
}

// BenchmarkQueueDequeue measures dropping tickets from the store upon lease acquisition.
func BenchmarkQueueDequeue(b *testing.B) {
	b.Run("DropExisting", func(b *testing.B) {
		dir := b.TempDir()
		s := NewStore(dir)
		now := time.Unix(1_000_000, 0)
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			id := fmt.Sprintf("drop%08d", i)
			tk := Ticket{ID: id, Actor: "session:worker", Lane: "gateway"}
			if _, err := s.Mint(tk, now); err != nil {
				b.Fatalf("mint: %v", err)
			}
			if err := s.Drop(id); err != nil {
				b.Fatalf("drop: %v", err)
			}
		}
	})

	b.Run("DropMissing", func(b *testing.B) {
		dir := b.TempDir()
		s := NewStore(dir)
		b.ResetTimer()
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			id := fmt.Sprintf("missing%08d", i)
			if err := s.Drop(id); err != nil {
				b.Fatalf("drop: %v", err)
			}
		}
	})
}

// TestBenchmarksSanity verifies that benchmarks execute cleanly without panic.
func TestBenchmarksSanity(t *testing.T) {
	res := testing.Benchmark(BenchmarkTicketID)
	if res.N <= 0 {
		t.Fatalf("BenchmarkTicketID failed to execute: %+v", res)
	}
	resPriority := testing.Benchmark(BenchmarkPrioritySorting)
	if resPriority.N <= 0 {
		t.Fatalf("BenchmarkPrioritySorting failed to execute: %+v", resPriority)
	}
}
