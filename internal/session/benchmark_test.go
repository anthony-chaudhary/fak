package session

import (
	"fmt"
	"sync/atomic"
	"testing"
	"time"
)

// BenchmarkTableDecide measures the per-turn boundary admission gate across unbounded,
// bounded-turn, and multi-session parallel contention scenarios.
func BenchmarkTableDecide(b *testing.B) {
	b.Run("Unbounded", func(b *testing.B) {
		tbl := NewTable()
		trace := "trace-unbounded"
		tbl.SetPriority(trace, 1)

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			v := tbl.Decide(trace)
			if !v.Proceed {
				b.Fatal("unexpected non-proceed verdict")
			}
		}
	})

	b.Run("BoundedTurns", func(b *testing.B) {
		tbl := NewTable()
		trace := "trace-bounded"
		tbl.SetBudget(trace, Budget{TurnsLeft: b.N + 10, TokensLeft: Unbounded})

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			v := tbl.Decide(trace)
			if !v.Proceed {
				b.Fatal("unexpected non-proceed verdict")
			}
		}
	})

	b.Run("Parallel", func(b *testing.B) {
		tbl := NewTable()
		const numTraces = 64
		traces := make([]string, numTraces)
		for i := 0; i < numTraces; i++ {
			traces[i] = fmt.Sprintf("trace-par-%d", i)
			tbl.SetPriority(traces[i], i%10)
		}

		b.ReportAllocs()
		b.ResetTimer()
		var counter uint64
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				idx := atomic.AddUint64(&counter, 1) % numTraces
				v := tbl.Decide(traces[idx])
				if !v.Proceed {
					b.Fatal("unexpected non-proceed verdict")
				}
			}
		})
	})
}

// BenchmarkTableDebitUsage measures post-turn token and cost accounting under unbounded,
// full multi-axis (tokens, context, spend, duration), and parallel workloads.
func BenchmarkTableDebitUsage(b *testing.B) {
	b.Run("Unbounded", func(b *testing.B) {
		tbl := NewTable()
		trace := "trace-usage-unbounded"
		tbl.SetPriority(trace, 1)
		u := Usage{
			OutputTokens:  120,
			ContextTokens: 2048,
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			st := tbl.DebitUsage(trace, u)
			if st.Run != Running {
				b.Fatal("unexpected non-running state")
			}
		}
	})

	b.Run("ContextAndSpend", func(b *testing.B) {
		tbl := NewTable()
		trace := "trace-usage-budgeted"
		tbl.SetBudget(trace, Budget{
			TokensLeft:          (b.N + 10) * 100,
			ContextTokensLeft:   (b.N + 10) * 2000,
			SpendMicroCentsLeft: int64(b.N+10) * 500,
		})
		u := Usage{
			OutputTokens:   50,
			ContextTokens:  1000,
			CostMicroCents: 250,
			DurationNanos:  int64(10 * time.Millisecond),
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			st := tbl.DebitUsage(trace, u)
			if st.Run != Running {
				b.Fatal("unexpected non-running state")
			}
		}
	})

	b.Run("Parallel", func(b *testing.B) {
		tbl := NewTable()
		const numTraces = 64
		traces := make([]string, numTraces)
		for i := 0; i < numTraces; i++ {
			traces[i] = fmt.Sprintf("trace-debit-par-%d", i)
			tbl.SetPriority(traces[i], 1)
		}
		u := Usage{OutputTokens: 50, ContextTokens: 500}

		b.ReportAllocs()
		b.ResetTimer()
		var counter uint64
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				idx := atomic.AddUint64(&counter, 1) % numTraces
				st := tbl.DebitUsage(traces[idx], u)
				if st.Run != Running {
					b.Fatal("unexpected non-running state")
				}
			}
		})
	})
}

// BenchmarkTableDebitToolCall measures the dispatched tool-call runaway floor gate.
func BenchmarkTableDebitToolCall(b *testing.B) {
	tbl := NewTable()
	trace := "trace-toolcall"
	tbl.SetBudget(trace, Budget{ToolCallsLeft: b.N + 10})

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v := tbl.DebitToolCall(trace)
		if !v.Proceed {
			b.Fatal("unexpected tool call refusal")
		}
	}
}

// BenchmarkTableGet measures pure RLock reads of session drive records.
func BenchmarkTableGet(b *testing.B) {
	tbl := NewTable()
	const numTraces = 256
	traces := make([]string, numTraces)
	for i := 0; i < numTraces; i++ {
		traces[i] = fmt.Sprintf("trace-get-%d", i)
		tbl.SetPriority(traces[i], i%10)
	}

	b.Run("Sequential", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			st := tbl.Get(traces[i%numTraces])
			if st.TraceID != traces[i%numTraces] {
				b.Fatalf("Get returned wrong trace: got %q, want %q", st.TraceID, traces[i%numTraces])
			}
		}
	})

	b.Run("Parallel", func(b *testing.B) {
		var counter uint64
		b.ReportAllocs()
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				idx := atomic.AddUint64(&counter, 1) % numTraces
				st := tbl.Get(traces[idx])
				if st.TraceID != traces[idx] {
					b.Fatalf("Get returned wrong trace: got %q, want %q", st.TraceID, traces[idx])
				}
			}
		})
	})
}

// BenchmarkTableSnapshot measures scheduler-consumption snapshot generation and sorting
// across 10, 100, and 1000 resident sessions.
func BenchmarkTableSnapshot(b *testing.B) {
	for _, count := range []int{10, 100, 1000} {
		b.Run(fmt.Sprintf("%d_sessions", count), func(b *testing.B) {
			tbl := NewTableWithLimit(count * 2)
			for i := 0; i < count; i++ {
				trace := fmt.Sprintf("trace-%04d", i)
				tbl.SetPriority(trace, i%10)
				tbl.SetPace(trace, Pace{MaxTokensPerTurn: 500, MinTurnGapMs: 10})
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				snap := tbl.Snapshot()
				if len(snap) != count {
					b.Fatalf("got snapshot len %d, want %d", len(snap), count)
				}
			}
		})
	}
}

// BenchmarkTableTransition measures state machine transition throughput under alternating states.
func BenchmarkTableTransition(b *testing.B) {
	tbl := NewTable()
	trace := "trace-trans"
	tbl.SetPriority(trace, 1)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		to := Throttled
		reason := "throttle-test"
		if i%2 == 1 {
			to = Running
			reason = ""
		}
		st, ok := tbl.Transition(trace, to, reason)
		if !ok || st.Run != to {
			b.Fatalf("transition to %v failed", to)
		}
	}
}

// BenchmarkTableCompareAndSet measures optimistic-concurrency revision-checked mutations.
func BenchmarkTableCompareAndSet(b *testing.B) {
	tbl := NewTable()
	trace := "trace-cas"
	st, _ := tbl.SetPriority(trace, 0)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		st.Priority = i
		next, ok := tbl.CompareAndSet(trace, st.Rev, st)
		if !ok {
			b.Fatalf("CAS failed at iteration %d (rev %d)", i, st.Rev)
		}
		st = next
	}
}

// BenchmarkTableRecontinue measures lineage handoff across context resets with carried budgets.
func BenchmarkTableRecontinue(b *testing.B) {
	tbl := NewTableWithLimit(512)
	now := time.Date(2026, time.September, 5, 12, 0, 0, 0, time.UTC)
	const numTraces = 64
	traces := make([]string, numTraces)
	for i := 0; i < numTraces; i++ {
		traces[i] = fmt.Sprintf("trace-rec-%d", i)
	}
	tbl.SetBudget(traces[0], Budget{ContextTokensLeft: 1000})
	tbl.StartTimeBudget(traces[0], 10*time.Minute, now)

	fresh := Budget{TurnsLeft: 100, TokensLeft: 50000}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		parent := traces[i%numTraces]
		child := traces[(i+1)%numTraces]
		st := tbl.RecontinueAt(parent, child, fresh, now)
		if st.Generation == 0 {
			b.Fatal("recontinue generation was 0")
		}
	}
}

// BenchmarkSchedulerPick measures scheduling election over active sessions under
// StrictPriority and WeightedFair policies.
func BenchmarkSchedulerPick(b *testing.B) {
	for _, policy := range []Policy{StrictPriority, WeightedFair} {
		b.Run(policy.String(), func(b *testing.B) {
			tbl := NewTable()
			const numSessions = 20
			for i := 0; i < numSessions; i++ {
				trace := fmt.Sprintf("sched-trace-%02d", i)
				tbl.SetPriority(trace, (i%5)+1)
				tbl.SetBudget(trace, Budget{TurnsLeft: 1000, TokensLeft: 100000})
			}

			sched := NewScheduler(policy)
			sched.Attach(tbl, AttachOptions{})

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				winner, ok := sched.Pick()
				if !ok || winner.TraceID == "" {
					b.Fatal("Pick returned no winner")
				}
			}
		})
	}
}

// BenchmarkPoolDrawReturn measures fleet-wide token pool allocation and reclamation.
func BenchmarkPoolDrawReturn(b *testing.B) {
	b.Run("Single", func(b *testing.B) {
		pool := NewPool(100_000_000)

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			granted, ok := pool.Draw(1000)
			if !ok || granted != 1000 {
				b.Fatalf("Draw failed: granted %d, ok %v", granted, ok)
			}
			pool.Return(1000)
		}
	})

	b.Run("Parallel", func(b *testing.B) {
		pool := NewPool(100_000_000)

		b.ReportAllocs()
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				granted, ok := pool.Draw(500)
				if !ok || granted != 500 {
					b.Fatalf("Draw failed: granted %d, ok %v", granted, ok)
				}
				pool.Return(500)
			}
		})
	})
}

// BenchmarkCostRing measures per-session cost ring pushing and summary computation.
func BenchmarkCostRing(b *testing.B) {
	b.Run("Push", func(b *testing.B) {
		var ring CostRing
		c := TurnCost{OutputTokens: 150, ContextTokens: 2500}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			ring = ring.push(c)
		}
		if ring.Count == 0 {
			b.Fatal("unexpected empty cost ring")
		}
	})

	b.Run("Summary", func(b *testing.B) {
		var ring CostRing
		for i := 0; i < CostRingSize; i++ {
			ring = ring.push(TurnCost{OutputTokens: 100 + i*10, ContextTokens: 1000 + i*100})
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			s := ring.CostSummary()
			if s.Count != CostRingSize {
				b.Fatalf("got summary count %d, want %d", s.Count, CostRingSize)
			}
		}
	})
}

// BenchmarkTimeBudget measures wall-clock envelope evaluation and queries.
func BenchmarkTimeBudget(b *testing.B) {
	tbl := NewTable()
	trace := "trace-time"
	now := time.Date(2026, time.September, 5, 12, 0, 0, 0, time.UTC)
	tbl.StartTimeBudget(trace, 1*time.Hour, now)

	b.Run("DecideTimeBudget", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			v := tbl.DecideTimeBudget(trace, now.Add(time.Duration(i%100)*time.Millisecond))
			if !v.Proceed {
				b.Fatal("unexpected non-proceed verdict")
			}
		}
	})

	b.Run("QueryTimeBudget", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			qv := tbl.QueryTimeBudget(trace, now.Add(time.Duration(i%100)*time.Millisecond))
			if qv.Exceeded {
				b.Fatal("unexpected time budget exceeded")
			}
		}
	})
}

func BenchmarkParseBudgetEnvelope(b *testing.B) {
	spec := "turns=20,calls=50,tokens=200000,context=64000,wall=2h,spend=$25,throughput=40/s,max-tokens=1024,gap=250ms"
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		env, err := ParseBudgetEnvelope(spec)
		if err != nil {
			b.Fatalf("ParseBudgetEnvelope failed: %v", err)
		}
		if env.Budget.TurnsLeft != 20 {
			b.Fatalf("unexpected turns_left")
		}
	}
}

func BenchmarkDescriptorRegistry_RegisterAndGet(b *testing.B) {
	store := NewMemStore()
	reg := NewRegistry(store)
	st := DefaultState("bench-trace")
	now := time.Now()
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		id := fmt.Sprintf("session-%d", i%256)
		_, err := reg.Register(id, "localhost", st, DefaultDescriptorTTL, now)
		if err != nil {
			b.Fatalf("register failed: %v", err)
		}
		_, found, err := reg.Get(id)
		if err != nil || !found {
			b.Fatalf("get failed")
		}
	}
}

func BenchmarkComposePace(b *testing.B) {
	pace := Pace{MaxTokensPerTurn: 512}
	tp := Throughput{ObservedTokensPerSec: 20, ExpectedTokensPerSec: 40}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		budget := pace.ComposePace(tp, 4096, 1024)
		if budget <= 0 {
			b.Fatalf("invalid composed budget")
		}
	}
}

func BenchmarkTimeBudget_ElapsedAndRemaining(b *testing.B) {
	now := time.Unix(1000000, 0)
	tb := NewTimeBudget().WithLimit(10 * time.Minute).Start(now)
	queryTime := now.Add(2 * time.Minute)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		elapsed := tb.Elapsed(queryTime)
		rem, ok := tb.Remaining(queryTime)
		if !ok || elapsed <= 0 || rem <= 0 {
			b.Fatalf("unexpected time budget query result")
		}
	}
}
