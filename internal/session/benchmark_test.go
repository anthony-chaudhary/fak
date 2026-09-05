package session

import (
	"fmt"
	"testing"
	"time"
)

func BenchmarkTable_Decide_Unbounded(b *testing.B) {
	t := NewTable()
	trace := "bench-trace"
	t.SetBudget(trace, Budget{TurnsLeft: Unbounded, TokensLeft: Unbounded})
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		v := t.Decide(trace)
		if !v.Proceed {
			b.Fatalf("expected proceed")
		}
	}
}

func BenchmarkTable_Decide_Bounded(b *testing.B) {
	t := NewTable()
	trace := "bench-trace"
	t.SetBudget(trace, Budget{TurnsLeft: b.N + 1000, TokensLeft: Unbounded})
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		v := t.Decide(trace)
		if !v.Proceed {
			b.Fatalf("expected proceed")
		}
	}
}

func BenchmarkTable_DebitUsage(b *testing.B) {
	t := NewTable()
	trace := "bench-trace"
	t.SetBudget(trace, Budget{TokensLeft: b.N*100 + 10000, ContextTokensLeft: 10000000})
	u := Usage{OutputTokens: 50, ContextTokens: 1000, DurationNanos: 100000}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		st := t.DebitUsage(trace, u)
		if st.Run == Stopped {
			b.Fatalf("unexpected stopped state")
		}
	}
}

func BenchmarkTable_DebitToolCall(b *testing.B) {
	t := NewTable()
	trace := "bench-trace"
	t.SetBudget(trace, Budget{ToolCallsLeft: b.N + 1000})
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		v := t.DebitToolCall(trace)
		if !v.Proceed {
			b.Fatalf("expected proceed")
		}
	}
}

func BenchmarkTable_Snapshot(b *testing.B) {
	t := NewTable()
	for i := 0; i < 64; i++ {
		trace := fmt.Sprintf("trace-%d", i)
		t.SetBudget(trace, Budget{TurnsLeft: 100, TokensLeft: 5000})
		t.SetPriority(trace, i%8)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		snap := t.Snapshot()
		if len(snap) != 64 {
			b.Fatalf("expected 64 sessions in snapshot")
		}
	}
}

func BenchmarkTable_Transition(b *testing.B) {
	t := NewTable()
	trace := "bench-trace"
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if i%2 == 0 {
			t.Transition(trace, Throttled, "bench_throttle")
		} else {
			t.Transition(trace, Running, "")
		}
	}
}

func BenchmarkPool_DrawAndReturn(b *testing.B) {
	p := NewPool(1000000)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		granted, ok := p.Draw(100)
		if !ok || granted != 100 {
			b.Fatalf("draw failed")
		}
		p.Return(100)
	}
}

func BenchmarkCostRing_Push(b *testing.B) {
	r := CostRing{}
	c := TurnCost{OutputTokens: 150, ContextTokens: 4000}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		r = r.push(c)
	}
}

func BenchmarkCostSummary(b *testing.B) {
	r := CostRing{}
	for i := 0; i < CostRingSize; i++ {
		r = r.push(TurnCost{OutputTokens: 100 + i*10, ContextTokens: 1000 + i*50})
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		s := r.CostSummary()
		if s.Latest == 0 {
			b.Fatalf("invalid cost summary")
		}
	}
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

func BenchmarkScheduler_PickStrictPriority(b *testing.B) {
	t := NewTable()
	for i := 0; i < 32; i++ {
		trace := fmt.Sprintf("trace-%d", i)
		t.SetBudget(trace, Budget{TurnsLeft: 1000, TokensLeft: 50000})
		t.SetPriority(trace, i%4)
	}
	s := NewScheduler(StrictPriority)
	s.Attach(t, AttachOptions{})
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		st, ok := s.Pick()
		if !ok || st.TraceID == "" {
			b.Fatalf("pick failed")
		}
	}
}

func BenchmarkScheduler_PickWeightedFair(b *testing.B) {
	t := NewTable()
	for i := 0; i < 32; i++ {
		trace := fmt.Sprintf("trace-%d", i)
		t.SetBudget(trace, Budget{TurnsLeft: 1000, TokensLeft: 50000})
		t.SetPriority(trace, i%4)
	}
	s := NewScheduler(WeightedFair)
	s.Attach(t, AttachOptions{})
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		st, ok := s.Pick()
		if !ok || st.TraceID == "" {
			b.Fatalf("pick failed")
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
