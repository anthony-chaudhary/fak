package plancfi

import (
	"context"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// BenchmarkAdjudicate_NoPlan measures inactive CFI baseline overhead on an undeclared trace.
func BenchmarkAdjudicate_NoPlan(b *testing.B) {
	ctx := context.Background()
	adj := New(NewLedger())
	c := call("search_flights", "unplanned-trace")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v := adj.Adjudicate(ctx, c)
		if v.Kind != abi.VerdictDefer {
			b.Fatalf("expected Defer, got %v", v.Kind)
		}
	}
}

// BenchmarkAdjudicate_AllowedSetConform measures conforming call adjudication in AllowedSet mode.
func BenchmarkAdjudicate_AllowedSetConform(b *testing.B) {
	ctx := context.Background()
	l := NewLedger()
	l.Declare("trace-1", airlinePlan)
	adj := New(l)
	c := call("search_flights", "trace-1")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v := adj.Adjudicate(ctx, c)
		if v.Kind != abi.VerdictDefer {
			b.Fatalf("expected Defer, got %v", v.Kind)
		}
	}
}

// BenchmarkAdjudicate_AllowedSetDeviation measures deviation detection and verdict fabrication in AllowedSet mode.
func BenchmarkAdjudicate_AllowedSetDeviation(b *testing.B) {
	ctx := context.Background()
	l := NewLedger()
	l.Declare("trace-1", airlinePlan)
	adj := New(l)
	c := call("send_email", "trace-1")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v := adj.Adjudicate(ctx, c)
		if v.Kind != VerdictRequireApproval {
			b.Fatalf("expected RequireApproval, got %v", v.Kind)
		}
	}
}

// BenchmarkAdjudicate_SequenceConform measures conforming sequence transitions in Sequence mode.
func BenchmarkAdjudicate_SequenceConform(b *testing.B) {
	ctx := context.Background()
	l := NewLedger()
	tools := []string{"step_a", "step_b", "step_c", "step_d"}
	l.Declare("trace-seq", Plan{Tools: tools, Mode: Sequence})
	adj := New(l)
	calls := make([]*abi.ToolCall, len(tools))
	for i, t := range tools {
		calls[i] = call(t, "trace-seq")
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c := calls[i%len(calls)]
		v := adj.Adjudicate(ctx, c)
		if v.Kind != abi.VerdictDefer {
			b.Fatalf("expected Defer, got %v", v.Kind)
		}
	}
}

// BenchmarkAdjudicate_SequenceDeviation measures deviation trapping on out-of-sequence calls.
func BenchmarkAdjudicate_SequenceDeviation(b *testing.B) {
	ctx := context.Background()
	l := NewLedger()
	l.Declare("trace-seq", Plan{Tools: []string{"step_a", "step_b"}, Mode: Sequence})
	adj := New(l)
	c := call("step_z", "trace-seq")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v := adj.Adjudicate(ctx, c)
		if v.Kind != VerdictRequireApproval {
			b.Fatalf("expected RequireApproval, got %v", v.Kind)
		}
	}
}

// BenchmarkAdjudicate_Parallel measures concurrent adjudication throughput across goroutines.
func BenchmarkAdjudicate_Parallel(b *testing.B) {
	ctx := context.Background()
	l := NewLedger()
	l.Declare("trace-par", airlinePlan)
	adj := New(l)
	c := call("search_flights", "trace-par")
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			v := adj.Adjudicate(ctx, c)
			if v.Kind != abi.VerdictDefer {
				b.Fatalf("expected Defer, got %v", v.Kind)
			}
		}
	})
}

// BenchmarkLedger_DeclareClear measures trace plan lifecycle management latency.
func BenchmarkLedger_DeclareClear(b *testing.B) {
	l := NewLedger()
	p := Plan{Tools: []string{"tool_1", "tool_2"}, Mode: AllowedSet}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		l.Declare("trace-lifecycle", p)
		if !l.Declared("trace-lifecycle") {
			b.Fatal("expected trace to be declared")
		}
		l.Clear("trace-lifecycle")
	}
}

// BenchmarkLedger_Conforms measures tool lookup against multi-element plan sets.
func BenchmarkLedger_Conforms(b *testing.B) {
	l := NewLedger()
	tools := make([]string, 32)
	for i := range tools {
		tools[i] = fmt.Sprintf("tool_%d", i)
	}
	l.Declare("trace-large", Plan{Tools: tools, Mode: AllowedSet})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		target := tools[i%len(tools)]
		if !l.conforms("trace-large", target) {
			b.Fatalf("expected %s to conform", target)
		}
	}
}
