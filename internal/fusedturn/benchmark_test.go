package fusedturn_test

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/fusedturn"
)

var (
	benchClassSink     fusedturn.OpClass
	benchCallSink      *abi.ToolCall
	benchFusedTurnSink fusedturn.FusedTurn
	benchSummarySink   fusedturn.Summary
	benchRowsSink      []fusedturn.AdjudicatedOp
	benchFamiliesSink  []fusedturn.OpClass
	benchBoolSink      bool
	benchIntSink       int
)

type benchDecider struct{}

func (benchDecider) BatchDecide(ctx context.Context, calls []*abi.ToolCall) []abi.Verdict {
	verdicts := make([]abi.Verdict, len(calls))
	for i := range calls {
		verdicts[i] = abi.Verdict{Kind: abi.VerdictAllow}
	}
	return verdicts
}

type benchWeightEngine struct{ weight bool }

func (benchWeightEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	return &abi.Result{Status: abi.StatusOK}, nil
}
func (benchWeightEngine) Caps() []abi.Capability { return nil }
func (e benchWeightEngine) WeightBearing() bool  { return e.weight }

func init() {
	abi.RegisterEngine("bench-weighty", benchWeightEngine{weight: true})
	abi.RegisterEngine("bench-classical", benchWeightEngine{weight: false})
}

func BenchmarkClassify(b *testing.B) {
	classical := fusedturn.Classical("git_commit", abi.Ref{})
	weight := fusedturn.Weight("qwen3.8", "inference", abi.Ref{})
	unknown := &abi.ToolCall{Tool: "bash"}

	b.Run("Classical", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchClassSink = fusedturn.Classify(classical)
		}
	})

	b.Run("Weight", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchClassSink = fusedturn.Classify(weight)
		}
	})

	b.Run("Unknown", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchClassSink = fusedturn.Classify(unknown)
		}
	})

	b.Run("Nil", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchClassSink = fusedturn.Classify(nil)
		}
	})
}

func BenchmarkClassifyResolved(b *testing.B) {
	declared := fusedturn.Classical("git_commit", abi.Ref{})
	declared.Engine = "bench-weighty"

	undeclaredWeight := &abi.ToolCall{Tool: "inference", Engine: "bench-weighty"}
	undeclaredClassical := &abi.ToolCall{Tool: "bash", Engine: "bench-classical"}
	unregistered := &abi.ToolCall{Tool: "mystery", Engine: "nonexistent"}

	b.Run("ExplicitDeclared", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchClassSink = fusedturn.ClassifyResolved(declared)
		}
	})

	b.Run("WeightBearingEngine", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchClassSink = fusedturn.ClassifyResolved(undeclaredWeight)
		}
	})

	b.Run("ClassicalEngine", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchClassSink = fusedturn.ClassifyResolved(undeclaredClassical)
		}
	})

	b.Run("UnregisteredRoute", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchClassSink = fusedturn.ClassifyResolved(unregistered)
		}
	})
}

func BenchmarkConstructors(b *testing.B) {
	b.Run("Classical", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchCallSink = fusedturn.Classical("git_commit", abi.Ref{})
		}
	})

	b.Run("Weight", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchCallSink = fusedturn.Weight("qwen3.8", "inference", abi.Ref{})
		}
	})

	b.Run("Tag", func(b *testing.B) {
		baseCall := &abi.ToolCall{Tool: "step"}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchCallSink = fusedturn.Tag(baseCall, fusedturn.ClassClassical)
		}
	})
}

func BenchmarkFuse(b *testing.B) {
	c1 := fusedturn.Classical("git_commit", abi.Ref{})
	w1 := fusedturn.Weight("qwen3.8", "chat", abi.Ref{})
	u1 := &abi.ToolCall{Tool: "bash"}

	smallBatch := []*abi.ToolCall{c1, w1}

	burstBatch := make([]*abi.ToolCall, 16)
	for i := range burstBatch {
		switch i % 3 {
		case 0:
			burstBatch[i] = c1
		case 1:
			burstBatch[i] = w1
		default:
			burstBatch[i] = u1
		}
	}

	largeBatch := make([]*abi.ToolCall, 64)
	for i := range largeBatch {
		switch i % 3 {
		case 0:
			largeBatch[i] = c1
		case 1:
			largeBatch[i] = w1
		default:
			largeBatch[i] = u1
		}
	}

	b.Run("SmallTurn", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchFusedTurnSink = fusedturn.Fuse(smallBatch)
		}
	})

	b.Run("BurstBatch16", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchFusedTurnSink = fusedturn.Fuse(burstBatch)
		}
	})

	b.Run("LargeBatch64", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchFusedTurnSink = fusedturn.Fuse(largeBatch)
		}
	})
}

func BenchmarkFusedTurnPredicates(b *testing.B) {
	ft := fusedturn.Fuse([]*abi.ToolCall{
		fusedturn.Classical("git_commit", abi.Ref{}),
		fusedturn.Weight("qwen3.8", "chat", abi.Ref{}),
		{Tool: "bash"},
	})

	b.Run("Fused", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchBoolSink = ft.Fused()
		}
	})

	b.Run("Counts", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchIntSink = ft.Classical() + ft.Weight() + ft.Unknown()
		}
	})
}

func BenchmarkSummary(b *testing.B) {
	ft := fusedturn.Fuse([]*abi.ToolCall{
		fusedturn.Classical("git_commit", abi.Ref{}),
		fusedturn.Weight("qwen3.8", "chat", abi.Ref{}),
		{Tool: "bash"},
	})

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSummarySink = ft.Summary()
	}
}

func BenchmarkAdjudicate(b *testing.B) {
	ctx := context.Background()
	decider := benchDecider{}

	calls16 := make([]*abi.ToolCall, 16)
	for i := range calls16 {
		if i%2 == 0 {
			calls16[i] = fusedturn.Classical("read_file", abi.Ref{})
		} else {
			calls16[i] = fusedturn.Weight("qwen3.8", "chat", abi.Ref{})
		}
	}
	ft16 := fusedturn.Fuse(calls16)

	calls64 := make([]*abi.ToolCall, 64)
	for i := range calls64 {
		if i%2 == 0 {
			calls64[i] = fusedturn.Classical("read_file", abi.Ref{})
		} else {
			calls64[i] = fusedturn.Weight("qwen3.8", "chat", abi.Ref{})
		}
	}
	ft64 := fusedturn.Fuse(calls64)

	b.Run("Batch16", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchRowsSink = ft16.Adjudicate(ctx, decider)
		}
	})

	b.Run("Batch64", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchRowsSink = ft64.Adjudicate(ctx, decider)
		}
	})
}

func BenchmarkGovernedFamilies(b *testing.B) {
	ctx := context.Background()
	decider := benchDecider{}
	ft := fusedturn.Fuse([]*abi.ToolCall{
		fusedturn.Classical("read_file", abi.Ref{}),
		fusedturn.Weight("qwen3.8", "chat", abi.Ref{}),
		{Tool: "bash"},
	})
	rows := ft.Adjudicate(ctx, decider)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchFamiliesSink = fusedturn.GovernedFamilies(rows)
	}
}

func TestBenchmarkSanity(t *testing.T) {
	call := fusedturn.Classical("git_commit", abi.Ref{})
	if got := fusedturn.Classify(call); got != fusedturn.ClassClassical {
		t.Fatalf("expected ClassClassical, got %v", got)
	}
}
