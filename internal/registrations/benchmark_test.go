package registrations_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/gitgate"
	_ "github.com/anthony-chaudhary/fak/internal/registrations"
)

func makeCmdCall(tool, key, cmd string) *abi.ToolCall {
	b, _ := json.Marshal(map[string]string{key: cmd})
	return &abi.ToolCall{
		Tool: tool,
		Args: abi.Ref{Kind: abi.RefInline, Inline: b},
	}
}

// BenchmarkAdjudicationPipeline measures evaluation latency across the registered
// adjudicator chain in a b.N loop for representative production tool calls.
func BenchmarkAdjudicationPipeline(b *testing.B) {
	ctx := context.Background()
	call := &abi.ToolCall{
		Tool: "read_file",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"path":"main.go"}`)},
	}
	adjudicators := abi.AdjudicatorsFor(call)
	if len(adjudicators) == 0 {
		b.Fatal("no adjudicators registered in defconfig")
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, a := range adjudicators {
			_ = a.Adjudicate(ctx, call)
		}
	}
}

// BenchmarkAdjudicationGitGateDecision measures gitgate command evaluation and
// witness decision recording configured by registrations.decision_recorder.
func BenchmarkAdjudicationGitGateDecision(b *testing.B) {
	ctx := context.Background()
	statusCall := makeCmdCall("bash", "command", "git status")
	diffCall := makeCmdCall("bash", "command", "git diff HEAD~1")

	b.Run("GitStatus", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = gitgate.Default.Adjudicate(ctx, statusCall)
		}
	})

	b.Run("GitDiff", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = gitgate.Default.Adjudicate(ctx, diffCall)
		}
	})
}

// BenchmarkResultAdmissionPipeline measures write-time result admission throughput
// across registered ResultAdmitters (normgate, ctxmmu, ifc) in a b.N loop.
func BenchmarkResultAdmissionPipeline(b *testing.B) {
	ctx := context.Background()
	call := &abi.ToolCall{
		Tool: "read_file",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"path":"main.go"}`)},
	}
	res := &abi.Result{
		Call:    call,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"content":"package main\n\nfunc main() {}\n"}`)},
		Status:  abi.StatusOK,
	}

	admitters := abi.ResultAdmittersFor(call)
	if len(admitters) == 0 {
		b.Fatal("no result admitters registered in defconfig")
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, ra := range admitters {
			_ = ra.Admit(ctx, call, res)
		}
	}
}

// BenchmarkActiveResolver measures Ref Put and Resolve operations via the ActiveResolver
// registered by internal/blob in the defconfig.
func BenchmarkActiveResolver(b *testing.B) {
	ctx := context.Background()
	resolver := abi.ActiveResolver()
	if resolver == nil {
		b.Fatal("no active resolver registered in defconfig")
	}

	smallPayload := []byte(`{"status":"ok","value":42}`)
	largePayload := make([]byte, 1024)
	for i := range largePayload {
		largePayload[i] = byte(i % 256)
	}

	b.Run("PutInline", func(b *testing.B) {
		b.SetBytes(int64(len(smallPayload)))
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = resolver.Put(ctx, smallPayload)
		}
	})

	b.Run("ResolveInline", func(b *testing.B) {
		ref, err := resolver.Put(ctx, smallPayload)
		if err != nil {
			b.Fatalf("resolver.Put failed: %v", err)
		}
		b.SetBytes(int64(len(smallPayload)))
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = resolver.Resolve(ctx, ref)
		}
	})

	b.Run("PutBlob", func(b *testing.B) {
		b.SetBytes(int64(len(largePayload)))
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = resolver.Put(ctx, largePayload)
		}
	})

	b.Run("ResolveBlob", func(b *testing.B) {
		ref, err := resolver.Put(ctx, largePayload)
		if err != nil {
			b.Fatalf("resolver.Put failed: %v", err)
		}
		b.SetBytes(int64(len(largePayload)))
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_, _ = resolver.Resolve(ctx, ref)
		}
	})
}

// BenchmarkStewardsEvaluation measures the execution latency of registered
// stewards verifying invariant soundness across the defconfig population.
func BenchmarkStewardsEvaluation(b *testing.B) {
	ctx := context.Background()
	stewards := abi.Stewards()
	if len(stewards) == 0 {
		b.Fatal("no stewards registered in defconfig")
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, s := range stewards {
			_, _ = s.Check(ctx)
		}
	}
}

// BenchmarkEngineRegistryLookup measures engine identification and driver lookup
// across all registered engines in abi in a b.N loop.
func BenchmarkEngineRegistryLookup(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ids := abi.EngineIDs()
		for _, id := range ids {
			_ = abi.Engine(id)
		}
	}
}

// BenchmarkConcurrentAdjudication measures concurrent adjudication throughput
// across parallel worker goroutines under defconfig registration load.
func BenchmarkConcurrentAdjudication(b *testing.B) {
	ctx := context.Background()
	call := &abi.ToolCall{
		Tool: "read_file",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"path":"main.go"}`)},
	}
	adjudicators := abi.AdjudicatorsFor(call)

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			for _, a := range adjudicators {
				_ = a.Adjudicate(ctx, call)
			}
		}
	})
}

// TestBenchmarksRun verifies that benchmark functions execute properly in a standard
// test run when not in -short mode.
func TestBenchmarksRun(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping benchmark runner in -short mode")
	}

	res := testing.Benchmark(BenchmarkAdjudicationPipeline)
	if res.N == 0 {
		t.Fatal("BenchmarkAdjudicationPipeline produced 0 iterations")
	}
}
