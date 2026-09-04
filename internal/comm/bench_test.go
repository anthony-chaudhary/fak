package comm

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

type benchKernel struct{}

func (b *benchKernel) Submit(ctx context.Context, c *abi.ToolCall) (abi.SubmissionHandle, abi.Verdict) {
	return abi.SubmissionHandle{Seq: 1}, abi.Verdict{Kind: abi.VerdictAllow, By: "bench"}
}

func (b *benchKernel) Reap(ctx context.Context, h abi.SubmissionHandle) (*abi.Result, error) {
	return &abi.Result{Status: abi.StatusOK}, nil
}

func (b *benchKernel) Syscall(ctx context.Context, c *abi.ToolCall) (*abi.Result, abi.Verdict) {
	return &abi.Result{Call: c, Status: abi.StatusOK}, abi.Verdict{Kind: abi.VerdictAllow, By: "bench"}
}

func (b *benchKernel) Resolver() abi.Resolver                        { return nil }
func (b *benchKernel) Negotiate(c []abi.Capability) []abi.Capability { return c }

// TestCommBenchmarkSanity verifies that the benchmark harness and group setup are operational.
func TestCommBenchmarkSanity(t *testing.T) {
	ms := []Member{
		{ID: "rank-0", Lane: "lane-a", Weight: 1.0},
		{ID: "rank-1", Lane: "lane-b", Weight: 1.0},
	}
	g, err := New("wave-test", "trace-test", ms)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if g.Size() != 2 {
		t.Fatalf("Size: got %d, want 2", g.Size())
	}
}

// BenchmarkComm measures rank routing and collective communication operations in a loop.
func BenchmarkComm(b *testing.B) {
	ms := []Member{
		{ID: "rank-0", Lane: "lane-a", Weight: 1.0},
		{ID: "rank-1", Lane: "lane-b", Weight: 1.0},
		{ID: "rank-2", Lane: "lane-a", Weight: 1.0},
		{ID: "rank-3", Lane: "lane-b", Weight: 1.0},
	}
	g, err := New("wave-bench", "trace-bench", ms)
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	k := &benchKernel{}
	ctx := context.Background()
	payload := abi.Ref{
		Kind:   abi.RefInline,
		Inline: []byte("benchmark-payload"),
		Len:    17,
		Scope:  abi.ScopeFleet,
		Taint:  abi.TaintTainted,
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		// Exercise rank routing
		r, err := g.Rank("rank-2")
		if err != nil || r != 2 {
			b.Fatalf("unexpected rank: %d, err: %v", r, err)
		}
		// Exercise collective broadcast
		if _, err := g.Broadcast(ctx, k, payload); err != nil {
			b.Fatalf("Broadcast: %v", err)
		}
	}
}
