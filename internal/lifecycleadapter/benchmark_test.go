package lifecycleadapter

import (
	"context"
	"testing"
	"time"
)

// BenchmarkLifecycleAdapter measures end-to-end capability negotiation and execution
// across builtin process-forest lifecycle adapters.
func BenchmarkLifecycleAdapter(b *testing.B) {
	ctx := context.Background()
	native := NativeFAK()
	codex := Codex()
	claude := Claude()

	deadline := time.Now().Add(time.Hour)
	reqNative := Request{
		TransactionID: "tx-bench",
		ForestID:      "forest-bench",
		MemberID:      "member-fak",
		Generation:    1,
		Operation:     Checkpoint,
		Deadline:      deadline,
	}
	reqCodex := Request{
		TransactionID: "tx-bench",
		ForestID:      "forest-bench",
		MemberID:      "member-codex",
		Generation:    1,
		Operation:     Pause,
		Deadline:      deadline,
	}
	reqClaude := Request{
		TransactionID: "tx-bench",
		ForestID:      "forest-bench",
		MemberID:      "member-claude",
		Generation:    1,
		Operation:     Resume,
		Deadline:      deadline,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if n, r := Execute(ctx, reqNative, native); !n.Supported || r.State != ResultCompleted {
			b.Fatalf("native execution failed: n=%+v r=%+v", n, r)
		}
		if n, r := Execute(ctx, reqCodex, codex); !n.Supported || r.State != ResultCompleted {
			b.Fatalf("codex execution failed: n=%+v r=%+v", n, r)
		}
		if n, r := Execute(ctx, reqClaude, claude); !n.Supported || r.State != ResultCompleted {
			b.Fatalf("claude execution failed: n=%+v r=%+v", n, r)
		}
	}
}

// BenchmarkNegotiate measures the capability matching path without adapter invocation.
func BenchmarkNegotiate(b *testing.B) {
	native := NativeFAK()
	req := Request{
		TransactionID: "tx-bench",
		ForestID:      "forest-bench",
		MemberID:      "member-fak",
		Generation:    1,
		Operation:     Prepare,
		Deadline:      time.Now().Add(time.Hour),
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		n, err := Negotiate(req, native)
		if err != nil || !n.Supported {
			b.Fatalf("negotiate failed: %v", err)
		}
	}
}

// BenchmarkExecuteCustomAdapter measures execution throughput with an injected handler.
func BenchmarkExecuteCustomAdapter(b *testing.B) {
	ctx := context.Background()
	doc := CapabilityDocument{
		Protocol:              ProtocolVersion,
		AdapterKind:           "custom-fast",
		Operations:            []Operation{Prepare, Readiness},
		ApplicationCheckpoint: false,
	}
	adapter := Custom(doc, func(_ context.Context, req Request) Result {
		return Result{
			State:       ResultCompleted,
			ReadbackRef: "custom:" + string(req.Operation),
		}
	})

	req := Request{
		TransactionID: "tx-custom",
		ForestID:      "forest-custom",
		MemberID:      "member-custom",
		Generation:    2,
		Operation:     Readiness,
		Deadline:      time.Now().Add(time.Hour),
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		n, r := Execute(ctx, req, adapter)
		if !n.Supported || r.State != ResultCompleted {
			b.Fatalf("custom execution failed: n=%+v r=%+v", n, r)
		}
	}
}
