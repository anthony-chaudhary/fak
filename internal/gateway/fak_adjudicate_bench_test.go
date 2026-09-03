package gateway

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/engine"
)

func newAdjudicateBenchmarkServer(b *testing.B) *Server {
	b.Helper()
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterEngine("mock", engine.MockEngine)
	abi.RegisterAdjudicator(0, toolAdj{})
	srv, err := New(Config{EngineID: "test", Model: "test-model", VDSO: true})
	if err != nil {
		b.Fatalf("New: %v", err)
	}
	srv.logf = nil
	b.Cleanup(srv.Close)
	return srv
}

func BenchmarkFakAdjudicateEqualFixture(b *testing.B) {
	srv := newAdjudicateBenchmarkServer(b)
	ctx := context.Background()
	tc, err := srv.buildCall(ctx, "allow_read", `{"path":"fixture.txt","read_only":true}`, true, "fixture-witness", "bench-trace")
	if err != nil {
		b.Fatal(err)
	}
	mcpArgs := json.RawMessage(`{"name":"fak_adjudicate","arguments":{"tool":"allow_read","arguments":{"path":"fixture.txt","read_only":true},"read_only":true,"witness":"fixture-witness","trace_id":"bench-trace"}}`)

	b.Run("direct_kernel_decide", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			if verdict := srv.k.Decide(ctx, tc); verdict.Kind != abi.VerdictAllow {
				b.Fatalf("verdict = %v, want ALLOW", verdict.Kind)
			}
		}
	})
	b.Run("mcp_fak_adjudicate", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			result, rpcErr := srv.callTool(ctx, mcpArgs)
			if rpcErr != nil || result == nil {
				b.Fatalf("call result=%v error=%+v", result, rpcErr)
			}
		}
	})
}
