package guardedremotemaint

import (
	"context"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/remotemaint"
)

// BenchmarkExecuteAuthenticatedRoundTrip measures one authenticated MCP round
// trip of the canonical fak_remote_maintenance tool through the embedded
// gateway middleware, including request encoding, adjudication chain
// traversal, and receipt admission/binding - the per-operation cost of the
// facade's production path.
func BenchmarkExecuteAuthenticatedRoundTrip(b *testing.B) {
	g := facadeGrant(time.Now().UTC())
	s, err := NewAgentSession(HostConfig{Authority: newFacadeAuthority(), AgentID: "bench-agent"})
	if err != nil {
		b.Fatal(err)
	}
	defer s.Close()
	req := remotemaint.Request{ProfileID: g.ProfileID, OperationDigest: g.OperationDigest}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := s.Execute(context.Background(), req)
		if err != nil {
			b.Fatal(err)
		}
		if !out.Authorized || out.Effect != "completed" || out.Receipt == nil || !out.Receipt.Verified {
			b.Fatalf("unexpected outcome=%+v", out)
		}
	}
}
