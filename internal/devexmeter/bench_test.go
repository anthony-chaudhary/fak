package devexmeter

import (
	"testing"
)

// BenchmarkDevExMeter exercises friction folding and waste counting across simulated tool events.
func BenchmarkDevExMeter(b *testing.B) {
	events := []ToolEvent{
		{SessionID: "s1", Turn: 1, Seq: 1, Action: "read", Path: "internal/devexmeter/devexmeter.go", ContentDigest: "h1", Tokens: 120},
		{SessionID: "s1", Turn: 1, Seq: 2, Action: "read", Path: "internal/devexmeter/devexmeter.go", ContentDigest: "h1", Tokens: 120, ReadCause: CausePostCompaction},
		{SessionID: "s1", Turn: 2, Seq: 3, Tool: "Bash", ArgsDigest: "cmd1", Verdict: "DENY", Reason: "POLICY_BLOCK", Tokens: 45},
		{SessionID: "s1", Turn: 2, Seq: 4, Tool: "Bash", ArgsDigest: "cmd1", Verdict: "DENY", Reason: "POLICY_BLOCK", Tokens: 50},
		{SessionID: "s1", Turn: 3, Seq: 5, Action: "edit", Edit: true, UsefulAction: true, Tokens: 80},
		{SessionID: "s2", Turn: 1, Seq: 1, Tool: "Edit", ArgsDigest: "cmd2", Verdict: "DENY", Reason: "OUT_OF_TREE_WRITE", Tokens: 30},
		{SessionID: "s2", Turn: 1, Seq: 2, Tool: "Edit", ArgsDigest: "cmd2", Verdict: "ALLOW", Tokens: 35},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rep := FoldFriction(events)
		if rep.Events != len(events) {
			b.Fatalf("expected %d events, got %d", len(events), rep.Events)
		}
	}
}
