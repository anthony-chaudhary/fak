package devexmeter

import "testing"

func TestFoldFrictionRanksMeasuredWaste(t *testing.T) {
	events := []ToolEvent{
		{SessionID: "s", Turn: 1, Seq: 1, Action: "edit"},
		{SessionID: "s", Turn: 1, Seq: 2, Action: "read", Path: "a.go", ContentDigest: "h1", Tokens: 1},
		{SessionID: "s", Turn: 1, Seq: 3, Action: "read", Path: "a.go", ContentDigest: "h1", Tokens: 30, ReadCause: CausePostCompaction},
		{SessionID: "s", Turn: 1, Seq: 4, Tool: "Bash", ArgsDigest: "same", Verdict: "DENY", Reason: "POLICY_BLOCK", Tokens: 1},
		{SessionID: "s", Turn: 1, Seq: 5, Tool: "Bash", ArgsDigest: "same", Verdict: "DENY", Reason: "POLICY_BLOCK", Tokens: 40},
		{SessionID: "s", Turn: 1, Seq: 6, Tool: "Bash", Outcome: "timeout", Reason: "TOOL_TIMEOUT", Tokens: 20},
	}

	rep := FoldFriction(events)
	if rep.Schema != FrictionSchema {
		t.Fatalf("schema = %q, want %q", rep.Schema, FrictionSchema)
	}
	if len(rep.Ranked) < 3 {
		t.Fatalf("ranked = %+v, want at least retry/read/hang classes", rep.Ranked)
	}
	want := []struct {
		class  string
		tokens int
	}{
		{ClassRetryAfterRefusal, 40},
		{ClassReReadWaste, 30},
		{ClassHangRecovery, 20},
	}
	for i, w := range want {
		if rep.Ranked[i].Class != w.class || rep.Ranked[i].WastedTokens != w.tokens {
			t.Fatalf("rank %d = %+v, want class=%s tokens=%d", i, rep.Ranked[i], w.class, w.tokens)
		}
	}
	if rep.Abandoned.AbandonedTurns != 0 {
		t.Fatalf("abandoned = %+v, want artifact turn excluded", rep.Abandoned)
	}
}

func TestReadWasteCountsUnchangedReReadsByCause(t *testing.T) {
	events := []ToolEvent{
		{SessionID: "s1", Turn: 1, Action: "read", Path: "a.go", ContentDigest: "h1", Tokens: 10},
		{SessionID: "s1", Turn: 2, Action: "read", Path: "a.go", ContentDigest: "h1", Tokens: 15, AfterCompaction: true},
		{SessionID: "s1", Turn: 3, Action: "read", Path: "a.go", ContentDigest: "h2", Tokens: 99},
		{SessionID: "s2", Turn: 1, Action: "read", Path: "a.go", ContentDigest: "h1", Tokens: 5},
		{SessionID: "s1", Turn: 4, Action: "read", Path: "b.go", ContentDigest: "h3", Tokens: 4},
		{SessionID: "s1", Turn: 5, Action: "read", Path: "b.go", ContentDigest: "h3", Tokens: 6, AfterRevert: true},
		{SessionID: "s1", Turn: 6, Action: "read", Path: "c.go", Tokens: 3},
		{SessionID: "s1", Turn: 7, Action: "read", Path: "c.go", Tokens: 3},
	}

	rep := FoldReadWaste(events)
	if rep.Reads != 8 || rep.ReReads != 2 || rep.UnhashedReads != 2 {
		t.Fatalf("read counts = %+v, want reads=8 rereads=2 unhashed=2", rep)
	}
	if rep.WastedTokens != 21 || rep.WastedTurns != 2 {
		t.Fatalf("waste = %+v, want tokens=21 turns=2", rep)
	}
	if got := causeByName(rep.ByCause, CausePostCompaction); got.WastedTokens != 15 || got.Events != 1 {
		t.Fatalf("post-compaction = %+v, want one 15-token re-read", got)
	}
	if got := causeByName(rep.ByCause, CausePostRevert); got.WastedTokens != 6 || got.Events != 1 {
		t.Fatalf("post-revert = %+v, want one 6-token re-read", got)
	}
	if len(rep.Files) != 2 || rep.Files[0].Path != "a.go" {
		t.Fatalf("files = %+v, want a.go worst first", rep.Files)
	}
}

func TestRetryAfterRefusalAttributesLoopCostToFirstReason(t *testing.T) {
	events := []ToolEvent{
		{SessionID: "s1", Turn: 1, Tool: "Bash", ArgsDigest: "d1", Verdict: "DENY", Reason: "POLICY_BLOCK", Tokens: 1},
		{SessionID: "s1", Turn: 2, Tool: "Bash", ArgsDigest: "d1", Verdict: "DENY", Reason: "POLICY_BLOCK", Tokens: 11},
		{SessionID: "s1", Turn: 3, Tool: "Bash", ArgsDigest: "d1", Verdict: "ALLOW"},
		{SessionID: "s1", Turn: 4, Tool: "Edit", ArgsDigest: "d2", Verdict: "DENY", Reason: "OUT_OF_TREE_WRITE", Tokens: 1},
		{SessionID: "s1", Turn: 5, Tool: "Edit", ArgsDigest: "d2", Verdict: "DENY", Reason: "OUT_OF_TREE_WRITE", Tokens: 13},
		{SessionID: "s2", Turn: 1, Tool: "Edit", ArgsDigest: "d2", Verdict: "ALLOW"},
		{SessionID: "s1", Turn: 6, Tool: "Bash", Verdict: "DENY", Reason: "LEASE_HELD"},
	}

	rep := FoldRetryAfterRefusal(events)
	if rep.RefusedCalls != 2 || rep.Unkeyed != 1 || rep.RetryRows != 2 {
		t.Fatalf("report = %+v, want 2 refused calls, 1 unkeyed, 2 retry rows", rep)
	}
	if rep.WastedTokens != 24 || rep.WastedTurns != 2 {
		t.Fatalf("waste = %+v, want tokens=24 turns=2", rep)
	}
	out := retryByReason(rep.ByReason, "OUT_OF_TREE_WRITE")
	if out.Looped != 1 || out.Cleared != 0 || out.WastedTokens != 13 {
		t.Fatalf("OUT_OF_TREE_WRITE = %+v, want looped with 13 wasted tokens", out)
	}
	policy := retryByReason(rep.ByReason, "POLICY_BLOCK")
	if policy.Cleared != 1 || policy.Looped != 0 || policy.WastedTokens != 11 || policy.ClearRate() != 1 {
		t.Fatalf("POLICY_BLOCK = %+v, want cleared with one wasted retry", policy)
	}
}

func TestTimeToFirstUsefulActionRollsUpByLane(t *testing.T) {
	events := []ToolEvent{
		{SessionID: "a1", Lane: "cmd", Turn: 1, Seq: 1, Action: "read", Tokens: 10, AtMillis: 100},
		{SessionID: "a1", Lane: "cmd", Turn: 2, Seq: 2, Action: "read", Tokens: 20, AtMillis: 250},
		{SessionID: "a1", Lane: "cmd", Turn: 2, Seq: 3, Action: "edit", Tokens: 1, AtMillis: 300},
		{SessionID: "a2", Lane: "cmd", Turn: 1, Seq: 1, Action: "read", Tokens: 50, AtMillis: 1000},
		{SessionID: "a2", Lane: "cmd", Turn: 2, Seq: 2, Action: "commit", AtMillis: 1100},
		{SessionID: "b1", Lane: "docs", Turn: 1, Seq: 1, Action: "read", Tokens: 9, AtMillis: 1},
	}

	rep := FoldTimeToFirstUsefulAction(events)
	if rep.Sessions != 3 || rep.SessionsWithAction != 2 || rep.SessionsWithoutAction != 1 {
		t.Fatalf("sessions = %+v, want 3 total, 2 with action, 1 without", rep)
	}
	cmd := laneByName(rep.ByLane, "cmd")
	if cmd.TokensBeforeAction != 80 || cmd.TurnsBeforeAction != 2 {
		t.Fatalf("cmd = %+v, want total tokens=80 turns=2", cmd)
	}
	if cmd.MedianTokensBeforeAction != 40 || cmd.MedianTurnsBeforeAction != 1 || cmd.MedianMillisToFirstAction != 150 {
		t.Fatalf("cmd medians = %+v, want tokens=40 turns=1 ms=150", cmd)
	}
	docs := laneByName(rep.ByLane, "docs")
	if docs.SessionsWithoutAction != 1 {
		t.Fatalf("docs = %+v, want one no-action session", docs)
	}
}

func TestAbandonedTurnsClassifyBlockedLoopedAndGiveUp(t *testing.T) {
	events := []ToolEvent{
		{SessionID: "s", Turn: 1, Seq: 1, Action: "read", Tokens: 5},
		{SessionID: "s", Turn: 2, Seq: 1, Tool: "Bash", ArgsDigest: "lease", Verdict: "DENY", Reason: "LEASE_HELD", Tokens: 7},
		{SessionID: "s", Turn: 3, Seq: 1, Tool: "Bash", ArgsDigest: "lease", Verdict: "DENY", Reason: "LEASE_HELD", Tokens: 9},
		{SessionID: "s", Turn: 4, Seq: 1, Action: "edit", Tokens: 1},
	}

	rep := FoldAbandonedTurns(events)
	if rep.Turns != 4 || rep.AbandonedTurns != 3 || rep.AbandonedRate != 0.75 {
		t.Fatalf("report = %+v, want 3/4 abandoned", rep)
	}
	if rep.WastedTokens != 21 {
		t.Fatalf("wasted tokens = %d, want 21", rep.WastedTokens)
	}
	if got := causeByName(rep.ByCause, CauseNoProgressGiveUp); got.Events != 1 || got.WastedTokens != 5 {
		t.Fatalf("give-up cause = %+v, want one 5-token turn", got)
	}
	if got := causeByName(rep.ByCause, CauseBlockedByLease); got.Events != 1 || got.WastedTokens != 7 {
		t.Fatalf("lease cause = %+v, want one 7-token turn", got)
	}
	if got := causeByName(rep.ByCause, CauseLoopedRefusal); got.Events != 1 || got.WastedTokens != 9 {
		t.Fatalf("loop cause = %+v, want one 9-token turn", got)
	}
}

func causeByName(rows []CauseTotal, name string) CauseTotal {
	for _, row := range rows {
		if row.Cause == name {
			return row
		}
	}
	return CauseTotal{}
}

func retryByReason(rows []RefusalRetry, reason string) RefusalRetry {
	for _, row := range rows {
		if row.Reason == reason {
			return row
		}
	}
	return RefusalRetry{}
}

func laneByName(rows []LaneOnboarding, lane string) LaneOnboarding {
	for _, row := range rows {
		if row.Lane == lane {
			return row
		}
	}
	return LaneOnboarding{}
}
