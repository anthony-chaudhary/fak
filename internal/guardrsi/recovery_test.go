package guardrsi

import (
	"strings"
	"testing"
)

// row builds one journal row for the recovery fold: a DECIDE row with an
// explicit verdict, an identity (tool + digest), and a refusal reason.
func row(tool, digest, verdict, reason string) map[string]any {
	return map[string]any{"kind": "DECIDE", "tool": tool, "args_digest": digest, "verdict": verdict, "reason": reason}
}

func TestRecoveryClearedAfterExactRetry(t *testing.T) {
	p := writeJournal(t, []map[string]any{
		row("Bash", "d1", "DENY", "LANE_DRAINED"),
		row("Bash", "d1", "DENY", "LANE_DRAINED"),
		row("Bash", "d1", "ALLOW", "NONE"),
	})
	rep := FoldRecovery([]string{p})
	if rep.RefusedCalls != 1 || len(rep.ByReason) != 1 {
		t.Fatalf("report = %+v", rep)
	}
	r := rep.ByReason[0]
	if r.Reason != "LANE_DRAINED" || r.Cleared != 1 || r.Looped != 0 || r.NoRetry != 0 || r.WastedRows != 1 {
		t.Fatalf("reason = %+v, want cleared=1 with 1 wasted re-deny", r)
	}
	if got := r.ClearRate(); got != 1 {
		t.Fatalf("clear rate = %v, want 1", got)
	}
}

func TestRecoveryLoopedNeverClears(t *testing.T) {
	p := writeJournal(t, []map[string]any{
		row("Bash", "d1", "DENY", "POLICY_BLOCK"),
		row("Bash", "d1", "DENY", "POLICY_BLOCK"),
		row("Bash", "d1", "DENY", "POLICY_BLOCK"),
	})
	r := FoldRecovery([]string{p}).ByReason[0]
	if r.Looped != 1 || r.Cleared != 0 || r.WastedRows != 2 {
		t.Fatalf("reason = %+v, want looped=1 wasted=2", r)
	}
}

func TestRecoveryNoExactRetryStaysUndecided(t *testing.T) {
	p := writeJournal(t, []map[string]any{
		row("Edit", "d1", "DENY", "OUT_OF_TREE_WRITE"),
		// A different digest on the same tool is a repaired call — a NEW identity,
		// never a clear of the refused one.
		row("Edit", "d2", "ALLOW", "NONE"),
	})
	r := FoldRecovery([]string{p}).ByReason[0]
	if r.NoRetry != 1 || r.Cleared != 0 || r.Looped != 0 || r.WastedRows != 0 {
		t.Fatalf("reason = %+v, want no_retry=1 (changed args never correlates)", r)
	}
}

func TestRecoveryDenyAfterClearOpensNewEpisode(t *testing.T) {
	p := writeJournal(t, []map[string]any{
		row("Bash", "d1", "DENY", "LEASE_HELD"),
		row("Bash", "d1", "ALLOW", "NONE"),
		row("Bash", "d1", "DENY", "LEASE_HELD"),
	})
	r := FoldRecovery([]string{p}).ByReason[0]
	if r.RefusedCalls != 2 || r.Cleared != 1 || r.NoRetry != 1 {
		t.Fatalf("reason = %+v, want one cleared episode and one fresh unretried refusal", r)
	}
}

func TestRecoveryNeverCorrelatesAcrossJournals(t *testing.T) {
	deny := writeJournal(t, []map[string]any{row("Bash", "d1", "DENY", "POLICY_BLOCK")})
	allow := writeJournal(t, []map[string]any{row("Bash", "d1", "ALLOW", "NONE")})
	r := FoldRecovery([]string{deny, allow}).ByReason[0]
	if r.Cleared != 0 || r.NoRetry != 1 {
		t.Fatalf("reason = %+v, want no cross-journal clear (journals are per-session)", r)
	}
}

func TestRecoveryWorstFirstByWastedRows(t *testing.T) {
	p := writeJournal(t, []map[string]any{
		row("Bash", "a", "DENY", "SMALL"),
		row("Bash", "b", "DENY", "BIG"),
		row("Bash", "b", "DENY", "BIG"),
		row("Bash", "b", "DENY", "BIG"),
	})
	rep := FoldRecovery([]string{p})
	if rep.ByReason[0].Reason != "BIG" || rep.ByReason[1].Reason != "SMALL" {
		t.Fatalf("order = %+v, want BIG (2 wasted rows) first", rep.ByReason)
	}
}

func TestRecoveryKindFallbackAndTransformClears(t *testing.T) {
	p := writeJournal(t, []map[string]any{
		// Older rows carry no verdict — the kind fallback decides, exactly as FoldRows.
		{"kind": "RESULT_DENY", "tool": "Bash", "args_digest": "d1", "reason": "POLICY_BLOCK"},
		{"kind": "DECIDE", "tool": "Bash", "args_digest": "d1", "verdict": "TRANSFORM", "reason": "NONE"},
	})
	r := FoldRecovery([]string{p}).ByReason[0]
	if r.Cleared != 1 {
		t.Fatalf("reason = %+v, want a TRANSFORM to close the episode as cleared", r)
	}
}

func TestRecoveryUnkeyedRefusalIsCountedNotCorrelated(t *testing.T) {
	p := writeJournal(t, []map[string]any{
		{"kind": "DENY", "tool": "Bash", "verdict": "DENY", "reason": "POLICY_BLOCK"},
	})
	rep := FoldRecovery([]string{p})
	if rep.Unkeyed != 1 || rep.RefusedCalls != 0 || len(rep.ByReason) != 0 {
		t.Fatalf("report = %+v, want 1 unkeyed refusal and no fabricated episode", rep)
	}
	if out := RenderRecovery(rep); !strings.Contains(out, "no args-digest") {
		t.Fatalf("render = %q, want the unkeyed limit surfaced", out)
	}
}

func TestRecoveryBlankReasonIsVisibleBucket(t *testing.T) {
	p := writeJournal(t, []map[string]any{row("Bash", "d1", "DENY", "")})
	r := FoldRecovery([]string{p}).ByReason[0]
	if r.Reason != "(blank)" {
		t.Fatalf("reason = %+v, want the blank token surfaced, not dropped", r)
	}
}

func TestRecoveryRenderStatesTheIdentityLimit(t *testing.T) {
	p := writeJournal(t, []map[string]any{
		row("Bash", "d1", "DENY", "POLICY_BLOCK"),
		row("Bash", "d1", "DENY", "POLICY_BLOCK"),
	})
	out := RenderRecovery(FoldRecovery([]string{p}))
	for _, want := range []string{"POLICY_BLOCK", "worst:", "limit:", "wasted-rows 1"} {
		if !strings.Contains(out, want) {
			t.Fatalf("render = %q, missing %q", out, want)
		}
	}
}

func TestRecoveryEmptyJournalGradesNothing(t *testing.T) {
	rep := FoldRecovery([]string{writeJournal(t, nil)})
	if rep.RefusedCalls != 0 || rep.Rows != 0 {
		t.Fatalf("report = %+v, want zero fold", rep)
	}
	if out := RenderRecovery(rep); !strings.Contains(out, "nothing to grade") {
		t.Fatalf("render = %q, want an honest empty verdict", out)
	}
}
