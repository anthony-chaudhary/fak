package guardrsi

import (
	"strings"
	"testing"
)

// A CHILD_CRASH row is the worst honesty hole a session can carry: it ranks above
// every verdict-quality gap in WorstBucket and takes a full per-row penalty in
// VerdictQuality — a guard that fails to keep its wrapped agent alive has a deeper
// problem than an unexplained block.
func TestChildCrashIsWorstBucketAndFullPenalty(t *testing.T) {
	p := writeJournal(t, []map[string]any{
		{"verdict": "ALLOW", "kind": "DECIDE"},
		{"verdict": "ALLOW", "kind": "DECIDE"},
		{"kind": "CHILD_CRASH", "reason": "SIGNAL_CRASH", "tool": "claude", "trace_id": "guard", "exit_code": -1},
	})
	fold := FoldRows([]string{p})
	if fold.TotalRows != 3 || fold.ChildCrash != 1 || fold.ByCrashClass["SIGNAL_CRASH"] != 1 {
		t.Fatalf("fold = %+v, want 3 rows / 1 crash / 1 SIGNAL_CRASH", fold)
	}
	// The crash must NOT bleed into verdict/reason accounting: its Kind is not an
	// unknown verdict and its Reason is not a denial reason.
	if fold.UnknownVerdict != 0 || fold.BlankReasonOnDeny != 0 || len(fold.ByReason) != 0 {
		t.Fatalf("crash leaked into verdict accounting: %+v", fold)
	}
	// One crash over three rows -> full 1.0 penalty -> (1 - 1/3)*100 = 66.667.
	if got, want := VerdictQuality(fold), 66.667; got != want {
		t.Fatalf("quality = %v, want %v (full per-row crash penalty)", got, want)
	}
	worst := WorstBucket(fold)
	if worst.Bucket != "child_crash" || worst.Count != 1 || !strings.Contains(worst.Lever, "SIGNAL_CRASH") {
		t.Fatalf("worst = %+v, want the child_crash bucket naming the class", worst)
	}
}

// A crash outranks a blank-reason-on-deny (the previously worst bucket), and a
// repaired fold (crash retired) scores strictly higher — so RunIteration credits
// closing the crash hole.
func TestChildCrashOutranksBlankReasonAndCreditsRepair(t *testing.T) {
	p := writeJournal(t, []map[string]any{
		{"verdict": "ALLOW", "kind": "DECIDE"},
		{"verdict": "DENY", "kind": "DENY"}, // blank reason on deny
		{"kind": "CHILD_CRASH", "reason": "OOM", "tool": "claude", "trace_id": "guard", "exit_code": 137},
	})
	fold := FoldRows([]string{p})
	if fold.ChildCrash != 1 || fold.BlankReasonOnDeny != 1 {
		t.Fatalf("fold = %+v, want one crash AND one blank-reason", fold)
	}
	if worst := WorstBucket(fold); worst.Bucket != "child_crash" {
		t.Fatalf("worst = %q, want child_crash to outrank blank_reason_on_deny", worst.Bucket)
	}
	if !strings.Contains(WorstBucket(fold).Lever, "OOM") {
		t.Fatalf("worst lever should name the dominant crash class OOM: %q", WorstBucket(fold).Lever)
	}
	// Retiring the crash AND the blank reason (RunIteration's repair) must strictly
	// improve the metric.
	repaired := fold
	repaired.ChildCrash = 0
	repaired.BlankReasonOnDeny = 0
	if VerdictQuality(repaired) <= VerdictQuality(fold) {
		t.Fatalf("repaired quality %v not strictly above baseline %v", VerdictQuality(repaired), VerdictQuality(fold))
	}
}

// An unclassified crash (a CHILD_CRASH row with no reason) still counts and still
// ranks worst — a crash with a missing class is not silently dropped.
func TestUnclassifiedCrashStillCounts(t *testing.T) {
	p := writeJournal(t, []map[string]any{
		{"verdict": "ALLOW", "kind": "DECIDE"},
		{"kind": "CHILD_CRASH", "tool": "claude", "trace_id": "guard"},
	})
	fold := FoldRows([]string{p})
	if fold.ChildCrash != 1 || fold.ByCrashClass["UNCLASSIFIED"] != 1 {
		t.Fatalf("fold = %+v, want an UNCLASSIFIED crash counted", fold)
	}
	if WorstBucket(fold).Bucket != "child_crash" {
		t.Fatalf("unclassified crash should still be the worst bucket: %+v", WorstBucket(fold))
	}
}
