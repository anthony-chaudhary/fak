package repoguard

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// defaultSeverityOf resolves severity with the shipped default posture (no
// overrides, enforce mode) — the labeling most tests exercise.
func defaultSeverityOf(reason string) Severity {
	return ResolveSeverity(reason, nil, "enforce")
}

func TestDecisionsFromViolations_LabelsFromResolvedSeverity(t *testing.T) {
	vs := []Violation{
		{Reason: guardReason, Op: "rm", Target: "../tools", Resolved: "C:/x/tools", Why: "outside"},
		{Reason: ReasonInteractiveHang, Op: "vim", Target: "vim x", Why: "hangs"},
		{Reason: ReasonForegroundSleep, Op: "sleep", Target: "sleep 30", Why: "blocks"},
	}
	rows := DecisionsFromViolations(vs, "Bash", "sess-1", "enforce", "2026-07-03T00:00:00Z", defaultSeverityOf)
	if len(rows) != 3 {
		t.Fatalf("want 3 rows, got %d", len(rows))
	}
	// Default posture: OUT_OF_TREE is silent-record, the two wasted-turn rungs warn.
	if rows[0].Decision != "record" {
		t.Errorf("out-of-tree row should be record by default, got %q", rows[0].Decision)
	}
	if rows[1].Decision != "advisory" {
		t.Errorf("interactive row should be advisory, got %q", rows[1].Decision)
	}
	if rows[2].Decision != "advisory" {
		t.Errorf("foreground-sleep row should be advisory, got %q", rows[2].Decision)
	}
	if rows[0].Tool != "Bash" || rows[0].Session != "sess-1" || rows[0].Mode != "enforce" {
		t.Errorf("context fields not carried: %+v", rows[0])
	}
	if rows[0].Schema != DecisionRecordSchema {
		t.Errorf("schema not stamped: %q", rows[0].Schema)
	}
}

func TestDecisionsFromViolations_DenyOverrideRelabels(t *testing.T) {
	vs := []Violation{{Reason: guardReason, Op: "rm", Target: "../tools", Resolved: "/x"}}
	sev := func(reason string) Severity {
		return ResolveSeverity(reason, map[string]Severity{guardReason: SeverityDeny}, "enforce")
	}
	rows := DecisionsFromViolations(vs, "Bash", "s", "enforce", "t", sev)
	if len(rows) != 1 || rows[0].Decision != "deny" {
		t.Fatalf("per-reason deny override should label the row deny, got %+v", rows)
	}
}

func TestDecisionsFromViolations_OffDropsRow(t *testing.T) {
	vs := []Violation{{Reason: guardReason, Op: "rm", Target: "../tools", Resolved: "/x"}}
	sev := func(reason string) Severity { return SeverityOff }
	if rows := DecisionsFromViolations(vs, "Bash", "s", "enforce", "t", sev); len(rows) != 0 {
		t.Errorf("SeverityOff should yield no rows, got %+v", rows)
	}
}

func TestDecisionsFromViolations_Empty(t *testing.T) {
	if rows := DecisionsFromViolations(nil, "Bash", "", "enforce", "t", defaultSeverityOf); rows != nil {
		t.Errorf("empty violations should yield nil rows, got %v", rows)
	}
}

func TestAppendAndSummarize_RoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "sub", "decisions.jsonl")

	// Two separate appends must accumulate (append-only, survives across hook calls).
	// Under the default posture OUT_OF_TREE is silent-record, FOREGROUND_SLEEP advisory.
	first := DecisionsFromViolations(
		[]Violation{{Reason: guardReason, Op: "rm", Target: "../a", Resolved: "/x/a"}},
		"Bash", "s1", "enforce", "2026-07-03T01:00:00Z", defaultSeverityOf)
	if err := AppendDecisions(path, first); err != nil {
		t.Fatalf("first append: %v", err)
	}
	second := DecisionsFromViolations(
		[]Violation{
			{Reason: guardReason, Op: "mv", Target: "../b", Resolved: "/x/b"},
			{Reason: ReasonForegroundSleep, Op: "sleep", Target: "sleep 9"},
		},
		"Bash", "s1", "enforce", "2026-07-03T02:00:00Z", defaultSeverityOf)
	if err := AppendDecisions(path, second); err != nil {
		t.Fatalf("second append: %v", err)
	}

	sum, err := SummarizeDecisionsFile(path, 10)
	if err != nil {
		t.Fatalf("summarize: %v", err)
	}
	if sum.Total != 3 {
		t.Errorf("Total: want 3, got %d", sum.Total)
	}
	if sum.Denies != 0 {
		t.Errorf("Denies: want 0 (all soft by default), got %d", sum.Denies)
	}
	if sum.Recorded != 2 {
		t.Errorf("Recorded: want 2 (two out-of-tree), got %d", sum.Recorded)
	}
	if sum.Advisories != 1 {
		t.Errorf("Advisories: want 1, got %d", sum.Advisories)
	}
	if sum.ByReason[guardReason] != 2 {
		t.Errorf("ByReason[%s]: want 2, got %d", guardReason, sum.ByReason[guardReason])
	}
	if sum.FirstTs != "2026-07-03T01:00:00Z" || sum.LastTs != "2026-07-03T02:00:00Z" {
		t.Errorf("time span wrong: first=%q last=%q", sum.FirstTs, sum.LastTs)
	}
	if len(sum.Recent) != 3 {
		t.Errorf("Recent: want 3, got %d", len(sum.Recent))
	}
}

func TestAppendDecisions_EmptyIsNoFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "decisions.jsonl")
	if err := AppendDecisions(path, nil); err != nil {
		t.Fatalf("append nil: %v", err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Errorf("empty append must not create a file; stat err=%v", err)
	}
}

func TestSummarizeDecisionsFile_MissingIsEmpty(t *testing.T) {
	sum, err := SummarizeDecisionsFile(filepath.Join(t.TempDir(), "nope.jsonl"), 5)
	if err != nil {
		t.Fatalf("missing journal should not error: %v", err)
	}
	if sum.Total != 0 {
		t.Errorf("missing journal should be empty, got Total=%d", sum.Total)
	}
}

func TestSummarizeDecisions_ToleratesTornLine(t *testing.T) {
	in := strings.Join([]string{
		`{"schema":"repoguard.decision/v1","decision":"deny","reason":"OUT_OF_TREE_WRITE"}`,
		`{ this is not json`,
		`{"schema":"repoguard.decision/v1","decision":"advisory","reason":"FOREGROUND_SLEEP"}`,
	}, "\n")
	sum, err := SummarizeDecisions(strings.NewReader(in), 5)
	if err != nil {
		t.Fatalf("summarize: %v", err)
	}
	if sum.Total != 2 {
		t.Errorf("torn line should be skipped, want Total=2 got %d", sum.Total)
	}
}

func TestRenderSummary_EmptyAndPopulated(t *testing.T) {
	empty := RenderSummary(DecisionSummary{ByReason: map[string]int{}})
	if !strings.Contains(empty, "no recorded decisions") {
		t.Errorf("empty render should say so, got %q", empty)
	}
	sum := DecisionSummary{
		Total: 2, Denies: 2, ByReason: map[string]int{guardReason: 2},
		Recent: []DecisionRecord{{Decision: "deny", Tool: "Bash", Reason: guardReason, Resolved: "/x/a"}},
	}
	out := RenderSummary(sum)
	if !strings.Contains(out, "2 finding(s)") || !strings.Contains(out, guardReason) {
		t.Errorf("populated render missing content: %q", out)
	}
}
