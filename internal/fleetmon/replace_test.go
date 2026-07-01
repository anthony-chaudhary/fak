package fleetmon

import (
	"strings"
	"testing"
	"time"
)

func TestReplaceEligibilityGate(t *testing.T) {
	now := time.Now()
	w := PlanWorker{Issue: 42, Session: "issue-42", IssueURL: "https://example/42", Area: "gateway"}
	cases := []struct {
		class    Classification
		force    bool
		eligible bool
	}{
		{ClassDead, false, true},
		{ClassAuthRateBlocked, false, true},
		{ClassStaleTranscript, false, true},
		{ClassHealthy, false, false},        // never relaunch a healthy worker
		{ClassCompletedFinal, false, false}, // never relaunch a completed worker
		{ClassStaleChild, false, false},     // the janitor's job, not replacement
		{ClassAttention, false, false},
		{ClassHealthy, true, true},   // --force overrides the gate
		{ClassAttention, true, true}, // --force overrides the gate
	}
	for _, tc := range cases {
		d := EvaluateReplace(ReplaceRequest{Worker: w, Class: tc.class, Index: 1, Force: tc.force, Now: now})
		if d.Eligible != tc.eligible {
			t.Errorf("class %s force=%v: want eligible=%v, got %v (%s)", tc.class, tc.force, tc.eligible, d.Eligible, d.Reason)
		}
	}
}

func TestReplaceSessionNaming(t *testing.T) {
	now := time.Now()
	d := EvaluateReplace(ReplaceRequest{Worker: PlanWorker{Issue: 1856, Session: "issue-1856"}, Class: ClassDead, Index: 2, Now: now})
	if d.NewSession != "issue-1856-replacement-2" {
		t.Fatalf("want issue-1856-replacement-2, got %q", d.NewSession)
	}
}

func TestReplaceZeroIndexDefaultsToOne(t *testing.T) {
	now := time.Now()
	d := EvaluateReplace(ReplaceRequest{Worker: PlanWorker{Issue: 9, Session: "issue-9"}, Class: ClassDead, Index: 0, Now: now})
	if d.NewSession != "issue-9-replacement-1" {
		t.Fatalf("index 0 should default to 1, got %q", d.NewSession)
	}
}

func TestReplacePromptCarriesForwardDiscipline(t *testing.T) {
	now := time.Now()
	w := PlanWorker{Issue: 1856, Session: "issue-1856", IssueURL: "https://github.com/anthony-chaudhary/fak/issues/1856", Area: "fleet monitor"}
	d := EvaluateReplace(ReplaceRequest{Worker: w, Class: ClassAuthRateBlocked, Index: 1, Now: now})
	p := d.Prompt
	// The safe prompt must carry forward every non-negotiable rule.
	for _, must := range []string{
		"#1856",
		"https://github.com/anthony-chaudhary/fak/issues/1856",
		"fleet monitor",           // expected area
		"git add -A",              // the never-add-all rule
		"force-push",              // the never-force-push rule
		"trunk",                   // stay on trunk
		"final report",            // the final-report requirement
		"witness",                 // proof requirement
		"REPLACEMENT",             // it identifies as a replacement
	} {
		if !strings.Contains(p, must) {
			t.Errorf("replacement prompt must mention %q", must)
		}
	}
}

func TestReplaceCustomTemplate(t *testing.T) {
	now := time.Now()
	tpl := "Redo issue {{issue}} at {{issue_url}} in area {{area}}."
	d := EvaluateReplace(ReplaceRequest{
		Worker:   PlanWorker{Issue: 7, Session: "issue-7", IssueURL: "u", Area: "core"},
		Class:    ClassDead, Index: 1, Template: tpl, Now: now,
	})
	if d.Prompt != "Redo issue 7 at u in area core." {
		t.Fatalf("custom template not rendered: %q", d.Prompt)
	}
}

func TestReplaceLedgerRowSupersedes(t *testing.T) {
	now := time.Now()
	d := EvaluateReplace(ReplaceRequest{Worker: PlanWorker{Issue: 5, Session: "issue-5"}, Class: ClassDead, Index: 1, RunID: "run-9", Now: now})
	if d.LedgerRow == nil {
		t.Fatal("an eligible replacement must produce a superseding ledger row")
	}
	r := d.LedgerRow
	if r.Outcome != string(OutcomeSuperseded) {
		t.Errorf("ledger row outcome should be superseded, got %s", r.Outcome)
	}
	if r.Session != "issue-5" || r.SupersededBy != "issue-5-replacement-1" {
		t.Errorf("ledger row should link issue-5 -> issue-5-replacement-1, got %+v", r)
	}
	if r.RunID != "run-9" {
		t.Errorf("run id should carry through, got %q", r.RunID)
	}
}

func TestReplaceRefusedRowIsNil(t *testing.T) {
	now := time.Now()
	d := EvaluateReplace(ReplaceRequest{Worker: PlanWorker{Issue: 5, Session: "issue-5"}, Class: ClassHealthy, Index: 1, Now: now})
	if d.LedgerRow != nil {
		t.Fatal("a refused replacement must not produce a ledger row")
	}
	if d.NewSession != "" {
		t.Fatal("a refused replacement must not name a new session")
	}
}

func TestReplaceAccountOverride(t *testing.T) {
	now := time.Now()
	d := EvaluateReplace(ReplaceRequest{Worker: PlanWorker{Issue: 5, Session: "issue-5", Account: "acct-a"}, Class: ClassDead, Index: 1, Account: "acct-b", Now: now})
	if d.Account != "acct-b" {
		t.Fatalf("account override should win, got %q", d.Account)
	}
}
