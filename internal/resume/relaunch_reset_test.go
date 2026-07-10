package resume

import "testing"

func TestNewRelaunchResetRowIsWellFormedAndDerivesFromPlan(t *testing.T) {
	row := WatchdogPlanRow{
		Session:       "7d9b2146-8772-4dc5-87d6-8f240985d733",
		Account:       ".claude-gem7",
		ResumeAccount: ".claude-gem9",
		Disp:          "STOPPED_MIDTOOL",
		Rehomed:       true,
	}
	got := NewRelaunchResetRow(row, 2)

	if !got.WellFormed() {
		t.Fatalf("relaunch reset row is not well-formed: %+v", got)
	}
	if got.Schema != RelaunchResetSchema {
		t.Errorf("Schema = %q, want %q", got.Schema, RelaunchResetSchema)
	}
	if got.Session != row.Session {
		t.Errorf("Session = %q, want %q", got.Session, row.Session)
	}
	if got.Cause != "STOPPED_MIDTOOL" {
		t.Errorf("Cause = %q, want STOPPED_MIDTOOL", got.Cause)
	}
	if got.Attempt != 2 {
		t.Errorf("Attempt = %d, want 2", got.Attempt)
	}
	if got.TS != "" {
		t.Errorf("TS = %q, want empty (no clock in the pure core)", got.TS)
	}
	from, to, changed := got.Rehome()
	if from != ".claude-gem7" || to != ".claude-gem9" || !changed {
		t.Errorf("Rehome() = (%q,%q,%v), want (.claude-gem7,.claude-gem9,true)", from, to, changed)
	}
	if !got.Rehomed {
		t.Errorf("Rehomed = false, want true")
	}
}

func TestNewRelaunchResetRowTotalOverBlankAndNegativeAttempt(t *testing.T) {
	// A blank plan row yields a schema-stamped but session-less row (not well-formed),
	// and the fold skips it — never a panic.
	blank := NewRelaunchResetRow(WatchdogPlanRow{}, -3)
	if blank.WellFormed() {
		t.Errorf("blank plan row should not be well-formed: %+v", blank)
	}
	if blank.Attempt != 0 {
		t.Errorf("negative attempt = %d, want floored to 0", blank.Attempt)
	}

	// A re-home with no explicit ResumeAccount keeps RelaunchAccount == PriorAccount.
	noRehome := NewRelaunchResetRow(WatchdogPlanRow{
		Session: "abc",
		Account: ".claude-gem7",
	}, 0)
	from, to, changed := noRehome.Rehome()
	if from != to || changed {
		t.Errorf("no re-home: Rehome() = (%q,%q,%v), want equal accounts and changed=false", from, to, changed)
	}
}

func TestFoldRelaunchResetsLastWritePerSession(t *testing.T) {
	rows := []RelaunchResetRow{
		NewRelaunchResetRow(WatchdogPlanRow{Session: "S1", Account: "a", Disp: "STOPPED_MIDTOOL"}, 0),
		NewRelaunchResetRow(WatchdogPlanRow{Session: "S2", Account: "b", Disp: "STOPPED_DONE"}, 0),
		{Schema: RelaunchResetSchema, Session: "   "},                                                 // blank session: skipped, no key
		NewRelaunchResetRow(WatchdogPlanRow{Session: "S1", Account: "a", Disp: "STOPPED_MIDTOOL"}, 1), // later wins
	}
	got := FoldRelaunchResets(rows)

	if len(got) != 2 {
		t.Fatalf("fold size = %d, want 2 (S1,S2); blank skipped: %+v", len(got), got)
	}
	if got["S1"].Attempt != 1 {
		t.Errorf("S1 attempt = %d, want 1 (last write wins)", got["S1"].Attempt)
	}
	if got["S2"].Cause != "STOPPED_DONE" {
		t.Errorf("S2 cause = %q, want STOPPED_DONE", got["S2"].Cause)
	}
}

func TestFoldRelaunchResetsTotalOverNilAndEmpty(t *testing.T) {
	if got := FoldRelaunchResets(nil); len(got) != 0 {
		t.Errorf("fold over nil = %+v, want empty map", got)
	}
	if got := FoldRelaunchResets([]RelaunchResetRow{}); len(got) != 0 {
		t.Errorf("fold over empty = %+v, want empty map", got)
	}
	if got := FoldRelaunchResets([]RelaunchResetRow{{Session: ""}}); len(got) != 0 {
		t.Errorf("fold over blank-session-only = %+v, want empty map", got)
	}
}
