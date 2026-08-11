package terminalrelief

import (
	"testing"
	"time"
)

func testConfig() Config {
	return Config{HandleThreshold: 10000, ThreadThreshold: 500, Consecutive: 3, Cooldown: time.Hour}
}
func safeFacts() Facts {
	return Facts{PID: 42, Handles: 12000, Threads: 600, Dashboards: []Command{{Argv: []string{"fak", "info", "--gateway-url", "http://127.0.0.1:1"}}}}
}

func TestBelowThresholdNeverApplies(t *testing.T) {
	now := time.Now().UTC()
	d := Decide(Facts{PID: 42, Handles: 9999, Threads: 499}, State{Schema: Schema, PID: 42, Consecutive: 9}, testConfig(), now, true)
	if d.Verdict != "BELOW_THRESHOLD" || d.Apply || d.State.Consecutive != 0 {
		t.Fatalf("decision=%+v", d)
	}
}
func TestPersistentSafePressureProducesApply(t *testing.T) {
	now := time.Now().UTC()
	s := State{}
	for i := 1; i <= 3; i++ {
		d := Decide(safeFacts(), s, testConfig(), now.Add(time.Duration(i)*time.Second), true)
		s = d.State
		if i < 3 && (d.Verdict != "OBSERVE" || d.Apply) {
			t.Fatalf("tick %d=%+v", i, d)
		}
		if i == 3 && (d.Verdict != "APPLY" || !d.Apply || d.State.LastApplied == "") {
			t.Fatalf("tick %d=%+v", i, d)
		}
	}
}
func TestDryRunAndCooldownDoNotApply(t *testing.T) {
	now := time.Now().UTC()
	s := State{Schema: Schema, PID: 42, Consecutive: 2}
	d := Decide(safeFacts(), s, testConfig(), now, false)
	if d.Verdict != "WOULD_APPLY" || d.Apply {
		t.Fatalf("dry=%+v", d)
	}
	s = State{Schema: Schema, PID: 42, Consecutive: 2, LastApplied: now.Add(-time.Minute).Format(time.RFC3339Nano)}
	d = Decide(safeFacts(), s, testConfig(), now, true)
	if d.Verdict != "COOLDOWN" || d.Apply {
		t.Fatalf("cooldown=%+v", d)
	}
}
func TestUnsafeDescendantOrMissingDashboardAbstains(t *testing.T) {
	now := time.Now().UTC()
	f := safeFacts()
	f.UnsafeDescendants = []string{"vim.exe"}
	if d := Decide(f, State{Schema: Schema, PID: 42, Consecutive: 2}, testConfig(), now, true); d.Verdict != "ABSTAIN" || d.Apply {
		t.Fatalf("unknown=%+v", d)
	}
	f = safeFacts()
	f.Dashboards = nil
	if d := Decide(f, State{Schema: Schema, PID: 42, Consecutive: 2}, testConfig(), now, true); d.Verdict != "ABSTAIN" || d.Apply {
		t.Fatalf("missing=%+v", d)
	}
}
