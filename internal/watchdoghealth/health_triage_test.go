package watchdoghealth

import "testing"

// TestPartitionAttention pins the decenter split: a GAVE_UP monitor waits on a person, a
// DOWN monitor (the autoheal restarts it) and an UNKNOWN monitor (re-probe) are the fleet's
// to clear, and healthy / not-installed / healing monitors appear in neither bucket.
func TestPartitionAttention(t *testing.T) {
	d := Fold([]Monitor{
		{ID: "a-healthy", Installed: true, Alive: true},
		{ID: "b-noinstall"},
		{ID: "c-healing", Installed: true, Attempts: 2, MaxAttempts: 5},
		{ID: "d-down", Installed: true},
		{ID: "e-unknown", ProbeErr: true},
		{ID: "f-gaveup", Installed: true, Attempts: 5, MaxAttempts: 5},
	})
	needHuman, fleetClears := PartitionAttention(d)

	if len(needHuman) != 1 || needHuman[0].ID != "f-gaveup" || needHuman[0].Status != StatusGaveUp {
		t.Fatalf("needHuman = %+v, want only f-gaveup=GAVE_UP", needHuman)
	}
	if got := itemIDs(fleetClears); got != "d-down,e-unknown" {
		t.Fatalf("fleetClears ids = %q, want d-down,e-unknown", got)
	}
}

// TestPartitionAttentionKeepsDigestOrder confirms the buckets keep the digest's monitor
// order so the split cannot drift from the base status table's ordering.
func TestPartitionAttentionKeepsDigestOrder(t *testing.T) {
	d := Fold([]Monitor{
		{ID: "first-down", Installed: true},
		{ID: "second-unknown", ProbeErr: true},
	})
	_, fleetClears := PartitionAttention(d)
	if len(fleetClears) != 2 || fleetClears[0].ID != "first-down" || fleetClears[1].ID != "second-unknown" {
		t.Fatalf("fleetClears order = %+v, want [first-down second-unknown]", fleetClears)
	}
}

// TestTriageMonitorDispositions pins each attention-floor status to its disposition: DOWN is
// TAKE_OBVIOUS (the autoheal's runnable restart), UNKNOWN is FRESH_CONTEXT (re-probe), and
// GAVE_UP is HUMAN_RESIDUAL (a person's priority call).
func TestTriageMonitorDispositions(t *testing.T) {
	down := triageMonitor(Health{Monitor: Monitor{ID: "x", Installed: true}, Status: StatusDown})
	if down.NeedsHuman || down.Disposition != "TAKE_OBVIOUS" {
		t.Fatalf("DOWN disposition = %q needsHuman=%v, want TAKE_OBVIOUS/false", down.Disposition, down.NeedsHuman)
	}
	unknown := triageMonitor(Health{Monitor: Monitor{ID: "x", ProbeErr: true}, Status: StatusUnknown})
	if unknown.NeedsHuman || unknown.Disposition != "FRESH_CONTEXT" {
		t.Fatalf("UNKNOWN disposition = %q needsHuman=%v, want FRESH_CONTEXT/false", unknown.Disposition, unknown.NeedsHuman)
	}
	gaveup := triageMonitor(Health{Monitor: Monitor{ID: "x", Installed: true}, Status: StatusGaveUp})
	if !gaveup.NeedsHuman || gaveup.Disposition != "HUMAN_RESIDUAL" {
		t.Fatalf("GAVE_UP disposition = %q needsHuman=%v, want HUMAN_RESIDUAL/true", gaveup.Disposition, gaveup.NeedsHuman)
	}
}

// TestAttentionTriageLineEmptyWhenClear confirms an all-clear digest surfaces no split line
// (no page, no fleet churn) and a not-installed-only host likewise renders nothing.
func TestAttentionTriageLineEmptyWhenClear(t *testing.T) {
	if line := AttentionTriageLine(Fold([]Monitor{{ID: "ok", Installed: true, Alive: true}})); line != "" {
		t.Fatalf("all-healthy line = %q, want empty", line)
	}
	if line := AttentionTriageLine(Fold(nil)); line != "" {
		t.Fatalf("empty-digest line = %q, want empty", line)
	}
}

// TestNeedsHumanAttentionOnlyOnResidual confirms the enforce-mode --check condition trips on
// a GAVE_UP monitor but NOT on a fleet-clearable DOWN/UNKNOWN one — the whole point of the
// fold is that DOWN and UNKNOWN leave the human page.
func TestNeedsHumanAttentionOnlyOnResidual(t *testing.T) {
	fleetOnly := Fold([]Monitor{{ID: "d", Installed: true}, {ID: "u", ProbeErr: true}})
	if NeedsHumanAttention(fleetOnly) {
		t.Fatalf("a DOWN+UNKNOWN digest must not need a human under enforce")
	}
	if d := fleetOnly; !d.NeedsAttention {
		t.Fatalf("the base digest must still report NeedsAttention (default gate unchanged)")
	}
	withResidual := Fold([]Monitor{{ID: "g", Installed: true, Attempts: 9, MaxAttempts: 3}})
	if !NeedsHumanAttention(withResidual) {
		t.Fatalf("a GAVE_UP digest must need a human under enforce")
	}
}

// TestWatchdogTriageSelfcheck runs the packaged proof.
func TestWatchdogTriageSelfcheck(t *testing.T) {
	if err := TriageSelfcheck(); err != nil {
		t.Fatalf("TriageSelfcheck: %v", err)
	}
}
