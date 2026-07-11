package fleetpane

import "testing"

// TestPartitionWorkerHealth pins the decenter split: only the account auth wall
// waits on a person; dead/stale/attention are the fleet's to clear; healthy and
// completed-final are not an operator action and appear in neither bucket.
func TestPartitionWorkerHealth(t *testing.T) {
	h := WorkerHealth{Available: true, Counts: map[string]int{
		"healthy":              4,
		"completed-final":      2,
		"dead":                 3,
		"stale-transcript":     1,
		"auth-or-rate-blocked": 2,
		"attention":            1,
	}}
	needHuman, fleetClears := PartitionWorkerHealth(h)

	if len(needHuman) != 1 || needHuman[0].Class != "auth-or-rate-blocked" || needHuman[0].Count != 2 {
		t.Fatalf("needHuman = %+v, want only auth-or-rate-blocked=2", needHuman)
	}
	if got := bucketClasses(fleetClears); got != "attention,dead,stale-transcript" {
		t.Fatalf("fleetClears classes = %q, want attention,dead,stale-transcript", got)
	}
	if got := sumBuckets(fleetClears); got != 5 {
		t.Fatalf("fleetClears total = %d, want 5 (dead 3 + stale 1 + attention 1)", got)
	}
}

// TestPartitionWorkerHealthCanonicalOrder confirms the needs-you/fleet buckets keep
// the pane's canonical class order (healthClassOrder) so the split cannot drift from
// the base health line's ordering.
func TestPartitionWorkerHealthCanonicalOrder(t *testing.T) {
	h := WorkerHealth{Available: true, Counts: map[string]int{
		"attention":        1,
		"dead":             1,
		"stale-transcript": 1,
	}}
	_, fleetClears := PartitionWorkerHealth(h)
	want := []string{"dead", "stale-transcript", "attention"} // canonical order, not map order
	if len(fleetClears) != len(want) {
		t.Fatalf("fleetClears = %+v, want %v", fleetClears, want)
	}
	for i, b := range fleetClears {
		if b.Class != want[i] {
			t.Fatalf("fleetClears[%d].Class = %q, want %q", i, b.Class, want[i])
		}
	}
}

// TestWorkerHealthTriageLine covers the render helper: a split renders both sides
// with per-class detail; an all-healthy or unavailable summary renders nothing.
func TestWorkerHealthTriageLine(t *testing.T) {
	line := WorkerHealthTriageLine(WorkerHealth{Available: true, Counts: map[string]int{
		"dead": 2, "auth-or-rate-blocked": 1,
	}})
	want := "health-triage: needs-you=1 (auth-or-rate-blocked=1) fleet-clears=2 (dead=2)"
	if line != want {
		t.Fatalf("triage line = %q, want %q", line, want)
	}
	if got := WorkerHealthTriageLine(WorkerHealth{Available: true, Counts: map[string]int{"healthy": 3}}); got != "" {
		t.Fatalf("all-healthy line = %q, want empty", got)
	}
	if got := WorkerHealthTriageLine(WorkerHealth{Available: false}); got != "" {
		t.Fatalf("unavailable line = %q, want empty", got)
	}
}

// TestFleetTriageEnforced pins the soak switch to "enforce" only.
func TestFleetTriageEnforced(t *testing.T) {
	if !FleetTriageEnforced("enforce") || !FleetTriageEnforced("ENFORCE") || !FleetTriageEnforced(" enforce ") {
		t.Fatal("enforce (any case/space) must flip the fold")
	}
	for _, mode := range []string{"", "warn", "on", "true"} {
		if FleetTriageEnforced(mode) {
			t.Fatalf("mode %q must not flip the fold", mode)
		}
	}
}

// TestFleetTriageSelfcheck runs the shipped no-I/O proof.
func TestFleetTriageSelfcheck(t *testing.T) {
	if err := TriageSelfcheck(); err != nil {
		t.Fatalf("TriageSelfcheck: %v", err)
	}
}
