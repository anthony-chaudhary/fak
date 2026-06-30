package main

import (
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gardenbundle"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// TestRegisterGardenTickLoopArmsDurableUnit proves the tick registers as a durable,
// armed loop unit (the #1281 precedent), so it is visible to `fak loop health` and
// re-arms at boot. Re-registering is idempotent (keeps the original CreatedUnixNano).
func TestRegisterGardenTickLoopArmsDurableUnit(t *testing.T) {
	registry := filepath.Join(t.TempDir(), "loop-registry.json")
	if err := registerGardenTickLoop(registry); err != nil {
		t.Fatalf("registerGardenTickLoop: %v", err)
	}
	reg, err := loopmgr.LoadRegistry(registry)
	if err != nil {
		t.Fatalf("LoadRegistry: %v", err)
	}
	job, ok := reg.Get(gardenTickLoopID)
	if !ok {
		t.Fatalf("loop %q not registered", gardenTickLoopID)
	}
	if !job.State.Armed() {
		t.Fatalf("loop %q state = %q, want armed", gardenTickLoopID, job.State)
	}
	if job.Schedule.IntervalSeconds != gardenTickIntervalSeconds {
		t.Fatalf("interval = %d, want %d", job.Schedule.IntervalSeconds, gardenTickIntervalSeconds)
	}
	created := job.CreatedUnixNano

	// Re-register: idempotent, original creation timestamp preserved.
	if err := registerGardenTickLoop(registry); err != nil {
		t.Fatalf("re-register: %v", err)
	}
	reg2, _ := loopmgr.LoadRegistry(registry)
	job2, _ := reg2.Get(gardenTickLoopID)
	if job2.CreatedUnixNano != created {
		t.Fatalf("re-register changed CreatedUnixNano %d -> %d", created, job2.CreatedUnixNano)
	}
}

// TestWitnessGardenTickRecordsRunEnd proves every tick appends a witnessed run-end to
// the loop ledger, so a witnessed run end (the tick + its findings) lives in the ledger.
func TestWitnessGardenTickRecordsRunEnd(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")
	plan := gardenbundle.PlanTick([]gardenbundle.MemberResult{
		{Key: "stale_leases", Label: "stale leases", State: "action"},
	}, false)

	witnessGardenTick(ledger, plan, 2, 1, 0)

	events, _, err := loopmgr.LoadPrefix(ledger)
	if err != nil {
		t.Fatalf("LoadPrefix: %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("want 1 ledger event, got %d", len(events))
	}
	ev := events[0]
	if ev.LoopID != gardenTickLoopID {
		t.Fatalf("LoopID = %q, want %q", ev.LoopID, gardenTickLoopID)
	}
	if ev.Kind != loopmgr.EventEnd || ev.Status != loopmgr.StatusWitnessedDone {
		t.Fatalf("kind/status = %s/%s, want end/witnessed_done", ev.Kind, ev.Status)
	}
	if ev.Metrics["reaped_leases"] != 2 {
		t.Fatalf("reaped_leases metric = %d, want 2", ev.Metrics["reaped_leases"])
	}
}
