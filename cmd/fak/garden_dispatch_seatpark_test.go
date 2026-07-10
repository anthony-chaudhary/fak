package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/seatpark"
)

// seatParkEvent is a garden-dispatch run-end with the given park reason at tsUnix seconds.
func seatParkEvent(reason string, tsUnix int64) loopmgr.Event {
	return loopmgr.Event{
		LoopID:     gardenDispatchLoopID,
		Kind:       loopmgr.EventEnd,
		Reason:     reason,
		TSUnixNano: tsUnix * 1_000_000_000,
	}
}

func TestDeriveSeatParkState_EmptyLedger(t *testing.T) {
	if parks, last := deriveSeatParkState(nil); parks != 0 || last != 0 {
		t.Fatalf("empty ledger => (0,0), got (%d,%d)", parks, last)
	}
}

func TestDeriveSeatParkState_CountsConsecutiveNoSeat(t *testing.T) {
	events := []loopmgr.Event{
		seatParkEvent(seatParkReasonNoSeat, 1000),
		seatParkEvent(seatParkReasonNoSeat, 1100),
		seatParkEvent(seatParkReasonNoSeat, 1200), // newest
	}
	parks, last := deriveSeatParkState(events)
	if parks != 3 {
		t.Fatalf("parks = %d, want 3", parks)
	}
	if last != 1200 {
		t.Fatalf("lastParkUnix = %d, want 1200 (most recent)", last)
	}
}

func TestDeriveSeatParkState_ParkedRunsAreNeutral(t *testing.T) {
	// A deferred (parked) run neither counts nor ends the tail — it is the consequence
	// of parking, not a new no-seat failure.
	events := []loopmgr.Event{
		seatParkEvent(seatParkReasonNoSeat, 1000),
		seatParkEvent(string(seatpark.StatusParked), 1050),
		seatParkEvent(seatParkReasonNoSeat, 1100),
		seatParkEvent(string(seatpark.StatusParked), 1150), // newest, neutral
	}
	parks, last := deriveSeatParkState(events)
	if parks != 2 {
		t.Fatalf("parks = %d, want 2 (parked runs neutral)", parks)
	}
	if last != 1100 {
		t.Fatalf("lastParkUnix = %d, want 1100 (most recent NoSeat, not the parked marker)", last)
	}
}

func TestDeriveSeatParkState_ProgressRunEndsTail(t *testing.T) {
	// A prior progress run (Reason "") is a tail boundary: only the runs AFTER it count.
	events := []loopmgr.Event{
		seatParkEvent(seatParkReasonNoSeat, 900), // before the boundary — not counted
		seatParkEvent("", 1000),                  // progress boundary
		seatParkEvent(seatParkReasonNoSeat, 1100),
	}
	parks, last := deriveSeatParkState(events)
	if parks != 1 {
		t.Fatalf("parks = %d, want 1 (only the run after the boundary)", parks)
	}
	if last != 1100 {
		t.Fatalf("lastParkUnix = %d, want 1100", last)
	}
}

func TestDeriveSeatParkState_ExhaustedEndsTail(t *testing.T) {
	// Exhaustion resets the cycle: a fresh no-seat run after it starts the count over.
	events := []loopmgr.Event{
		seatParkEvent(seatParkReasonNoSeat, 900),
		seatParkEvent(string(seatpark.StatusExhausted), 1000), // boundary
		seatParkEvent(seatParkReasonNoSeat, 1100),
	}
	if parks, _ := deriveSeatParkState(events); parks != 1 {
		t.Fatalf("parks = %d, want 1 (exhaustion resets the cycle)", parks)
	}
}

func TestDeriveSeatParkState_IgnoresOtherLoopsAndKinds(t *testing.T) {
	events := []loopmgr.Event{
		{LoopID: "some-other-loop", Kind: loopmgr.EventEnd, Reason: seatParkReasonNoSeat, TSUnixNano: 1000 * 1_000_000_000},
		{LoopID: gardenDispatchLoopID, Kind: loopmgr.EventStart, Reason: seatParkReasonNoSeat, TSUnixNano: 1050 * 1_000_000_000},
		seatParkEvent(seatParkReasonNoSeat, 1100),
	}
	if parks, _ := deriveSeatParkState(events); parks != 1 {
		t.Fatalf("parks = %d, want 1 (other loops/kinds ignored)", parks)
	}
}

func TestDeriveSeatParkState_IntegratesWithSeatparkDecide(t *testing.T) {
	// One recent no-seat park, now inside the backoff window => Decide parks the next run;
	// once the 30s default window elapses it becomes ready to retry.
	events := []loopmgr.Event{seatParkEvent(seatParkReasonNoSeat, 1000)}
	parks, last := deriveSeatParkState(events)

	if d := seatpark.Decide(seatpark.Input{Parks: parks, LastParkUnix: last, NowUnix: 1010}); d.ShouldAttempt() {
		t.Fatalf("with a recent no-seat park the next run should PARK, got %q", d.Status)
	}
	if d := seatpark.Decide(seatpark.Input{Parks: parks, LastParkUnix: last, NowUnix: 1031}); !d.ShouldAttempt() {
		t.Fatalf("after the backoff window the run should retry, got %q", d.Status)
	}
}

func TestGardenDispatchSeatRefuses_KeysOnSeatWallsOnly(t *testing.T) {
	// Only genuine account-seat walls arm the park; a drained queue, a fault, or the
	// worker-slot cap (a worker-count wall, not an account-seat wall) must not.
	for _, v := range []string{"REFUSE_NO_ACCOUNT", "WEEKLY_CAPPED"} {
		if !gardenDispatchSeatRefuses[v] {
			t.Errorf("%q should arm the no-seat park", v)
		}
	}
	for _, v := range []string{"NO_LANE", "NO_ISSUE", "REFUSE_HOST_DIRTY", "SPAWN_FAILED", "REFUSE_AT_CAP"} {
		if gardenDispatchSeatRefuses[v] {
			t.Errorf("%q must NOT arm the no-seat park (not an account-seat wall)", v)
		}
	}
}

// TestGardenDispatchParksOnRecentNoSeatRefuse drives the whole bridge: with a just-now
// no-seat park in the durable ledger, a LIVE run must PARK (return before loading a
// candidate or probing preflight) rather than burst another spawn attempt — while a
// dry-run (inspection) is never parked.
func TestGardenDispatchParksOnRecentNoSeatRefuse(t *testing.T) {
	withDispatchJSONHelper(t, dispatchHappyHelper(t))
	gardenDispatchRouterFor(t)
	gardenDispatchSpawnerFor(t)
	root := t.TempDir()
	initDispatchGit(t, root)
	fixture := gardenDispatchIssuesFixture(t)
	ledger := filepath.Join(t.TempDir(), "loops.jsonl")

	// Permissive policy so the loop-governor gate (Gate 1) admits despite the recent
	// seeded event; the park gate (Gate 1.5) is the one under test.
	policyPath := filepath.Join(t.TempDir(), "policy.json")
	policy := loopmgr.Policies{Schema: loopmgr.SchemaPolicies, Loops: map[string]loopmgr.Policy{gardenDispatchLoopID: {}}}
	pb, _ := json.Marshal(policy)
	if err := os.WriteFile(policyPath, pb, 0o644); err != nil {
		t.Fatalf("write policy: %v", err)
	}

	// Seed a just-now no-seat park so the next LIVE run sits inside the backoff window.
	if _, err := loopmgr.Append(ledger, loopmgr.Event{
		LoopID: gardenDispatchLoopID, Kind: loopmgr.EventEnd,
		Status: loopmgr.StatusWitnessedDone, Reason: seatParkReasonNoSeat, Summary: "seed",
	}); err != nil {
		t.Fatalf("seed ledger: %v", err)
	}

	var stdout, stderr bytes.Buffer
	code := runGardenDispatch(&stdout, &stderr, []string{
		"--workspace", root, "--input", fixture, "--json",
		"--apply", "--ledger", ledger, "--policy", policyPath,
	})
	if code != 0 {
		t.Fatalf("live code=%d stderr=%s", code, stderr.String())
	}
	var got gardenDispatchResultJSON
	if err := json.Unmarshal(stdout.Bytes(), &got); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, stdout.String())
	}
	if got.Verdict != string(seatpark.StatusParked) {
		t.Fatalf("Verdict = %q, want %q (a recent no-seat refuse must park, not burst)", got.Verdict, seatpark.StatusParked)
	}
	if got.Spawned != 0 || got.Considered != 0 {
		t.Fatalf("a parked run must not load candidates or spawn: considered=%d spawned=%d", got.Considered, got.Spawned)
	}

	// A dry-run is never parked — inspection always runs.
	var dstdout, dstderr bytes.Buffer
	if dcode := runGardenDispatch(&dstdout, &dstderr, []string{
		"--workspace", root, "--input", fixture, "--json",
		"--ledger", ledger, "--policy", policyPath,
	}); dcode != 0 {
		t.Fatalf("dry-run code=%d stderr=%s", dcode, dstderr.String())
	}
	var dgot gardenDispatchResultJSON
	if err := json.Unmarshal(dstdout.Bytes(), &dgot); err != nil {
		t.Fatalf("dry-run unmarshal: %v\n%s", err, dstdout.String())
	}
	if dgot.Verdict == string(seatpark.StatusParked) {
		t.Fatalf("dry-run must never park (inspection always runs), got %q", dgot.Verdict)
	}
}
