package dispatchsweep

import (
	"errors"
	"testing"
)

func cfg(maxAgents int, live bool) Config {
	return Config{MaxAgents: maxAgents, MaxWorkers: 2, Backend: "claude", Live: live}
}

func spawnedTick(issue int) TickResult {
	return TickResult{Action: "spawned", Verdict: "SPAWNED", OK: true, Issue: issue,
		Lane: "docs", Account: "acct-a", PreflightVerdict: "SPAWN_OK"}
}

func refuseTick(verdict string) TickResult {
	return TickResult{Action: "refused", Verdict: verdict, OK: false, Lane: "docs",
		PreflightVerdict: verdict}
}

// seqTick returns a TickFunc that yields the given results in order, then panics if over-drawn
// (a test that draws more ticks than it staged is a bug we want loud, not silent).
func seqTick(results ...TickResult) TickFunc {
	return func(i int) (TickResult, error) {
		if i >= len(results) {
			panic("seqTick over-drawn: loop did not stop when expected")
		}
		return results[i], nil
	}
}

func TestDryRunPlansExactlyOneTick(t *testing.T) {
	calls := 0
	tick := func(i int) (TickResult, error) {
		calls++
		return TickResult{Action: "would_spawn", Verdict: "WOULD_SPAWN", OK: true,
			Issue: 1310, Lane: "docs", PreflightVerdict: "SPAWN_OK"}, nil
	}
	settles := 0
	rec := RunSweep(cfg(100, false), tick, func() { settles++ })

	if calls != 1 {
		t.Fatalf("dry-run must run exactly ONE tick, ran %d", calls)
	}
	if settles != 0 {
		t.Errorf("dry-run must never settle, settled %d", settles)
	}
	if rec.Mode != "dry-run" || rec.StopVerdict != "DRY_RUN" {
		t.Errorf("mode/verdict = %q/%q, want dry-run/DRY_RUN", rec.Mode, rec.StopVerdict)
	}
	if rec.SpawnedCount != 1 || len(rec.SpawnedIssues) != 1 || rec.SpawnedIssues[0] != 1310 {
		t.Errorf("spawned = %d %v, want 1 [1310]", rec.SpawnedCount, rec.SpawnedIssues)
	}
	if !rec.OK {
		t.Errorf("a clean dry-run plan must be ok")
	}
}

func TestDrainsUntilRefuseAtCap(t *testing.T) {
	tick := seqTick(spawnedTick(10), spawnedTick(11), refuseTick("REFUSE_AT_CAP"))
	settles := 0
	rec := RunSweep(cfg(100, true), tick, func() { settles++ })

	if rec.SpawnedCount != 2 {
		t.Fatalf("spawned = %d, want 2", rec.SpawnedCount)
	}
	if got := rec.SpawnedIssues; len(got) != 2 || got[0] != 10 || got[1] != 11 {
		t.Errorf("spawned issues = %v, want [10 11]", got)
	}
	if rec.StopVerdict != "REFUSE_AT_CAP" {
		t.Errorf("stop verdict = %q, want REFUSE_AT_CAP", rec.StopVerdict)
	}
	if settles != 2 {
		t.Errorf("settle calls = %d, want 2 (one after each live spawn)", settles)
	}
	if !rec.OK {
		t.Errorf("a clean cap-stop must be ok")
	}
}

func TestStopsOnQueueDrainedNoLane(t *testing.T) {
	tick := seqTick(spawnedTick(10), refuseTick("NO_LANE"))
	rec := RunSweep(cfg(100, true), tick, func() {})
	if rec.SpawnedCount != 1 || rec.StopVerdict != "NO_LANE" {
		t.Fatalf("got %d / %q, want 1 / NO_LANE", rec.SpawnedCount, rec.StopVerdict)
	}
	if rec.StopReason == "" || rec.StopReason[:6] != "queue " {
		t.Errorf("NO_LANE reason should describe a drained queue, got %q", rec.StopReason)
	}
	if !rec.OK {
		t.Errorf("a drained queue is an expected boundary, must be ok")
	}
}

func TestMaxAgentsCeilingHonored(t *testing.T) {
	// An endless supply of spawnable issues — only the ceiling can stop it.
	n := 0
	tick := func(i int) (TickResult, error) {
		n++
		return spawnedTick(n), nil
	}
	rec := RunSweep(cfg(3, true), tick, func() {})
	if rec.SpawnedCount != 3 || rec.StopVerdict != "MAX_AGENTS" {
		t.Fatalf("got %d / %q, want 3 / MAX_AGENTS", rec.SpawnedCount, rec.StopVerdict)
	}
	if !rec.OK {
		t.Errorf("hitting the best-effort ceiling is expected, must be ok")
	}
}

func TestSettleBetweenLiveSpawnsOnly(t *testing.T) {
	// 2 spawns then a refuse: settle fires after each spawn, never after the refuse.
	tick := seqTick(spawnedTick(10), spawnedTick(11), refuseTick("NO_ISSUE"))
	settles := 0
	RunSweep(cfg(100, true), tick, func() { settles++ })
	if settles != 2 {
		t.Errorf("settle calls = %d, want 2", settles)
	}
}

func TestTickErrorIsNotOK(t *testing.T) {
	tick := func(i int) (TickResult, error) {
		return TickResult{}, errors.New("python tick blew up")
	}
	rec := RunSweep(cfg(100, true), tick, func() {})
	if rec.OK {
		t.Errorf("a tick error is a tooling fault, must NOT be ok")
	}
	if rec.StopVerdict != "TICK_ERROR" || rec.SpawnedCount != 0 {
		t.Errorf("got %q / %d, want TICK_ERROR / 0", rec.StopVerdict, rec.SpawnedCount)
	}
}

func TestWeeklyCappedStopsAndIsOK(t *testing.T) {
	rec := RunSweep(cfg(100, true), seqTick(refuseTick("WEEKLY_CAPPED")), func() {})
	if rec.SpawnedCount != 0 || rec.StopVerdict != "WEEKLY_CAPPED" {
		t.Fatalf("got %d / %q, want 0 / WEEKLY_CAPPED", rec.SpawnedCount, rec.StopVerdict)
	}
	if !rec.OK {
		t.Errorf("a weekly-cap refusal is an expected boundary, must be ok")
	}
}

func TestUnknownVerdictFallsThroughNotSwallowed(t *testing.T) {
	rec := RunSweep(cfg(100, true), seqTick(refuseTick("BRAND_NEW_VERDICT")), func() {})
	if rec.StopVerdict != "BRAND_NEW_VERDICT" {
		t.Fatalf("stop verdict = %q, want BRAND_NEW_VERDICT", rec.StopVerdict)
	}
	if rec.StopReason != "tick refused: BRAND_NEW_VERDICT" {
		t.Errorf("unknown verdict reason = %q", rec.StopReason)
	}
	// An unrecognized refusal is a boundary (the tick declined), not a tooling fault: ok stays true.
	if !rec.OK {
		t.Errorf("an unknown refusal verdict should remain ok (boundary, not fault)")
	}
}

func TestSpawnedIssuesNeverNil(t *testing.T) {
	rec := RunSweep(cfg(100, true), seqTick(refuseTick("NO_LANE")), func() {})
	if rec.SpawnedIssues == nil {
		t.Errorf("SpawnedIssues must be a non-nil empty slice for stable JSON")
	}
}
