package dormancysim

import (
	"context"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dormancy"
	"github.com/anthony-chaudhary/fak/internal/rehydrate"
)

// epoch is a fixed, wall-clock-free base instant for every simulation. Nothing in this
// package reads time.Now(), so the whole harness is deterministic from this constant.
var epoch = time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)

// recordingGate builds a Gate of all five canonical rungs; each records that its Check
// actually ran (by appending its reason to *ran) and clears. It is the deterministic rung
// double the rehydrate spine is exercised through — the same shape rehydrate's own
// fullLadder uses, restated here because those rungs are unexported test doubles.
func recordingGate(ran *[]rehydrate.Reason) *rehydrate.Gate {
	mk := func(reason rehydrate.Reason) rehydrate.Rung {
		return rehydrate.NewRung(reason, func(context.Context) rehydrate.Verdict {
			*ran = append(*ran, reason)
			return rehydrate.Clear()
		})
	}
	return rehydrate.NewGate(
		mk(rehydrate.ColdCache),
		mk(rehydrate.StaleCred),
		mk(rehydrate.StaleRecall),
		mk(rehydrate.StaleLease),
		mk(rehydrate.StalePlan),
	)
}

func eq(a, b []rehydrate.Reason) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestTimeTravelTransitions is the core acceptance: a single injected clock, fast-forwarded
// across each horizon boundary, deterministically exercises the warm/cool/cold/frozen/ancient
// transitions and drives strictly-monotonic rehydration — a longer gap runs at least as many
// rungs as a shorter one, in ladder order, with no real waits and no wall-clock read.
func TestTimeTravelTransitions(t *testing.T) {
	// The ordered rung ladder, band by band (rehydrate's canonical staging: COLD_CACHE at
	// cool; +STALE_CRED,STALE_RECALL at cold; +STALE_LEASE,STALE_PLAN at frozen; ancient
	// runs the same five as frozen — the closed vocabulary stops at five today).
	cases := []struct {
		name  string
		gap   time.Duration
		band  dormancy.Horizon
		fired []rehydrate.Reason
	}{
		{"warm", 3 * time.Minute, dormancy.Warm, nil},
		{"cool", 30 * time.Minute, dormancy.Cool, []rehydrate.Reason{rehydrate.ColdCache}},
		{"cold", 6 * time.Hour, dormancy.Cold, []rehydrate.Reason{
			rehydrate.ColdCache, rehydrate.StaleCred, rehydrate.StaleRecall}},
		{"frozen", 10 * 24 * time.Hour, dormancy.Frozen, []rehydrate.Reason{
			rehydrate.ColdCache, rehydrate.StaleCred, rehydrate.StaleRecall,
			rehydrate.StaleLease, rehydrate.StalePlan}},
		{"ancient", 90 * 24 * time.Hour, dormancy.Ancient, []rehydrate.Reason{
			rehydrate.ColdCache, rehydrate.StaleCred, rehydrate.StaleRecall,
			rehydrate.StaleLease, rehydrate.StalePlan}},
	}

	prevCount := -1
	for _, tc := range cases {
		var ran []rehydrate.Reason
		sim := New(epoch, dormancy.At(epoch), recordingGate(&ran))
		adm := sim.Advance(context.Background(), tc.gap)

		if sim.Horizon() != tc.band {
			t.Errorf("%s: horizon=%v want %v", tc.name, sim.Horizon(), tc.band)
		}
		if adm.Horizon != tc.band {
			t.Errorf("%s: admission horizon=%v want %v", tc.name, adm.Horizon, tc.band)
		}
		if !adm.Admitted {
			t.Errorf("%s: not admitted (all rungs clear): %+v", tc.name, adm)
		}
		// The admission's ordered witness and the rungs' own side effects must agree, in order.
		if !eq(adm.RanReasons(), tc.fired) {
			t.Errorf("%s: admission ran=%v want %v", tc.name, adm.RanReasons(), tc.fired)
		}
		if !eq(ran, tc.fired) {
			t.Errorf("%s: rung side-effects=%v want %v (checks must actually run, in order)", tc.name, ran, tc.fired)
		}
		// Monotonic in the gap: a colder band never runs FEWER rungs than a warmer one.
		if len(tc.fired) < prevCount {
			t.Errorf("%s: rung count %d dropped below previous %d — staging must be monotonic in the gap",
				tc.name, len(tc.fired), prevCount)
		}
		prevCount = len(tc.fired)
	}
}

// TestNinetyDayDormancyFullRungSet is the headline acceptance sentence: a SINGLE test
// simulates a 90-day dormancy and asserts the FULL rehydration rung set (cred / lease /
// recall / cache / plan) fired, in ladder order, deterministically and with no real wait.
func TestNinetyDayDormancyFullRungSet(t *testing.T) {
	var ran []rehydrate.Reason
	sim := New(epoch, dormancy.At(epoch), recordingGate(&ran))

	adm := sim.Advance(context.Background(), 90*24*time.Hour)

	if sim.Horizon() != dormancy.Ancient {
		t.Fatalf("90-day gap horizon=%v want ancient", sim.Horizon())
	}
	if !adm.Admitted {
		t.Fatalf("90-day wake not admitted: %+v", adm)
	}
	want := []rehydrate.Reason{
		rehydrate.ColdCache,   // cache: prompt cache aged out (fires from cool)
		rehydrate.StaleCred,   // cred: token may have expired (fires from cold)
		rehydrate.StaleRecall, // recall: memory artifact may be invalid (fires from cold)
		rehydrate.StaleLease,  // lease: fence + reacquire (fires from frozen)
		rehydrate.StalePlan,   // plan: trunk moved on (fires from frozen)
	}
	if !eq(adm.RanReasons(), want) {
		t.Fatalf("90-day ran=%v want the full ordered rung set %v", adm.RanReasons(), want)
	}
	if !eq(ran, want) {
		t.Fatalf("90-day rung side-effects=%v want %v (every rung's Check must actually fire, in order)", ran, want)
	}
	// The simulated instant advanced exactly 90 days with no real time elapsed.
	if got := sim.Now().Sub(epoch); got != 90*24*time.Hour {
		t.Fatalf("simulated elapsed=%v want 90 days", got)
	}
}

// TestReArmSurvivesSimulatedRestart witnesses "durable wake / re-arm across restart" at the
// durable-stamp level: the LastActiveAt Stamp is unix-nanos, so it survives a process death
// (here: reconstructing the Simulator from the persisted stamp). After the "restart" the
// horizon still reflects the ORIGINAL last-active instant — dormancy is not reset to warm by
// a restart, which is exactly the durability the wake timers (#1188) rely on.
func TestReArmSurvivesSimulatedRestart(t *testing.T) {
	// Session goes dormant at epoch; 40 days pass; the process dies and restarts, then wakes.
	original := dormancy.At(epoch)

	// Persist across the "restart": a Stamp is plain unix-nanos, so round-trip it as one.
	persisted := dormancy.FromUnixNano(original.LastActiveUnixNano)
	if persisted != original {
		t.Fatalf("stamp did not survive persistence: %+v vs %+v", persisted, original)
	}

	// Reconstruct the harness from the persisted stamp, with the clock now 40 days later.
	wakeAt := epoch.Add(40 * 24 * time.Hour)
	var ran []rehydrate.Reason
	sim := New(wakeAt, persisted, recordingGate(&ran))

	adm := sim.AdmitNow(context.Background()) // wake immediately after restart, no further advance
	if sim.Horizon() != dormancy.Ancient {
		t.Fatalf("post-restart horizon=%v want ancient (40-day gap must survive the restart)", sim.Horizon())
	}
	if len(adm.RanReasons()) != 5 {
		t.Fatalf("post-restart ran %d rungs want 5 (restart must not reset dormancy to warm): %v",
			len(adm.RanReasons()), adm.RanReasons())
	}
}

// TestDeterministicAcrossRuns pins the no-wall-clock property: two independently constructed
// simulators advanced by the same schedule produce byte-identical admissions. If any organ on
// the exercised path read time.Now(), this would flake.
func TestDeterministicAcrossRuns(t *testing.T) {
	run := func() []rehydrate.Reason {
		var ran []rehydrate.Reason
		sim := New(epoch, dormancy.At(epoch), recordingGate(&ran))
		sim.Advance(context.Background(), 2*time.Hour)     // -> cold
		sim.Advance(context.Background(), 40*24*time.Hour) // -> ancient
		return ran
	}
	a, b := run(), run()
	if !eq(a, b) {
		t.Fatalf("non-deterministic across runs: %v vs %v", a, b)
	}
}

// TestClockNeverRunsBackward pins the monotonic clock: a non-positive Advance is ignored, so
// a backwards step cannot spuriously shrink the simulated gap.
func TestClockNeverRunsBackward(t *testing.T) {
	c := NewClock(epoch)
	c.Advance(time.Hour)
	if got := c.Advance(-time.Hour); got != epoch.Add(time.Hour) {
		t.Fatalf("negative advance moved the clock: now=%v want %v", got, epoch.Add(time.Hour))
	}
}
