package main

// Regression guard for the COLD concurrent same-read fill-race probe (#878).
//
// These tests pin the cold probe's INVARIANTS — the race accounting, the verdict
// mapping, and the schema — NOT a specific verdict. Per the issue's honesty bounds,
// the current kernel has no singleflight on the cold-miss path, so MEASURED_RACE is
// the expected (acceptable) verdict; we do not require a one-engine-call invariant.
// When singleflight is built later, the verdict will flip to SINGLEFLIGHT_CONFIRMED
// and this same test keeps proving the accounting stays honest.

import (
	"context"
	"testing"
	"time"
)

// TestColdProofRaceAccountingIsHonest runs a small cold probe and checks every
// derived field against its definition: a never-seen key must execute the engine at
// least once, cold_fill_races is engine_calls_per_key-1 clamped at zero, and the
// rolled-up verdict matches the worst-case engine-calls across all trials.
func TestColdProofRaceAccountingIsHonest(t *testing.T) {
	const workers, trials = 8, 4
	proof, err := buildColdProof(context.Background(), workers, trials, time.Millisecond)
	if err != nil {
		t.Fatal(err)
	}
	if proof.Schema != "fak.tokendemo.parallel-cold.v1" {
		t.Fatalf("schema = %q, want fak.tokendemo.parallel-cold.v1", proof.Schema)
	}
	if proof.Workers != workers || proof.Trials != trials {
		t.Fatalf("workers/trials = %d/%d, want %d/%d", proof.Workers, proof.Trials, workers, trials)
	}
	if len(proof.PerTrial) != trials {
		t.Fatalf("per-trial rows = %d, want %d", len(proof.PerTrial), trials)
	}

	var sumRaces, sumEngine int64
	racedTrials := 0
	for _, tr := range proof.PerTrial {
		// A never-seen key has nothing cached: at least one worker must hit the engine.
		if tr.EngineCalls < 1 {
			t.Fatalf("trial %d ran the engine %d times for a cold key, want >= 1", tr.Trial, tr.EngineCalls)
		}
		// cold_fill_races is defined as engine_calls_per_key - 1, clamped at zero.
		wantRaces := tr.EngineCalls - 1
		if wantRaces < 0 {
			wantRaces = 0
		}
		if tr.ColdFillRaces != wantRaces {
			t.Fatalf("trial %d cold_fill_races = %d, want %d (engine_calls %d - 1, clamped)",
				tr.Trial, tr.ColdFillRaces, wantRaces, tr.EngineCalls)
		}
		// vDSO hits can never exceed the workers that did not run the engine.
		if tr.VDSOHits < 0 || tr.VDSOHits > int64(workers) {
			t.Fatalf("trial %d vdso_hits = %d, out of range [0,%d]", tr.Trial, tr.VDSOHits, workers)
		}
		if tr.ColdFillRaces > 0 {
			racedTrials++
		}
		sumRaces += tr.ColdFillRaces
		sumEngine += tr.EngineCalls
	}

	if proof.TotalColdFillRaces != sumRaces {
		t.Fatalf("total_cold_fill_races = %d, want %d (sum of per-trial)", proof.TotalColdFillRaces, sumRaces)
	}
	if proof.TotalEngineCalls != sumEngine {
		t.Fatalf("total_engine_calls = %d, want %d (sum of per-trial)", proof.TotalEngineCalls, sumEngine)
	}
	if proof.TrialsWithRace != racedTrials {
		t.Fatalf("trials_with_race = %d, want %d", proof.TrialsWithRace, racedTrials)
	}
	if proof.MinEngineCallsPerKey < 1 || proof.MaxEngineCallsPerKey < proof.MinEngineCallsPerKey {
		t.Fatalf("engine-calls min/max = %d/%d, inconsistent", proof.MinEngineCallsPerKey, proof.MaxEngineCallsPerKey)
	}

	// The rolled-up verdict must agree with the verdict mapping over the worst case.
	if got, want := proof.Verdict, coldVerdict(proof.MaxEngineCallsPerKey); got != want {
		t.Fatalf("verdict = %q, want %q for max engine-calls %d", got, want, proof.MaxEngineCallsPerKey)
	}
	switch proof.Verdict {
	case coldVerdictRace:
		if proof.MaxEngineCallsPerKey <= 1 {
			t.Fatalf("MEASURED_RACE but max engine-calls is %d (<=1)", proof.MaxEngineCallsPerKey)
		}
	case coldVerdictSingleflight:
		if proof.MaxEngineCallsPerKey > 1 {
			t.Fatalf("SINGLEFLIGHT_CONFIRMED but max engine-calls is %d (>1)", proof.MaxEngineCallsPerKey)
		}
	default:
		t.Fatalf("verdict %q is outside the closed vocabulary", proof.Verdict)
	}

	// Timing fields are populated (a barrier-released arm always measures wall time).
	if proof.RawWallTotalNs < 0 || proof.FakWallTotalNs < 0 {
		t.Fatalf("negative wall time: raw %d fak %d", proof.RawWallTotalNs, proof.FakWallTotalNs)
	}
	if proof.RawP95Ns < proof.RawP50Ns || proof.FakP95Ns < proof.FakP50Ns {
		t.Fatalf("p95 below p50: raw %d/%d fak %d/%d", proof.RawP50Ns, proof.RawP95Ns, proof.FakP50Ns, proof.FakP95Ns)
	}
}

// TestColdVerdictMapping pins the closed verdict vocabulary: SINGLEFLIGHT_CONFIRMED is
// claimed only when no cold key ever fanned out (max <= 1); anything above is a race.
func TestColdVerdictMapping(t *testing.T) {
	for _, tc := range []struct {
		max  int64
		want string
	}{
		{0, coldVerdictSingleflight},
		{1, coldVerdictSingleflight},
		{2, coldVerdictRace},
		{64, coldVerdictRace},
	} {
		if got := coldVerdict(tc.max); got != tc.want {
			t.Fatalf("coldVerdict(%d) = %q, want %q", tc.max, got, tc.want)
		}
	}
}

// TestColdProofRejectsNonPositiveConfig guards the two argument floors.
func TestColdProofRejectsNonPositiveConfig(t *testing.T) {
	if _, err := buildColdProof(context.Background(), 0, 4, time.Millisecond); err == nil {
		t.Fatal("workers=0 should error")
	}
	if _, err := buildColdProof(context.Background(), 8, 0, time.Millisecond); err == nil {
		t.Fatal("trials=0 should error")
	}
}
