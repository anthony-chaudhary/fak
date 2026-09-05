package assumecheck

import (
	"context"
	"os"
	"testing"
)

var (
	benchSinkVerdict    Verdict
	benchSinkErr        error
	benchSinkAssumption Assumption
	benchSinkDriver     Driver
	benchSinkDrivers    []Driver
	benchSinkTickResult TickResult
	benchSinkEvidence   Evidence
)

// TestBenchmarkAssumeCheckSanity verifies that the benchmarked code paths execute
// cleanly before running benchmarks.
func TestBenchmarkAssumeCheckSanity(t *testing.T) {
	v := Check(SeatLaunchable, Evidence{
		Kind:       WitnessLedgerRead,
		Witnessed:  true,
		Holds:      true,
		AgeSeconds: 1,
	})
	if v.Outcome != OutcomeHolds {
		t.Fatalf("sanity Check: got %s, want %s", v.Outcome, OutcomeHolds)
	}

	if _, err := GuardAssumption(SeatLaunchable, Evidence{
		Kind:      WitnessLedgerRead,
		Witnessed: true,
		Holds:     true,
	}); err != nil {
		t.Fatalf("sanity GuardAssumption: %v", err)
	}

	res := Tick(
		map[string]Outcome{"seat-launchable": OutcomeHolds},
		[]Verdict{tickVerdict("seat-launchable", OutcomeViolated, "depleted")},
	)
	if len(res.Events) != 1 {
		t.Fatalf("sanity Tick: got %d events, want 1", len(res.Events))
	}
}

// BenchmarkAssumeCheck measures the pure decision function across holding,
// violated, unverifiable, and stale evidence profiles.
func BenchmarkAssumeCheck(b *testing.B) {
	a := SeatLaunchable
	holdsEv := Evidence{
		Kind:          WitnessLedgerRead,
		Witnessed:     true,
		Holds:         true,
		AgeSeconds:    5,
		MaxAgeSeconds: 60,
		Detail:        "seat verified in rotation roster",
	}
	violatedEv := Evidence{
		Kind:      WitnessLedgerRead,
		Witnessed: true,
		Holds:     false,
		Detail:    "seat excluded from launch pool",
	}
	staleEv := Evidence{
		Kind:          WitnessLedgerRead,
		Witnessed:     true,
		Holds:         true,
		AgeSeconds:    120,
		MaxAgeSeconds: 60,
	}
	mismatchEv := Evidence{
		Kind:      WitnessCommandProbe,
		Witnessed: true,
		Holds:     true,
	}

	b.Run("Holds", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkVerdict = Check(a, holdsEv)
		}
	})

	b.Run("Violated", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkVerdict = Check(a, violatedEv)
		}
	})

	b.Run("Stale", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkVerdict = Check(a, staleEv)
		}
	})

	b.Run("UnverifiableMismatch", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkVerdict = Check(a, mismatchEv)
		}
	})
}

// BenchmarkGuardAssumption measures the hard gate under both holding (nil error)
// and violated (AssumptionViolationError wrapping ErrAssumptionViolated) conditions.
func BenchmarkGuardAssumption(b *testing.B) {
	a := SeatLaunchable
	holdsEv := Evidence{
		Kind:      WitnessLedgerRead,
		Witnessed: true,
		Holds:     true,
		Detail:    "seat ready",
	}
	violatedEv := Evidence{
		Kind:      WitnessLedgerRead,
		Witnessed: true,
		Holds:     false,
		Detail:    "seat excluded",
	}

	b.Run("Holds", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkVerdict, benchSinkErr = GuardAssumption(a, holdsEv)
		}
	})

	b.Run("Violated", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkVerdict, benchSinkErr = GuardAssumption(a, violatedEv)
		}
	})
}

// BenchmarkRegistry measures assumption registry lookup and enumeration.
func BenchmarkRegistry(b *testing.B) {
	b.Run("Lookup", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkAssumption, _ = Lookup("seat-launchable")
		}
	})

	b.Run("RegistryCopy", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = Registry()
		}
	})
}

// BenchmarkDrivers measures driver lookup, driver enumeration, and in-memory probe gathering.
func BenchmarkDrivers(b *testing.B) {
	b.Run("ResolveDriver", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkDriver, _ = ResolveDriver(WitnessConfigFlag)
		}
	})

	b.Run("DriversList", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkDrivers = Drivers()
		}
	})

	b.Run("ConfigFlagDriver_GatherStat", func(b *testing.B) {
		d := NewConfigFlagDriverWithStat(func(string) (os.FileInfo, error) {
			return nil, nil
		})
		ctx := context.Background()
		target := Target{Path: "/config/seat.json"}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkEvidence = d.Gather(ctx, target)
		}
	})

	b.Run("CommandProbeDriver_GatherInProcess", func(b *testing.B) {
		d := NewCommandProbeDriverWithRunner(nil)
		ctx := context.Background()
		target := Target{
			Probe: func(context.Context) (string, int, error) {
				return "pool ready", 0, nil
			},
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkEvidence = d.Gather(ctx, target)
		}
	})
}

// BenchmarkTick measures pure loop-tick re-witness classification for both calm
// (no transitions) and transitioning (HOLDS->VIOLATED edge emitting events) cycles.
func BenchmarkTick(b *testing.B) {
	prev := map[string]Outcome{
		"seat-launchable":         OutcomeHolds,
		"seat-config-dir-present": OutcomeHolds,
		"seat-pool-not-depleted":  OutcomeHolds,
		"kernel-loop-alive":       OutcomeHolds,
	}
	calmVerdicts := []Verdict{
		{AssumptionID: "seat-launchable", Level: LevelInfra, Witness: WitnessLedgerRead, Outcome: OutcomeHolds, Reason: "held"},
		{AssumptionID: "seat-config-dir-present", Level: LevelInfra, Witness: WitnessConfigFlag, Outcome: OutcomeHolds, Reason: "held"},
		{AssumptionID: "seat-pool-not-depleted", Level: LevelLoop, Witness: WitnessCommandProbe, Outcome: OutcomeHolds, Reason: "held"},
		{AssumptionID: "kernel-loop-alive", Level: LevelSubsystem, Witness: WitnessCommandProbe, Outcome: OutcomeHolds, Reason: "held"},
	}
	edgeVerdicts := []Verdict{
		{AssumptionID: "seat-launchable", Level: LevelInfra, Witness: WitnessLedgerRead, Outcome: OutcomeHolds, Reason: "held"},
		{AssumptionID: "seat-config-dir-present", Level: LevelInfra, Witness: WitnessConfigFlag, Outcome: OutcomeViolated, Reason: "seat dir vanished"},
		{AssumptionID: "seat-pool-not-depleted", Level: LevelLoop, Witness: WitnessCommandProbe, Outcome: OutcomeHolds, Reason: "held"},
		{AssumptionID: "kernel-loop-alive", Level: LevelSubsystem, Witness: WitnessCommandProbe, Outcome: OutcomeViolated, Reason: "kernel loop down"},
	}

	b.Run("CalmTick", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkTickResult = Tick(prev, calmVerdicts)
		}
	})

	b.Run("TransitionTick", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkTickResult = Tick(prev, edgeVerdicts)
		}
	})
}
