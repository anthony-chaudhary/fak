package ops

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessres"
)

type stubHostSampler struct {
	sample harnessres.HostSample
	err    error
}

func (s stubHostSampler) SampleHost() (harnessres.HostSample, error) {
	return s.sample, s.err
}

// TestPauseGate is the witness for #14477:
//  1. A busy host with reactive pausing enabled yields SKIP for deferrable
//     junkwork and RUN_PRESSURE_RELIEF_ONLY for pressure relief in one tick.
//  2. A busy host with reactive pausing disabled (the default) yields RUN.
//  3. UNKNOWN/ERROR host state yields RUN, never SKIP (fail-open for maintenance).
//  4. Every SKIP carries a closed-vocabulary reason_class and trigger value.
//  5. A SKIP streak cannot run forever: hitting the deadline/max-skips cap
//     honors the deferral handoff (#14479) and resumes.
//
// fak-test:runtime fast est=1s lane=default
func TestPauseGate(t *testing.T) {
	gate := DefaultPauseEvaluator{}
	baseTime := time.Date(2026, 11, 27, 10, 0, 0, 0, time.UTC)

	busyHost := &HostLoad{
		State:         HostSampleMeasured,
		TotalRAMBytes: 100 * 1024 * 1024 * 1024,
		AvailRAMBytes: 8 * 1024 * 1024 * 1024, // 92% used
		HaveRAM:       true,
	}

	t.Run("busy host with reactive pausing enabled gates only junkwork", func(t *testing.T) {
		cfg := DefaultConfig()
		cfg.ReactivePauseEnabled = true

		d := gate.Decide(LoadCycle{Started: baseTime, Host: busyHost}, PauseState{}, cfg)
		if got := d.ForPhase(PhaseDeferrable); got != ActionSkip {
			t.Fatalf("deferrable action = %q, want %q", got, ActionSkip)
		}
		if got := d.ForPhase(PhasePressureRelief); got != ActionRunPressureReliefOnly {
			t.Fatalf("pressure-relief action = %q, want %q", got, ActionRunPressureReliefOnly)
		}
		if d.ForPhase(PhaseDeferrable) == d.ForPhase(PhasePressureRelief) {
			t.Fatalf("expected two different decisions in one tick for junkwork vs relief")
		}
		if !ValidPauseReasonClass(d.ReasonClass) || d.ReasonClass != PauseReasonHostPressure {
			t.Fatalf("reason_class = %q, want closed-vocab %q", d.ReasonClass, PauseReasonHostPressure)
		}
		if !strings.Contains(d.TriggerValue, "0.92") {
			t.Fatalf("trigger_value = %q, want observed trigger ratio 0.92", d.TriggerValue)
		}
		if !d.ShouldPause {
			t.Fatalf("expected ShouldPause=true on busy host transition")
		}
	})

	t.Run("busy host with reactive pausing disabled yields RUN", func(t *testing.T) {
		cfg := DefaultConfig()
		if cfg.ReactivePauseEnabled {
			t.Fatalf("DefaultConfig().ReactivePauseEnabled must be false (opt-in)")
		}

		d := gate.Decide(LoadCycle{Started: baseTime, Host: busyHost}, PauseState{}, cfg)
		if got := d.ForPhase(PhaseDeferrable); got != ActionRun {
			t.Fatalf("deferrable action = %q, want %q when disabled", got, ActionRun)
		}
		if got := d.ForPhase(PhasePressureRelief); got != ActionRun {
			t.Fatalf("relief action = %q, want %q when disabled", got, ActionRun)
		}
		if d.ShouldPause {
			t.Fatalf("expected ShouldPause=false when reactive pausing is disabled")
		}
	})

	t.Run("UNKNOWN and ERROR host states yield RUN even when enabled", func(t *testing.T) {
		cfg := DefaultConfig()
		cfg.ReactivePauseEnabled = true

		for _, host := range []*HostLoad{
			{State: HostSampleUnknown},
			{State: HostSampleError, Error: "sampler read failed"},
			nil,
		} {
			d := gate.Decide(LoadCycle{Started: baseTime, Host: host}, PauseState{}, cfg)
			if got := d.ForPhase(PhaseDeferrable); got != ActionRun {
				t.Fatalf("host=%v deferrable action = %q, want %q", host, got, ActionRun)
			}
			if got := d.ForPhase(PhasePressureRelief); got != ActionRun {
				t.Fatalf("host=%v relief action = %q, want %q", host, got, ActionRun)
			}
			if d.ShouldPause {
				t.Fatalf("host=%v must never trigger ShouldPause", host)
			}
		}
	})

	t.Run("SKIP streak cannot run forever (deadline handoff)", func(t *testing.T) {
		cfg := DefaultConfig()
		cfg.ReactivePauseEnabled = true
		cfg.MaxConsecutiveSkips = 3
		cfg.MaxPauseDuration = 10 * time.Minute

		paused := PauseState{
			Schema:      PauseSchemaV1,
			Paused:      true,
			ReasonClass: PauseReasonHostPressure,
			PausedAt:    baseTime,
			PausedTicks: 3,
			ReactivateWhen: ReactivateCondition{
				PressureRatioBelow: 0.75,
				MaxPause:           "10m",
			},
		}
		d := gate.Decide(LoadCycle{Started: baseTime.Add(2 * time.Minute), Host: busyHost}, paused, cfg)
		if !d.DeadlineHit {
			t.Fatalf("expected DeadlineHit=true when PausedTicks reaches MaxConsecutiveSkips")
		}
		if got := d.ForPhase(PhaseDeferrable); got != ActionRun {
			t.Fatalf("deferrable action at deadline = %q, want %q", got, ActionRun)
		}
		if !d.ShouldResume {
			t.Fatalf("expected ShouldResume=true at deadline handoff")
		}
	})

	t.Run("Engine.Tick integrates gate decisions and records SKIPPED/PRESSURE_RELIEF_ONLY", func(t *testing.T) {
		repo := setupTestAssetRepo(t)
		engine := newHermeticTestEngine(t, repo)
		engine.Config.ReactivePauseEnabled = true
		engine.Config.MaxConsecutiveSkips = 2
		engine.SetSelfCostFunc(func() (time.Duration, bool) { return 10 * time.Millisecond, true })
		clock, advance := fixedClock(baseTime)
		engine.SetClock(clock)
		engine.SetHostSampler(stubHostSampler{
			sample: harnessres.HostSample{
				TotalRAMBytes: 100 * 1024 * 1024 * 1024,
				AvailRAMBytes: 8 * 1024 * 1024 * 1024,
				HaveRAM:       true,
			},
		})

		ctx := context.Background()
		if err := engine.Tick(ctx, true); err != nil {
			t.Fatalf("tick 1: %v", err)
		}
		if !engine.Paused() {
			t.Fatalf("expected engine to enter durable pause on busy host")
		}

		cycles, err := engine.LoadLedger.Read()
		if err != nil {
			t.Fatalf("read load ledger: %v", err)
		}
		if len(cycles) != 1 {
			t.Fatalf("expected 1 cycle, got %d", len(cycles))
		}
		c := cycles[0]
		if c.Verdict != LoadPressureReliefOnly {
			t.Errorf("tick 1 verdict = %q, want %q", c.Verdict, LoadPressureReliefOnly)
		}
		if c.ReasonClass != PauseReasonHostPressure {
			t.Errorf("tick 1 reason_class = %q, want %q", c.ReasonClass, PauseReasonHostPressure)
		}
		if !strings.Contains(c.TriggerValue, "0.92") {
			t.Errorf("tick 1 trigger_value = %q, want 0.92", c.TriggerValue)
		}

		phaseByName := map[string]PhaseLoad{}
		for _, p := range c.Phases {
			phaseByName[p.Phase] = p
		}
		if phaseByName["asset_sync"].Ran {
			t.Errorf("asset_sync (deferrable) must be skipped under host pressure")
		}
		if !phaseByName["asset_sync"].Deferrable {
			t.Errorf("asset_sync must be marked Deferrable=true")
		}
		if !phaseByName["locks_worktrees"].Ran || !phaseByName["process_runaways"].Ran || !phaseByName["storage_reclaim"].Ran {
			t.Errorf("pressure relief phases must all run under pressure: %+v", phaseByName)
		}

		// Unknown sampler clears host-pressure pause and runs normally.
		advance(1 * time.Minute)
		engine.SetHostSampler(stubHostSampler{err: errors.New("sensor offline")})
		if err := engine.Tick(ctx, true); err != nil {
			t.Fatalf("tick 2 (error host): %v", err)
		}
		if engine.Paused() {
			t.Fatalf("error/unknown host sample must fail open and clear host-pressure pause")
		}
	})
}
