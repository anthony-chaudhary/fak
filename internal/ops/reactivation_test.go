package ops

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/harnessres"
)

// TestReactivation is the witness for #14479:
//  1. A pause that never clears resumes at exactly paused_at + MaxPause and
//     records MAX_PAUSE_REACHED.
//  2. The number of probes attempted during a pause is logarithmic (not linear)
//     in the pause duration, and pausing reduces host cost vs full ticks.
//  3. An UNKNOWN sample neither resumes nor extends the pause.
//  4. A declared condition satisfied for the required consecutive-sample count
//     resumes with a reason distinct from MAX_PAUSE_REACHED.
//  5. A corrupted or absent pause record means "not paused" (no permanent wedge).
//
// fak-test:runtime fast est=1s lane=default
func TestReactivation(t *testing.T) {
	baseTime := time.Date(2026, 11, 27, 12, 0, 0, 0, time.UTC)

	busyHost := &HostLoad{
		State:         HostSampleMeasured,
		TotalRAMBytes: 100 * 1024 * 1024 * 1024,
		AvailRAMBytes: 10 * 1024 * 1024 * 1024, // 90% used
		HaveRAM:       true,
	}
	clearHost := &HostLoad{
		State:         HostSampleMeasured,
		TotalRAMBytes: 100 * 1024 * 1024 * 1024,
		AvailRAMBytes: 40 * 1024 * 1024 * 1024, // 60% used
		HaveRAM:       true,
	}
	unknownHost := &HostLoad{State: HostSampleUnknown}

	t.Run("1_forced_resume_at_exact_max_pause_records_MAX_PAUSE_REACHED", func(t *testing.T) {
		maxPause := 30 * time.Minute
		state := PauseState{
			Schema:      PauseSchemaV1,
			Paused:      true,
			ReasonClass: PauseReasonHostPressure,
			PausedAt:    baseTime,
			ReactivateWhen: ReactivateCondition{
				PressureRatioBelow: 0.75,
				ConsecutiveSamples: 3,
				MaxPause:           maxPause.String(),
			},
		}
		policy := DefaultReactivationPolicy()

		// 1ns before paused_at + MaxPause: must NOT resume.
		policy.Now = baseTime.Add(maxPause - time.Nanosecond)
		if resume, reason := ShouldResume(busyHost, state, policy); resume {
			t.Fatalf("resumed before MaxPause: reason=%q", reason)
		}

		// At exact paused_at + MaxPause: MUST resume with MAX_PAUSE_REACHED.
		policy.Now = baseTime.Add(maxPause)
		resume, reason := ShouldResume(busyHost, state, policy)
		if !resume {
			t.Fatalf("expected forced resume at exact paused_at + MaxPause")
		}
		if reason != PauseReasonMaxPauseReached {
			t.Fatalf("forced resume reason = %q, want %q", reason, PauseReasonMaxPauseReached)
		}
	})

	t.Run("2_logarithmic_probe_schedule_reduces_host_cost", func(t *testing.T) {
		state := PauseState{
			Schema:      PauseSchemaV1,
			Paused:      true,
			ReasonClass: PauseReasonHostPressure,
			PausedAt:    baseTime,
			ReactivateWhen: ReactivateCondition{
				PressureRatioBelow: 0.75,
				ConsecutiveSamples: 2,
				MaxPause:           "2h",
			},
		}
		policy := ReactivationPolicy{
			BaseProbeInterval: 1 * time.Minute,
			MaxProbeInterval:  1 * time.Hour,
			MaxPause:          2 * time.Hour,
		}

		probes := 0
		totalTicks := 60
		for m := 1; m <= totalTicks; m++ {
			now := baseTime.Add(time.Duration(m) * time.Minute)
			if ShouldProbe(now, state, policy) {
				probes++
				state, _, _ = ApplyProbeSample(state, now, busyHost, policy)
			}
		}

		// Over 60m with 1m base and 60m cap: probes at 1, 3, 7, 15, 31 -> 5 probes.
		// Logarithmic bound: <= 7, far below linear 60.
		if probes == 0 || probes > 7 {
			t.Fatalf("probes over 60m = %d, want logarithmic in (0, 7]", probes)
		}

		// Verify Engine.Tick skips host sampling on non-probe ticks while paused,
		// so probe calls are a small fraction of total ticks.
		repo := setupTestAssetRepo(t)
		engine := newHermeticTestEngine(t, repo)
		engine.Config.ReactivePauseEnabled = true
		engine.Config.MaxPauseDuration = 30 * time.Minute
		engine.Config.MaxConsecutiveSkips = 1000
		clock, advance := fixedClock(baseTime)
		engine.SetClock(clock)
		engine.SetSelfCostFunc(func() (time.Duration, bool) { return 1 * time.Millisecond, true })

		sampleCalls := 0
		engine.SetHostSampler(countingHostSampler{
			sample: harnessres.HostSample{
				TotalRAMBytes: 100 * 1024 * 1024 * 1024,
				AvailRAMBytes: 10 * 1024 * 1024 * 1024,
				HaveRAM:       true,
			},
			onSample: func() { sampleCalls++ },
		})

		ctx := context.Background()
		if err := engine.Tick(ctx, true); err != nil {
			t.Fatalf("initial tick: %v", err)
		}
		if !engine.Paused() {
			t.Fatalf("expected engine to pause on busy host")
		}
		sampleCalls = 0
		for i := 0; i < 29; i++ {
			advance(1 * time.Minute)
			if err := engine.Tick(ctx, false); err != nil {
				t.Fatalf("paused tick %d: %v", i, err)
			}
		}
		// Over 29 minutes with 1m base and 15m cap, probes occur at +1, +3, +7, +15 -> 4 probes.
		if sampleCalls >= 10 {
			t.Fatalf("paused host sample calls = %d over 29 ticks, expected bounded probe fraction (<10)", sampleCalls)
		}
	})

	t.Run("3_unknown_sample_neither_resumes_nor_extends", func(t *testing.T) {
		maxPause := 10 * time.Minute
		state := PauseState{
			Schema:      PauseSchemaV1,
			Paused:      true,
			ReasonClass: PauseReasonHostPressure,
			PausedAt:    baseTime,
			ReactivateWhen: ReactivateCondition{
				PressureRatioBelow: 0.75,
				ConsecutiveSamples: 2,
				MaxPause:           maxPause.String(),
			},
		}
		policy := DefaultReactivationPolicy()
		policy.Now = baseTime.Add(2 * time.Minute)

		// One clear sample, then an UNKNOWN sample.
		var resume bool
		state, resume, _ = ApplyProbeSample(state, policy.Now, clearHost, policy)
		if resume {
			t.Fatalf("must not resume on 1/2 clear samples")
		}
		if state.ConsecutiveClearSamples != 1 {
			t.Fatalf("consecutive_clear_samples = %d, want 1", state.ConsecutiveClearSamples)
		}

		policy.Now = baseTime.Add(4 * time.Minute)
		state, resume, reason := ApplyProbeSample(state, policy.Now, unknownHost, policy)
		if resume {
			t.Fatalf("UNKNOWN sample must never resume (got reason %q)", reason)
		}
		if state.ConsecutiveClearSamples != 0 {
			t.Fatalf("UNKNOWN sample must reset consecutive_clear_samples to 0, got %d", state.ConsecutiveClearSamples)
		}
		if !state.PausedAt.Equal(baseTime) {
			t.Fatalf("UNKNOWN sample must never extend paused_at: got %v, want %v", state.PausedAt, baseTime)
		}
		if state.ReactivateWhen.MaxPause != maxPause.String() {
			t.Fatalf("UNKNOWN sample must never extend max_pause: got %q, want %q", state.ReactivateWhen.MaxPause, maxPause.String())
		}

		// Still force-resumes at original paused_at + MaxPause even if sample is UNKNOWN.
		policy.Now = baseTime.Add(maxPause)
		resume, reason = ShouldResume(unknownHost, state, policy)
		if !resume || reason != PauseReasonMaxPauseReached {
			t.Fatalf("at MaxPause with UNKNOWN sample: resume=%v reason=%q, want true / %q", resume, reason, PauseReasonMaxPauseReached)
		}
	})

	t.Run("4_declared_condition_satisfied_for_consecutive_samples_resumes", func(t *testing.T) {
		state := PauseState{
			Schema:      PauseSchemaV1,
			Paused:      true,
			ReasonClass: PauseReasonHostPressure,
			PausedAt:    baseTime,
			ReactivateWhen: ReactivateCondition{
				PressureRatioBelow: 0.75,
				ConsecutiveSamples: 3,
				MaxPause:           "30m",
			},
		}
		policy := DefaultReactivationPolicy()

		for i := 1; i <= 2; i++ {
			now := baseTime.Add(time.Duration(i) * time.Minute)
			policy.Now = now
			var resume bool
			var reason PauseReasonClass
			state, resume, reason = ApplyProbeSample(state, now, clearHost, policy)
			if resume {
				t.Fatalf("sample %d/3 resumed early with %q", i, reason)
			}
		}

		policy.Now = baseTime.Add(3 * time.Minute)
		resume, reason := ShouldResume(clearHost, state, policy)
		if !resume {
			t.Fatalf("expected resume on 3/3 consecutive clear samples")
		}
		if reason == "" || reason == PauseReasonMaxPauseReached {
			t.Fatalf("natural resume reason = %q, must be non-empty and distinct from %q", reason, PauseReasonMaxPauseReached)
		}
	})

	t.Run("5_corrupted_or_absent_pause_record_means_not_paused", func(t *testing.T) {
		dir := t.TempDir()
		path := DefaultPauseStatePath(dir)
		store, err := OpenPauseStore(path)
		if err != nil {
			t.Fatalf("OpenPauseStore: %v", err)
		}

		// Absent record -> not paused.
		absent := store.ReadFailSafe()
		if absent.Paused {
			t.Fatalf("absent pause record must mean not paused")
		}
		if ShouldProbe(baseTime, absent, DefaultReactivationPolicy()) {
			t.Fatalf("absent pause record must not probe")
		}
		if resume, _ := ShouldResume(busyHost, absent, DefaultReactivationPolicy()); resume {
			t.Fatalf("absent pause record must not report ShouldResume=true")
		}

		// Corrupted record -> not paused.
		if err := os.WriteFile(path, []byte("{corrupt-json"), 0o644); err != nil {
			t.Fatalf("write corrupt pause file: %v", err)
		}
		corrupt := store.ReadFailSafe()
		if corrupt.Paused {
			t.Fatalf("corrupt pause record must fail safe to not paused")
		}
	})
}

type countingHostSampler struct {
	sample   harnessres.HostSample
	err      error
	onSample func()
}

func (c countingHostSampler) SampleHost() (harnessres.HostSample, error) {
	if c.onSample != nil {
		c.onSample()
	}
	return c.sample, c.err
}
