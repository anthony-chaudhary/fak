package amdgpu

import (
	"fmt"
	"math"
	"strings"
	"testing"
	"time"
)

func TestStrixPowerFrontierReceipt(t *testing.T) {
	validReceipt := func() *StrixPowerFrontierReceipt {
		r := &StrixPowerFrontierReceipt{
			Schema:    StrixPowerFrontierSchema,
			Simulated: true,
			Identity: StrixPowerFrontierIdentity{
				Engine:                "fak-native",
				Backend:               "vulkan",
				Device:                DefaultStrixHaloArch,
				Concurrency:           1,
				SourceCommit:          "471e52ec8",
				ExecutableSHA256:      strings.Repeat("a", 64),
				BuildID:               "test-build",
				DriverID:              "test-radv",
				PowerController:       "test-power1-cap",
				ModelArtifactSHA256:   StrixPowerFrontierModelSHA256,
				TensorInventorySHA256: strings.Repeat("b", 64),
				PromptSHA256:          StrixPowerFrontierPromptSHA256,
			},
			PromptTokens:            StrixPowerFrontierPromptTokens,
			AcceptedTokensPerTrial:  StrixPowerFrontierAcceptedTokens,
			PredeclaredPPTWatts:     []int{45, 65, 85},
			Repetitions:             StrixPowerFrontierMinRepetitions,
			DeclaredMaxCVPercent:    5,
			PriorPowerSetting:       "65000000\n",
			RestoredPowerSetting:    "65000000\n",
			RestorationAttempted:    true,
			RestoreReadbackVerified: true,
		}
		sequence := 0
		for repetition := 0; repetition < r.Repetitions; repetition++ {
			for position := range r.PredeclaredPPTWatts {
				pointIndex := position
				if repetition%2 == 1 {
					pointIndex = len(r.PredeclaredPPTWatts) - 1 - position
				}
				ppt := r.PredeclaredPPTWatts[pointIndex]
				elapsed := int64(4*time.Second + time.Duration(ppt)*time.Millisecond)
				energyStart := uint64(1_000_000_000 + sequence*100_000_000)
				energyEnd := energyStart + uint64(40_000_000+ppt*100_000)
				r.Trials = append(r.Trials, StrixPowerFrontierTrial{
					PPTWatts:               ppt,
					Repetition:             repetition,
					Sequence:               sequence,
					AcceptedTokens:         StrixPowerFrontierAcceptedTokens,
					ElapsedNanoseconds:     elapsed,
					EnergyStartMicrojoules: energyStart,
					EnergyEndMicrojoules:   energyEnd,
					TokensPerSecond:        float64(StrixPowerFrontierAcceptedTokens) / (float64(elapsed) / 1e9),
					JoulesPerToken:         (float64(energyEnd-energyStart) / 1e6) / StrixPowerFrontierAcceptedTokens,
					GPUClockMHz:            2200 + float64(ppt),
					TemperatureC:           55 + float64(ppt)/10,
					Throttled:              ppt == 85,
					QualityPassed:          true,
					SettleMilliseconds:     1000,
					CooldownMilliseconds:   1000,
				})
				sequence++
			}
		}
		if err := r.Seal(); err != nil {
			t.Fatalf("Seal() error = %v", err)
		}
		return r
	}

	if err := validReceipt().Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}

	tests := []struct {
		name   string
		mutate func(*StrixPowerFrontierReceipt)
		want   string
	}{
		{"physical claim", func(r *StrixPowerFrontierReceipt) { r.Simulated = false }, ErrPhysicalExecutionUnwitnessed.Error()},
		{"selected default", func(r *StrixPowerFrontierReceipt) { r.SelectedPPTWatts = 65 }, ErrPhysicalExecutionUnwitnessed.Error()},
		{"incomplete repetitions", func(r *StrixPowerFrontierReceipt) { r.Repetitions = 4 }, "below minimum"},
		{"overflowing shape", func(r *StrixPowerFrontierReceipt) {
			r.PredeclaredPPTWatts = []int{1, 2, 3, 4}
			r.Repetitions = int(^uint(0)>>1)/2 + 1
			r.Trials = nil
		}, "trial count"},
		{"non-interleaved order", func(r *StrixPowerFrontierReceipt) { r.Trials[3].PPTWatts = 45 }, "PPT order"},
		{"non-finite telemetry", func(r *StrixPowerFrontierReceipt) { r.Trials[0].GPUClockMHz = math.NaN() }, "finite positive telemetry"},
		{"wrong accepted count", func(r *StrixPowerFrontierReceipt) { r.Trials[0].AcceptedTokens = 127 }, "accepted tokens"},
		{"foreign engine", func(r *StrixPowerFrontierReceipt) { r.Identity.Engine = "llama.cpp" }, "identity must be"},
		{"CPU backend", func(r *StrixPowerFrontierReceipt) { r.Identity.Backend = "cpu" }, "identity must be"},
		{"fallback", func(r *StrixPowerFrontierReceipt) { r.Trials[0].FallbackCount = 1 }, "fallback count"},
		{"quality failure", func(r *StrixPowerFrontierReceipt) { r.Trials[0].QualityPassed = false }, "quality gate"},
		{"authored rate", func(r *StrixPowerFrontierReceipt) { r.Trials[0].TokensPerSecond++ }, "throughput reconciliation"},
		{"loose variance gate", func(r *StrixPowerFrontierReceipt) { r.DeclaredMaxCVPercent = 5.1 }, "declared CV"},
		{"high measured variance", func(r *StrixPowerFrontierReceipt) {
			r.Trials[0].ElapsedNanoseconds *= 2
			r.Trials[0].TokensPerSecond = float64(StrixPowerFrontierAcceptedTokens) / (float64(r.Trials[0].ElapsedNanoseconds) / 1e9)
		}, "throughput CV"},
		{"restore mismatch", func(r *StrixPowerFrontierReceipt) { r.RestoredPowerSetting = "85000000\n" }, "restore read-back mismatch"},
		{"restore failed", func(r *StrixPowerFrontierReceipt) { r.RestoreError = "permission denied" }, "restoration failed"},
		{"tampered digest", func(r *StrixPowerFrontierReceipt) { r.Identity.DriverID = "other-radv" }, "digest mismatch"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := validReceipt()
			tt.mutate(r)
			if tt.name != "tampered digest" && tt.name != "physical claim" && tt.name != "selected default" && tt.name != "non-finite telemetry" {
				if err := r.Seal(); err != nil {
					t.Fatalf("Seal() error = %v", err)
				}
			}
			err := r.Validate()
			if err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Validate() error = %v, want containing %q", err, tt.want)
			}
		})
	}
}

func TestWorkloadGovernor_StateTransitions(t *testing.T) {
	var executedCommands []string
	cfg := DefaultWorkloadGovernorConfig()
	cfg.Executor = func(cmd string, args ...string) error {
		executedCommands = append(executedCommands, fmt.Sprintf("%s %v", cmd, args))
		return nil
	}

	gov := NewWorkloadGovernor(cfg)
	baseTime := time.Date(2026, 9, 6, 12, 0, 0, 0, time.UTC)

	// 1. Initial state must be Idle
	if gov.CurrentState() != StateIdle {
		t.Fatalf("initial state = %s, want %s", gov.CurrentState(), StateIdle)
	}

	// 2. Sub-threshold load (15%) -> stays Idle
	if err := gov.Tick(15.0, baseTime); err != nil {
		t.Fatalf("Tick failed: %v", err)
	}
	if gov.CurrentState() != StateIdle {
		t.Errorf("sub-threshold state = %s, want %s", gov.CurrentState(), StateIdle)
	}

	// 3. Active load (45%) for 1 second (< 2s threshold) -> stays Idle
	t1 := baseTime.Add(1 * time.Second)
	if err := gov.Tick(45.0, t1); err != nil {
		t.Fatalf("Tick failed: %v", err)
	}
	if gov.CurrentState() != StateIdle {
		t.Errorf("active 1s state = %s, want %s", gov.CurrentState(), StateIdle)
	}

	// 4. Active load (45%) reaches 2 consecutive seconds -> transitions to ActivePerformance
	t2 := t1.Add(2 * time.Second)
	if err := gov.Tick(45.0, t2); err != nil {
		t.Fatalf("Tick failed: %v", err)
	}
	if gov.CurrentState() != StateActivePerformance {
		t.Errorf("active 2s state = %s, want %s", gov.CurrentState(), StateActivePerformance)
	}
	if !gov.FanDutyLocked || gov.CurrentFanDuty != 100 {
		t.Errorf("FanDuty = %d (locked=%v), want 100 (locked=true)", gov.CurrentFanDuty, gov.FanDutyLocked)
	}
	if !gov.PowerProfileLocked || gov.CurrentPowerProfile != "accelerator-performance" {
		t.Errorf("PowerProfile = %s, want accelerator-performance", gov.CurrentPowerProfile)
	}

	// 5. Workload ends (util drops to 5%) -> transitions to CoolingDown
	t3 := baseTime.Add(10 * time.Second)
	if err := gov.Tick(5.0, t3); err != nil {
		t.Fatalf("Tick failed: %v", err)
	}
	if gov.CurrentState() != StateCoolingDown {
		t.Errorf("util drop state = %s, want %s", gov.CurrentState(), StateCoolingDown)
	}

	// 6. Workload resumes at 15s in cooldown (util = 50%) -> returns to ActivePerformance immediately
	t4 := baseTime.Add(25 * time.Second)
	if err := gov.Tick(50.0, t4); err != nil {
		t.Fatalf("Tick failed: %v", err)
	}
	if gov.CurrentState() != StateActivePerformance {
		t.Errorf("resumed workload state = %s, want %s", gov.CurrentState(), StateActivePerformance)
	}

	// 7. Workload drops to 0%
	t5 := baseTime.Add(30 * time.Second)
	_ = gov.Tick(0.0, t5)
	if gov.CurrentState() != StateCoolingDown {
		t.Errorf("drop 2 state = %s, want %s", gov.CurrentState(), StateCoolingDown)
	}

	// 8. 30s into cooldown (total 30s < 60s window) -> still CoolingDown
	t6 := t5.Add(30 * time.Second)
	_ = gov.Tick(0.0, t6)
	if gov.CurrentState() != StateCoolingDown {
		t.Errorf("cooldown 30s state = %s, want %s", gov.CurrentState(), StateCoolingDown)
	}

	// 9. 60s into sustained cooldown -> transitions to Idle and restores defaults
	t7 := t5.Add(60 * time.Second)
	if err := gov.Tick(0.0, t7); err != nil {
		t.Fatalf("Tick failed: %v", err)
	}
	if gov.CurrentState() != StateIdle {
		t.Errorf("cooldown 60s state = %s, want %s", gov.CurrentState(), StateIdle)
	}
	if gov.FanDutyLocked || gov.CurrentFanDuty != -1 {
		t.Errorf("FanDuty = %d, want -1 (automatic)", gov.CurrentFanDuty)
	}
	if gov.PowerProfileLocked || gov.CurrentPowerProfile != "balanced" {
		t.Errorf("PowerProfile = %s, want balanced", gov.CurrentPowerProfile)
	}

	if len(gov.TransitionLog) < 4 {
		t.Errorf("TransitionLog entries = %d, want >= 4", len(gov.TransitionLog))
	}
}

func TestWorkloadGovernor_CleanRestorationOnTermination(t *testing.T) {
	var restoredCalls []string
	cfg := DefaultWorkloadGovernorConfig()
	cfg.Executor = func(cmd string, args ...string) error {
		restoredCalls = append(restoredCalls, fmt.Sprintf("%s %v", cmd, args))
		return nil
	}

	gov := NewWorkloadGovernor(cfg)
	now := time.Now()

	// Ramp to performance
	_ = gov.Tick(50.0, now)
	_ = gov.Tick(50.0, now.Add(2*time.Second))

	if gov.CurrentState() != StateActivePerformance {
		t.Fatalf("expected ActivePerformance state")
	}

	// Terminate / Restore
	if err := gov.Restore(); err != nil {
		t.Fatalf("Restore failed: %v", err)
	}

	if gov.CurrentState() != StateIdle {
		t.Errorf("restored state = %s, want %s", gov.CurrentState(), StateIdle)
	}
	if gov.FanDutyLocked {
		t.Errorf("FanDutyLocked = true after restore, want false")
	}
	if gov.PowerProfileLocked {
		t.Errorf("PowerProfileLocked = true after restore, want false")
	}
}
