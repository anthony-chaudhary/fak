package amdgpu

import (
	"fmt"
	"testing"
	"time"
)

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
