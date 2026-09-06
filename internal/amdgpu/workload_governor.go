// Package amdgpu provides AMD GPU facts probing, hardware governor settings,
// and closed-loop workload-aware thermal and power governors for APU inference.
package amdgpu

import (
	"fmt"
	"sync"
	"time"
)

// WorkloadState represents the current operating thermal and power state of the APU governor.
type WorkloadState string

const (
	StateIdle              WorkloadState = "idle"
	StateActivePerformance WorkloadState = "active_performance"
	StateCoolingDown       WorkloadState = "cooling_down"

	DefaultGPUUtilThreshold    = 20.0             // 20% GPU utilization threshold for active inference
	DefaultActiveTriggerWindow = 2 * time.Second  // 2 consecutive seconds active to ramp fan/power
	DefaultIdleCooldownWindow  = 60 * time.Second // 60 seconds sustained idle before restoring defaults
	DefaultReconcileInterval   = 15 * time.Second // 15 seconds periodic watchdog reconciliation
)

// WorkloadGovernorConfig holds parameters and callbacks for the APU thermal and power governor.
type WorkloadGovernorConfig struct {
	GPUUtilThreshold    float64
	ActiveTriggerWindow time.Duration
	IdleCooldownWindow  time.Duration
	ReconcileInterval   time.Duration
	Executor            func(cmd string, args ...string) error
}

// DefaultWorkloadGovernorConfig returns standard operational settings for mobile and mini-PC APUs.
func DefaultWorkloadGovernorConfig() WorkloadGovernorConfig {
	return WorkloadGovernorConfig{
		GPUUtilThreshold:    DefaultGPUUtilThreshold,
		ActiveTriggerWindow: DefaultActiveTriggerWindow,
		IdleCooldownWindow:  DefaultIdleCooldownWindow,
		ReconcileInterval:   DefaultReconcileInterval,
		Executor: func(cmd string, args ...string) error {
			return nil
		},
	}
}

// WorkloadGovernor manages autonomous closed-loop fan duty and power profiles for APU workstations.
type WorkloadGovernor struct {
	mu           sync.Mutex
	cfg          WorkloadGovernorConfig
	currentState WorkloadState

	activeStart time.Time
	idleStart   time.Time
	lastTick    time.Time

	FanDutyLocked       bool
	PowerProfileLocked  bool
	CurrentFanDuty      int
	CurrentPowerProfile string

	TransitionLog []string
}

// NewWorkloadGovernor creates a governor with the specified configuration.
func NewWorkloadGovernor(cfg WorkloadGovernorConfig) *WorkloadGovernor {
	if cfg.GPUUtilThreshold <= 0 {
		cfg.GPUUtilThreshold = DefaultGPUUtilThreshold
	}
	if cfg.ActiveTriggerWindow <= 0 {
		cfg.ActiveTriggerWindow = DefaultActiveTriggerWindow
	}
	if cfg.IdleCooldownWindow <= 0 {
		cfg.IdleCooldownWindow = DefaultIdleCooldownWindow
	}
	if cfg.ReconcileInterval <= 0 {
		cfg.ReconcileInterval = DefaultReconcileInterval
	}
	if cfg.Executor == nil {
		cfg.Executor = func(cmd string, args ...string) error { return nil }
	}

	return &WorkloadGovernor{
		cfg:                 cfg,
		currentState:        StateIdle,
		CurrentFanDuty:      -1, // -1 represents automatic OS/firmware control
		CurrentPowerProfile: "balanced",
		TransitionLog:       make([]string, 0),
	}
}

// CurrentState returns the current operating state of the governor.
func (g *WorkloadGovernor) CurrentState() WorkloadState {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.currentState
}

// Tick evaluates a periodic GPU utilization measurement and updates the state machine.
func (g *WorkloadGovernor) Tick(gpuUtilPct float64, now time.Time) error {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.lastTick = now

	switch g.currentState {
	case StateIdle:
		if gpuUtilPct >= g.cfg.GPUUtilThreshold {
			if g.activeStart.IsZero() {
				g.activeStart = now
			} else if now.Sub(g.activeStart) >= g.cfg.ActiveTriggerWindow {
				// Transition to Active Performance
				return g.transitionToPerformance(now)
			}
		} else {
			g.activeStart = time.Time{}
		}

	case StateActivePerformance:
		if gpuUtilPct < g.cfg.GPUUtilThreshold {
			g.currentState = StateCoolingDown
			g.idleStart = now
			g.activeStart = time.Time{}
			g.logTransition(fmt.Sprintf("%s: active -> cooling_down (GPU util %.1f%% < %.1f%%)",
				now.Format(time.RFC3339), gpuUtilPct, g.cfg.GPUUtilThreshold))
		}

	case StateCoolingDown:
		if gpuUtilPct >= g.cfg.GPUUtilThreshold {
			// Resumed active workload during cooldown
			g.currentState = StateActivePerformance
			g.idleStart = time.Time{}
			g.logTransition(fmt.Sprintf("%s: cooling_down -> active_performance (workload resumed at %.1f%%)",
				now.Format(time.RFC3339), gpuUtilPct))
		} else if now.Sub(g.idleStart) >= g.cfg.IdleCooldownWindow {
			// Sustained idle cooldown satisfied, restore automatic balanced profile
			return g.transitionToIdle(now)
		}
	}

	return nil
}

func (g *WorkloadGovernor) transitionToPerformance(now time.Time) error {
	g.currentState = StateActivePerformance
	g.activeStart = time.Time{}
	g.idleStart = time.Time{}

	// 1. Lock fan duty to 100% via ectool to prevent thermal saturation
	if err := g.cfg.Executor("ectool", "fanduty", "100"); err != nil {
		return fmt.Errorf("amdgpu: failed to ramp fan duty: %w", err)
	}
	g.FanDutyLocked = true
	g.CurrentFanDuty = 100

	// 2. Lock TuneD profile to accelerator-performance
	if err := g.cfg.Executor("tuned-adm", "profile", "accelerator-performance"); err != nil {
		return fmt.Errorf("amdgpu: failed to set power profile: %w", err)
	}
	g.PowerProfileLocked = true
	g.CurrentPowerProfile = "accelerator-performance"

	g.logTransition(fmt.Sprintf("%s: idle -> active_performance (locked fan 100%%, tuned accelerator-performance)",
		now.Format(time.RFC3339)))
	return nil
}

func (g *WorkloadGovernor) transitionToIdle(now time.Time) error {
	g.currentState = StateIdle
	g.idleStart = time.Time{}
	g.activeStart = time.Time{}

	// 1. Restore automatic fan control
	if err := g.cfg.Executor("ectool", "autofanctrl"); err != nil {
		return fmt.Errorf("amdgpu: failed to restore automatic fan control: %w", err)
	}
	g.FanDutyLocked = false
	g.CurrentFanDuty = -1

	// 2. Restore balanced power profile
	if err := g.cfg.Executor("tuned-adm", "profile", "balanced"); err != nil {
		return fmt.Errorf("amdgpu: failed to restore balanced profile: %w", err)
	}
	g.PowerProfileLocked = false
	g.CurrentPowerProfile = "balanced"

	g.logTransition(fmt.Sprintf("%s: cooling_down -> idle (restored autofanctrl, balanced profile)",
		now.Format(time.RFC3339)))
	return nil
}

// Restore resets fan duty and power profiles to OS defaults upon shutdown or termination.
func (g *WorkloadGovernor) Restore() error {
	g.mu.Lock()
	defer g.mu.Unlock()

	if g.FanDutyLocked {
		_ = g.cfg.Executor("ectool", "autofanctrl")
		g.FanDutyLocked = false
		g.CurrentFanDuty = -1
	}
	if g.PowerProfileLocked {
		_ = g.cfg.Executor("tuned-adm", "profile", "balanced")
		g.PowerProfileLocked = false
		g.CurrentPowerProfile = "balanced"
	}
	g.currentState = StateIdle
	return nil
}

func (g *WorkloadGovernor) logTransition(entry string) {
	g.TransitionLog = append(g.TransitionLog, entry)
}
