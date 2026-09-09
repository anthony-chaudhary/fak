package workerworktree

import (
	"sync/atomic"
	"time"
)

// TriggerReceiptSchema is the canonical schema for worktree trigger receipts.
const TriggerReceiptSchema = "fak-worktree-trigger/1"

// TriggerKind identifies the cause or condition that evaluated the trigger.
type TriggerKind string

const (
	TriggerStaleSweep          TriggerKind = "TriggerStaleSweep"
	TriggerCapacityContraction TriggerKind = "TriggerCapacityContraction"
	TriggerDiskPressure        TriggerKind = "TriggerDiskPressure"
	TriggerPostLandReap        TriggerKind = "TriggerPostLandReap"
)

// TriggerDecision is the outcome of a trigger evaluation.
type TriggerDecision string

const (
	DecisionExecuted             TriggerDecision = "EXECUTED"
	DecisionSkippedCooldown      TriggerDecision = "SKIPPED_COOLDOWN"
	DecisionSkippedLockHeld      TriggerDecision = "SKIPPED_LOCK_HELD"
	DecisionSkippedUnderCapacity TriggerDecision = "SKIPPED_UNDER_CAPACITY"
	DecisionRefusedSafety        TriggerDecision = "REFUSED_SAFETY"

	// TriggerDecision* aliases
	TriggerDecisionExecuted             = DecisionExecuted
	TriggerDecisionSkippedCooldown      = DecisionSkippedCooldown
	TriggerDecisionSkippedLockHeld      = DecisionSkippedLockHeld
	TriggerDecisionSkippedUnderCapacity = DecisionSkippedUnderCapacity
	TriggerDecisionRefusedSafety        = DecisionRefusedSafety
)

// RefusalReason classifies why a trigger was skipped or refused.
type RefusalReason string

const (
	RefusalCooldownActive           RefusalReason = "COOLDOWN_ACTIVE"
	RefusalLockContention           RefusalReason = "LOCK_CONTENTION"
	RefusalBelowCapacitySetpoint    RefusalReason = "BELOW_CAPACITY_SETPOINT"
	RefusalOwnerOrLeaseActive       RefusalReason = "OWNER_OR_LEASE_ACTIVE"
	RefusalUnlandedWorkHeld         RefusalReason = "UNLANDED_WORK_HELD"
	RefusalForeignPlatformPreserved RefusalReason = "FOREIGN_PLATFORM_PRESERVED"

	// RefusalReason* aliases
	RefusalReasonCooldownActive           = RefusalCooldownActive
	RefusalReasonLockContention           = RefusalLockContention
	RefusalReasonBelowCapacitySetpoint    = RefusalBelowCapacitySetpoint
	RefusalReasonOwnerOrLeaseActive       = RefusalOwnerOrLeaseActive
	RefusalReasonUnlandedWorkHeld         = RefusalUnlandedWorkHeld
	RefusalReasonForeignPlatformPreserved = RefusalForeignPlatformPreserved

	// Raw string constants for convenience
	ReasonCooldownActive           = "COOLDOWN_ACTIVE"
	ReasonLockContention           = "LOCK_CONTENTION"
	ReasonBelowCapacitySetpoint    = "BELOW_CAPACITY_SETPOINT"
	ReasonOwnerOrLeaseActive       = "OWNER_OR_LEASE_ACTIVE"
	ReasonUnlandedWorkHeld         = "UNLANDED_WORK_HELD"
	ReasonForeignPlatformPreserved = "FOREIGN_PLATFORM_PRESERVED"
)

// TriggerOptions controls trigger setpoints and debouncing behavior.
type TriggerOptions struct {
	StaleSweepCooldown   time.Duration `json:"stale_sweep_cooldown"`
	CapacitySetpointHigh int           `json:"capacity_setpoint_high"`
	CapacitySetpointLow  int           `json:"capacity_setpoint_low"`
	DiskReserveBytes     int64         `json:"disk_reserve_bytes"`
}

// DefaultTriggerOptions returns standard production defaults for triggers.
func DefaultTriggerOptions() TriggerOptions {
	return TriggerOptions{
		StaleSweepCooldown:   30 * time.Second,
		CapacitySetpointHigh: 50,
		CapacitySetpointLow:  35,
		DiskReserveBytes:     2 * 1024 * 1024 * 1024, // 2GB
	}
}

// TriggerReceipt captures the auditable outcome, duration, and effects of a trigger pass.
type TriggerReceipt struct {
	Schema         string          `json:"schema"`
	Trigger        TriggerKind     `json:"trigger"`
	Decision       TriggerDecision `json:"decision"`
	Reason         string          `json:"reason,omitempty"`
	RefusalCode    RefusalReason   `json:"refusal_code,omitempty"`
	InspectedCount int             `json:"inspected_count"`
	ReapedCount    int             `json:"reaped_count"`
	RetainedCount  int             `json:"retained_count"`
	FreedBytes     int64           `json:"freed_bytes"`
	DurationNS     int64           `json:"duration_ns"`
	ReapedPaths    []string        `json:"reaped_paths,omitempty"`
	Timestamp      time.Time       `json:"timestamp"`
}

// DebouncedSweepGate provides single-flight, debounced execution for dead worktree sweeps.
type DebouncedSweepGate struct {
	lastSweepEpochNano atomic.Int64
	inFlight           atomic.Int32
}

// NewDebouncedSweepGate constructs an initialized DebouncedSweepGate.
func NewDebouncedSweepGate() *DebouncedSweepGate {
	return &DebouncedSweepGate{}
}

// DefaultSweepGate is the process-wide shared sweep gate.
var DefaultSweepGate = NewDebouncedSweepGate()

// ResetSweepGateForTest resets the shared default sweep gate state for testing.
func ResetSweepGateForTest() {
	if DefaultSweepGate != nil {
		DefaultSweepGate.Reset()
	} else {
		DefaultSweepGate = NewDebouncedSweepGate()
	}
}

// Reset clears recorded timestamps and flight state on this gate.
func (g *DebouncedSweepGate) Reset() {
	g.lastSweepEpochNano.Store(0)
	g.inFlight.Store(0)
}

// TrySweep attempts to perform a dead worktree sweep through the gate.
// If force is false and time since the last sweep is under cooldown, it returns
// Decision: SKIPPED_COOLDOWN and RefusalCode: COOLDOWN_ACTIVE immediately (<1µs).
// If a sweep is already in flight, it returns Decision: SKIPPED_LOCK_HELD and
// RefusalCode: LOCK_CONTENTION.
// When execution proceeds, it runs SweepDeadWorktrees, records the receipt, and returns Decision: EXECUTED.
func (g *DebouncedSweepGate) TrySweep(root, wtRoot string, git GitRunner, cooldown time.Duration, force bool) (TriggerReceipt, bool) {
	start := time.Now()
	if cooldown <= 0 {
		cooldown = DefaultTriggerOptions().StaleSweepCooldown
	}

	last := g.lastSweepEpochNano.Load()
	if !force && last > 0 {
		lastTime := time.Unix(0, last)
		if elapsed := time.Since(lastTime); elapsed < cooldown {
			return TriggerReceipt{
				Schema:         TriggerReceiptSchema,
				Trigger:        TriggerStaleSweep,
				Decision:       DecisionSkippedCooldown,
				Reason:         "sweep skipped: cooldown active",
				RefusalCode:    RefusalCooldownActive,
				InspectedCount: 0,
				ReapedCount:    0,
				RetainedCount:  0,
				FreedBytes:     0,
				DurationNS:     time.Since(start).Nanoseconds(),
				Timestamp:      time.Now().UTC(),
			}, false
		}
	}

	if !g.inFlight.CompareAndSwap(0, 1) {
		return TriggerReceipt{
			Schema:         TriggerReceiptSchema,
			Trigger:        TriggerStaleSweep,
			Decision:       DecisionSkippedLockHeld,
			Reason:         "sweep skipped: concurrent sweep in progress",
			RefusalCode:    RefusalLockContention,
			InspectedCount: 0,
			ReapedCount:    0,
			RetainedCount:  0,
			FreedBytes:     0,
			DurationNS:     time.Since(start).Nanoseconds(),
			Timestamp:      time.Now().UTC(),
		}, false
	}
	defer g.inFlight.Store(0)

	// Re-check cooldown after acquiring the lock in case a peer swept concurrently.
	if !force {
		last = g.lastSweepEpochNano.Load()
		if last > 0 {
			lastTime := time.Unix(0, last)
			if elapsed := time.Since(lastTime); elapsed < cooldown {
				return TriggerReceipt{
					Schema:         TriggerReceiptSchema,
					Trigger:        TriggerStaleSweep,
					Decision:       DecisionSkippedCooldown,
					Reason:         "sweep skipped: cooldown active",
					RefusalCode:    RefusalCooldownActive,
					InspectedCount: 0,
					ReapedCount:    0,
					RetainedCount:  0,
					FreedBytes:     0,
					DurationNS:     time.Since(start).Nanoseconds(),
					Timestamp:      time.Now().UTC(),
				}, false
			}
		}
	}

	report := SweepDeadWorktrees(root, wtRoot, git)
	now := time.Now()
	g.lastSweepEpochNano.Store(now.UnixNano())

	retained := report.Inspected - report.Pruned
	if retained < 0 {
		retained = 0
	}

	return TriggerReceipt{
		Schema:         TriggerReceiptSchema,
		Trigger:        TriggerStaleSweep,
		Decision:       DecisionExecuted,
		Reason:         "",
		RefusalCode:    "",
		InspectedCount: report.Inspected,
		ReapedCount:    report.Pruned,
		RetainedCount:  retained,
		FreedBytes:     0,
		DurationNS:     time.Since(start).Nanoseconds(),
		ReapedPaths:    report.Paths,
		Timestamp:      now.UTC(),
	}, true
}

// DebouncedSweepDeadWorktrees runs a dead worktree sweep protected by the default gate.
func DebouncedSweepDeadWorktrees(root, wtRoot string, git GitRunner) TriggerReceipt {
	receipt, _ := DefaultSweepGate.TrySweep(root, wtRoot, git, DefaultTriggerOptions().StaleSweepCooldown, false)
	return receipt
}
