package dispatchtick

import (
	"encoding/json"
	"strings"
	"testing"
)

// TestEffectClass_TwoDimensions checks orthogonal EffectClass and CapacityWeight declaration.
func TestEffectClass_TwoDimensions(t *testing.T) {
	classes := []EffectClass{
		EffectReadOnly,
		EffectSingletonMaintenance,
		EffectLeasedMutation,
		EffectHeavyCompute,
	}
	for _, c := range classes {
		if !c.Valid() {
			t.Errorf("expected class %q to be valid", c)
		}
	}
	if EffectClass("unknown").Valid() {
		t.Errorf("expected unknown class to be invalid")
	}
}

// TestEffectClass_ZeroCostMaintenanceForbidden proves that zero-cost maintenance is strictly refused.
func TestEffectClass_ZeroCostMaintenanceForbidden(t *testing.T) {
	engine := &MaintenancePreflightEngine{
		HostCapacityUnits: 4.0,
	}
	candidates := []MaintenanceProfile{
		{
			ID:             "zero-cost-janitor",
			Key:            "janitor",
			EffectClass:    EffectReadOnly,
			CapacityWeight: 0.0, // Forbidden!
		},
	}
	rep := engine.EvaluatePreflight(candidates)
	if rep.Admitted {
		t.Fatalf("zero-cost maintenance admitted, want refused")
	}
	if rep.BindingLimiter != "zero_cost_forbidden" {
		t.Errorf("limiter = %q, want zero_cost_forbidden", rep.BindingLimiter)
	}
	if !strings.Contains(rep.RefusalReason, "zero-cost maintenance is strictly forbidden") {
		t.Errorf("refusal reason missing expected wording: %s", rep.RefusalReason)
	}
}

// TestEffectClass_ReadOnlyTasksOverlap proves two read-only tasks with shareable/distinct keys
// can overlap without consuming two full engineering slots.
func TestEffectClass_ReadOnlyTasksOverlap(t *testing.T) {
	engine := &MaintenancePreflightEngine{
		HostCapacityUnits:   2.0, // 2 slots total
		EngineeringWorkers:  1,   // 1 engineering worker active (1.0 unit used, 1.0 unit left)
	}
	// Two lightweight read-only tasks, each 0.25 units (total 0.5 units < 1.0 unit remaining).
	candidates := []MaintenanceProfile{
		{
			ID:             "audit-1",
			Key:            "audit-status",
			EffectClass:    EffectReadOnly,
			CapacityWeight: 0.25,
		},
		{
			ID:             "audit-2",
			Key:            "audit-metrics",
			EffectClass:    EffectReadOnly,
			CapacityWeight: 0.25,
		},
	}
	rep := engine.EvaluatePreflight(candidates)
	if !rep.Admitted {
		t.Fatalf("read-only tasks refused: %s", rep.RefusalReason)
	}
	if rep.WeightedUnitsUsed != 1.5 {
		t.Errorf("weighted units used = %v, want 1.5", rep.WeightedUnitsUsed)
	}
	if rep.RemainingUnits != 0.5 {
		t.Errorf("remaining units = %v, want 0.5", rep.RemainingUnits)
	}
	if len(rep.AdmittedTasks) != 2 {
		t.Errorf("admitted tasks = %d, want 2", len(rep.AdmittedTasks))
	}
}

// TestEffectClass_SingletonMaintenanceCollision proves two singleton maintenance runs
// with the same key coalesce/refuse.
func TestEffectClass_SingletonMaintenanceCollision(t *testing.T) {
	engine := &MaintenancePreflightEngine{
		HostCapacityUnits: 4.0,
		ActiveMaintenance: []MaintenanceProfile{
			{
				ID:             "active-pusher",
				Key:            "push-reconciler",
				EffectClass:    EffectSingletonMaintenance,
				CapacityWeight: 0.5,
			},
		},
	}
	candidates := []MaintenanceProfile{
		{
			ID:             "duplicate-pusher",
			Key:            "push-reconciler", // Collides with active-pusher!
			EffectClass:    EffectSingletonMaintenance,
			CapacityWeight: 0.5,
		},
	}
	rep := engine.EvaluatePreflight(candidates)
	if rep.Admitted {
		t.Fatalf("singleton collision admitted, want refused")
	}
	if rep.BindingLimiter != "singleton_collision" {
		t.Errorf("limiter = %q, want singleton_collision", rep.BindingLimiter)
	}
	if len(rep.CollisionKeys) == 0 || rep.CollisionKeys[0] != "push-reconciler" {
		t.Errorf("collision keys = %+v, want ['push-reconciler']", rep.CollisionKeys)
	}
}

// TestEffectClass_MutatingReconciliationWithoutLeaseRefused proves mutating reconciliation
// without an explicit path/lane lease is refused even if labeled maintenance.
func TestEffectClass_MutatingReconciliationWithoutLeaseRefused(t *testing.T) {
	engine := &MaintenancePreflightEngine{
		HostCapacityUnits: 4.0,
	}
	// Mutating task without LeasePath or LeaseLane
	candidates := []MaintenanceProfile{
		{
			ID:             "unleased-gardener",
			Key:            "gardener",
			EffectClass:    EffectLeasedMutation,
			CapacityWeight: 0.5,
			LeasePath:      "", // No lease!
			LeaseLane:      "",
		},
	}
	rep := engine.EvaluatePreflight(candidates)
	if rep.Admitted {
		t.Fatalf("unleased mutating maintenance admitted, want refused")
	}
	if rep.BindingLimiter != "unleased_mutation_refusal" {
		t.Errorf("limiter = %q, want unleased_mutation_refusal", rep.BindingLimiter)
	}
	if !strings.Contains(rep.RefusalReason, "without an explicit path/lane lease is refused") {
		t.Errorf("refusal reason missing expected wording: %s", rep.RefusalReason)
	}

	// With explicit lease lane, it should pass.
	candidates[0].LeaseLane = "platform/dispatch"
	repWithLease := engine.EvaluatePreflight(candidates)
	if !repWithLease.Admitted {
		t.Fatalf("leased mutating maintenance refused: %s", repWithLease.RefusalReason)
	}
}

// TestEffectClass_HeavyComputeContractsHeadroom proves heavy build fixtures consume multiple
// capacity units and contract engineering headroom.
func TestEffectClass_HeavyComputeContractsHeadroom(t *testing.T) {
	engine := &MaintenancePreflightEngine{
		HostCapacityUnits:  4.0,
		EngineeringWorkers: 2, // 2 engineering slots active (2.0 units used)
	}
	// Heavy compute task requiring 2.5 units (would total 4.5 > 4.0 capacity)
	heavy := []MaintenanceProfile{
		{
			ID:             "heavy-build-matrix",
			Key:            "build-matrix",
			EffectClass:    EffectHeavyCompute,
			CapacityWeight: 2.5,
		},
	}
	repOver := engine.EvaluatePreflight(heavy)
	if repOver.Admitted {
		t.Fatalf("heavy compute over capacity admitted, want refused")
	}
	if repOver.BindingLimiter != "capacity_ceiling" {
		t.Errorf("limiter = %q, want capacity_ceiling", repOver.BindingLimiter)
	}

	// Fit within capacity (1.5 units, totaling 3.5 <= 4.0)
	heavy[0].CapacityWeight = 1.5
	repFit := engine.EvaluatePreflight(heavy)
	if !repFit.Admitted {
		t.Fatalf("fitting heavy compute refused: %s", repFit.RefusalReason)
	}
	if repFit.RemainingUnits != 0.5 {
		t.Errorf("remaining units = %v, want 0.5", repFit.RemainingUnits)
	}
}

// TestEffectClass_RecursiveSpawningRefused proves recursive spawning from maintenance
// is refused unless explicitly modeled as a dispatcher.
func TestEffectClass_RecursiveSpawningRefused(t *testing.T) {
	engine := &MaintenancePreflightEngine{
		HostCapacityUnits: 4.0,
	}
	// Spawns workers but not marked AllowsRecursiveSpawn
	candidates := []MaintenanceProfile{
		{
			ID:                   "recursive-janitor",
			Key:                  "janitor",
			EffectClass:          EffectReadOnly,
			CapacityWeight:       0.5,
			SpawnsWorkers:        true,
			AllowsRecursiveSpawn: false,
		},
	}
	rep := engine.EvaluatePreflight(candidates)
	if rep.Admitted {
		t.Fatalf("recursive spawning janitor admitted, want refused")
	}
	if rep.BindingLimiter != "recursive_spawn_forbidden" {
		t.Errorf("limiter = %q, want recursive_spawn_forbidden", rep.BindingLimiter)
	}

	// Explicitly modeled dispatcher
	candidates[0].AllowsRecursiveSpawn = true
	repDispatcher := engine.EvaluatePreflight(candidates)
	if !repDispatcher.Admitted {
		t.Fatalf("modeled dispatcher refused: %s", repDispatcher.RefusalReason)
	}
}

// TestEffectClass_JSONPreflightReportDetails checks that the JSON preflight report details
// engineering workers, maintenance runs, weighted units, collision keys, and the binding limiter separately.
func TestEffectClass_JSONPreflightReportDetails(t *testing.T) {
	engine := &MaintenancePreflightEngine{
		HostCapacityUnits:   8.0,
		EngineeringWorkers:  3,
	}
	candidates := []MaintenanceProfile{
		{
			ID:             "audit-run",
			Key:            "audit-run",
			EffectClass:    EffectReadOnly,
			CapacityWeight: 0.5,
		},
	}
	rep := engine.EvaluatePreflight(candidates)
	jsonStr, err := rep.JSON()
	if err != nil {
		t.Fatalf("failed to serialize JSON report: %v", err)
	}

	var m map[string]any
	if err := json.Unmarshal([]byte(jsonStr), &m); err != nil {
		t.Fatalf("invalid JSON emitted: %v", err)
	}

	requiredKeys := []string{
		"admitted",
		"engineering_workers",
		"maintenance_runs",
		"total_capacity_units",
		"weighted_units_used",
		"remaining_units",
	}
	for _, k := range requiredKeys {
		if _, ok := m[k]; !ok {
			t.Errorf("JSON report missing required key %q", k)
		}
	}
}

// TestEffectClass_UnprofiledTaskDefaultsToMutatingWork proves unprofiled tasks default
// to full-weight mutating work.
func TestEffectClass_UnprofiledTaskDefaultsToMutatingWork(t *testing.T) {
	unprofiled := DefaultMaintenanceProfile("legacy-task", "legacy-key")
	if unprofiled.EffectClass != EffectLeasedMutation {
		t.Errorf("default class = %q, want %q", unprofiled.EffectClass, EffectLeasedMutation)
	}
	if unprofiled.CapacityWeight != 1.0 {
		t.Errorf("default capacity weight = %v, want 1.0", unprofiled.CapacityWeight)
	}
}
