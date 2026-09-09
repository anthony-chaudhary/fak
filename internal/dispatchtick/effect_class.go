package dispatchtick

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// EffectClass is the operational side-effect classification of a scheduled or recurring task.
type EffectClass string

const (
	// EffectReadOnly tasks read workspace/host state and perform no mutation.
	EffectReadOnly EffectClass = "read_only"

	// EffectSingletonMaintenance tasks are idempotent operations where only a single instance
	// per key may execute at a time (concurrent runs coalesce or refuse).
	EffectSingletonMaintenance EffectClass = "singleton_maintenance"

	// EffectLeasedMutation tasks mutate git, state, or queues and require an explicit path or lane lease.
	EffectLeasedMutation EffectClass = "leased_mutation"

	// EffectHeavyCompute tasks execute heavy compilation, test matrix, or benchmark suites
	// with measured resource weights contracting available worker slots.
	EffectHeavyCompute EffectClass = "heavy_compute"
)

// Valid returns true if the effect class is one of the recognized categories.
func (e EffectClass) Valid() bool {
	switch e {
	case EffectReadOnly, EffectSingletonMaintenance, EffectLeasedMutation, EffectHeavyCompute:
		return true
	default:
		return false
	}
}

// MaintenanceProfile defines the resource footprint and effect contract of a recurring task.
type MaintenanceProfile struct {
	ID                   string      `json:"id"`
	Key                  string      `json:"key"`
	EffectClass          EffectClass `json:"effect_class"`
	CapacityWeight       float64     `json:"capacity_weight"`
	LeasePath            string      `json:"lease_path,omitempty"`
	LeaseLane            string      `json:"lease_lane,omitempty"`
	AllowsRecursiveSpawn bool        `json:"allows_recursive_spawn,omitempty"`
	SpawnsWorkers        bool        `json:"spawns_workers,omitempty"`
}

// DefaultMaintenanceProfile returns the fail-closed profile for unprofiled tasks:
// full-weight mutating work requiring explicit leases.
func DefaultMaintenanceProfile(id, key string) MaintenanceProfile {
	return MaintenanceProfile{
		ID:             id,
		Key:            key,
		EffectClass:    EffectLeasedMutation,
		CapacityWeight: 1.0,
	}
}

// MaintenancePreflightReport is the JSON-serializable diagnostic preflight report.
type MaintenancePreflightReport struct {
	Admitted           bool                 `json:"admitted"`
	RefusalReason      string               `json:"refusal_reason,omitempty"`
	EngineeringWorkers int                  `json:"engineering_workers"`
	MaintenanceRuns    int                  `json:"maintenance_runs"`
	TotalCapacityUnits float64              `json:"total_capacity_units"`
	WeightedUnitsUsed  float64              `json:"weighted_units_used"`
	RemainingUnits     float64              `json:"remaining_units"`
	CollisionKeys      []string             `json:"collision_keys,omitempty"`
	BindingLimiter     string               `json:"binding_limiter,omitempty"`
	AdmittedTasks      []string             `json:"admitted_tasks,omitempty"`
	Profiles           []MaintenanceProfile `json:"profiles,omitempty"`
}

// MaintenancePreflightEngine evaluates candidate maintenance and recurring tasks against host capacity.
type MaintenancePreflightEngine struct {
	HostCapacityUnits  float64 // Total capacity in weighted units (e.g., number of worker slots)
	EngineeringWorkers int     // Currently active engineering workers (each counts as 1.0 unit)
	ActiveMaintenance  []MaintenanceProfile
}

// EvaluatePreflight evaluates a batch of candidate tasks and returns the admission verdict and report.
func (e *MaintenancePreflightEngine) EvaluatePreflight(candidates []MaintenanceProfile) MaintenancePreflightReport {
	report := MaintenancePreflightReport{
		EngineeringWorkers: e.EngineeringWorkers,
		TotalCapacityUnits: e.HostCapacityUnits,
		Admitted:           true,
	}

	usedUnits := float64(e.EngineeringWorkers) * 1.0
	activeKeys := make(map[string]bool)
	singletonHeld := make(map[string]string) // key -> task ID

	// Track existing active maintenance
	for _, m := range e.ActiveMaintenance {
		usedUnits += m.CapacityWeight
		report.MaintenanceRuns++
		if m.Key != "" {
			activeKeys[m.Key] = true
			if m.EffectClass == EffectSingletonMaintenance {
				singletonHeld[m.Key] = m.ID
			}
		}
	}

	var collisions []string
	var admittedTasks []string
	var profilesEvaluated []MaintenanceProfile

	for _, cand := range candidates {
		// Rule 1: Validate effect class; default unprofiled to full-weight mutating work
		if !cand.EffectClass.Valid() {
			cand = DefaultMaintenanceProfile(cand.ID, cand.Key)
		}

		// Rule 2: Zero-cost maintenance is strictly forbidden
		if cand.CapacityWeight <= 0 {
			report.Admitted = false
			report.RefusalReason = fmt.Sprintf("task %q: zero-cost maintenance is strictly forbidden (capacity_weight must be > 0, got %v)", cand.ID, cand.CapacityWeight)
			report.BindingLimiter = "zero_cost_forbidden"
			break
		}

		// Rule 3: Mutating reconciliation without an explicit path/lane lease is refused
		if cand.EffectClass == EffectLeasedMutation {
			if strings.TrimSpace(cand.LeasePath) == "" && strings.TrimSpace(cand.LeaseLane) == "" {
				report.Admitted = false
				report.RefusalReason = fmt.Sprintf("task %q: mutating reconciliation without an explicit path/lane lease is refused even if labeled maintenance", cand.ID)
				report.BindingLimiter = "unleased_mutation_refusal"
				break
			}
		}

		// Rule 4: Recursive spawning from maintenance is refused unless explicitly modeled as a dispatcher
		if cand.SpawnsWorkers && !cand.AllowsRecursiveSpawn {
			report.Admitted = false
			report.RefusalReason = fmt.Sprintf("task %q: recursive spawning from maintenance is refused unless explicitly modeled as a dispatcher", cand.ID)
			report.BindingLimiter = "recursive_spawn_forbidden"
			break
		}

		// Rule 5: Singleton maintenance collision check: two singletons with the same key coalesce/refuse
		if cand.EffectClass == EffectSingletonMaintenance && cand.Key != "" {
			if holder, exists := singletonHeld[cand.Key]; exists {
				collisions = append(collisions, cand.Key)
				report.Admitted = false
				report.RefusalReason = fmt.Sprintf("singleton maintenance collision on key %q (already held by %q)", cand.Key, holder)
				report.BindingLimiter = "singleton_collision"
				break
			}
			singletonHeld[cand.Key] = cand.ID
		}

		// Rule 6: Capacity check (weighted footprint)
		if e.HostCapacityUnits > 0 && (usedUnits+cand.CapacityWeight) > e.HostCapacityUnits {
			report.Admitted = false
			report.RefusalReason = fmt.Sprintf("capacity ceiling exceeded: task %q requires %v units, only %v units remaining",
				cand.ID, cand.CapacityWeight, e.HostCapacityUnits-usedUnits)
			report.BindingLimiter = "capacity_ceiling"
			break
		}

		usedUnits += cand.CapacityWeight
		report.MaintenanceRuns++
		admittedTasks = append(admittedTasks, cand.ID)
		profilesEvaluated = append(profilesEvaluated, cand)
		if cand.Key != "" {
			activeKeys[cand.Key] = true
		}
	}

	sort.Strings(collisions)
	report.CollisionKeys = collisions
	report.WeightedUnitsUsed = usedUnits
	if e.HostCapacityUnits > 0 {
		report.RemainingUnits = e.HostCapacityUnits - usedUnits
		if report.RemainingUnits < 0 {
			report.RemainingUnits = 0
		}
	}
	report.AdmittedTasks = admittedTasks
	report.Profiles = profilesEvaluated

	return report
}

// JSON returns the indented JSON string of the report.
func (r MaintenancePreflightReport) JSON() (string, error) {
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return "", err
	}
	return string(b), nil
}
