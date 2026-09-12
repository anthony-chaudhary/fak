package model

import "fmt"

// MemoryComponentKind identifies an independently budgeted session memory pool.
type MemoryComponentKind string

const (
	MemoryComponentKV        MemoryComponentKind = "kv"
	MemoryComponentRecurrent MemoryComponentKind = "recurrent"
	MemoryComponentExpert    MemoryComponentKind = "expert"
	MemoryComponentScratch   MemoryComponentKind = "scratch"
)

// MemoryComponent describes bytes required by a pool without allocating it.
// A zero-byte component is permitted and still reserves its kind in the plan.
type MemoryComponent struct {
	Kind  MemoryComponentKind
	Bytes int64
}

// MemoryComponentPlan accounts for session memory independently of pool allocators.
type MemoryComponentPlan struct {
	Components []MemoryComponent
	TotalBytes int64
}

// PlanMemoryComponents validates each component and the combined device budget.
// A zero budget permits only zero-byte components or an empty plan.
// It returns no partial plan on failure and copies descriptors on success so
// callers may reuse their input slice without changing the validated result.
func PlanMemoryComponents(components []MemoryComponent, budgetBytes int64) (MemoryComponentPlan, error) {
	invalid := func(format string, args ...any) (MemoryComponentPlan, error) {
		return MemoryComponentPlan{}, &CacheRebuildError{Reason: CacheRebuildInvalidBudget, Err: fmt.Errorf(format, args...)}
	}
	if budgetBytes < 0 {
		return invalid("device budget must be non-negative")
	}
	seen := make(map[MemoryComponentKind]bool, 4)
	var total int64
	for _, component := range components {
		switch component.Kind {
		case MemoryComponentKV, MemoryComponentRecurrent, MemoryComponentExpert, MemoryComponentScratch:
		default:
			return invalid("unknown memory component %q", component.Kind)
		}
		if seen[component.Kind] {
			return invalid("duplicate memory component %q", component.Kind)
		}
		seen[component.Kind] = true
		if component.Bytes < 0 {
			return invalid("memory component %q bytes must be non-negative", component.Kind)
		}
		var ok bool
		total, ok = checkedAdd(total, component.Bytes)
		if !ok {
			return invalid("memory component %q overflows total bytes", component.Kind)
		}
		if total > budgetBytes {
			return invalid("requested %d bytes exceeds device budget %d", total, budgetBytes)
		}
	}
	return MemoryComponentPlan{Components: append([]MemoryComponent(nil), components...), TotalBytes: total}, nil
}
