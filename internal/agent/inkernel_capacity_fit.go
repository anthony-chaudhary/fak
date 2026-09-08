package agent

import "github.com/anthony-chaudhary/fak/internal/compute"

func (p *InKernelPlanner) capacityErrorFromFit(fe *compute.FitError) error {
	if fe == nil {
		return nil
	}
	scope := fe.Scope
	if scope == "" {
		scope = compute.MemoryScopeDevice
	}
	return &InKernelCapacityError{
		Want:  fe.Want,
		Avail: fe.Avail,
		Class: primaryDemandClass(fe.Demands, scope),
		Scope: scope,
		Site:  "capacity-precheck",
	}
}

func primaryDemandClass(plan compute.MemoryPlan, scope compute.MemoryScope) compute.MemoryClass {
	var bestClass compute.MemoryClass
	var bestBytes int64
	for _, d := range plan {
		if d.Bytes <= 0 || d.ScopeOrDefault() != scope {
			continue
		}
		class := d.Class
		if class == "" {
			class = compute.MemoryUnknown
		}
		if d.Bytes > bestBytes {
			bestBytes = d.Bytes
			bestClass = class
		}
	}
	if bestClass == "" {
		return compute.MemoryUnknown
	}
	return bestClass
}
