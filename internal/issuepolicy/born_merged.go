package issuepolicy

import "strings"

// Closed flag constants for BornMerged readout.
const (
	BornMergedPathsMissing          = "born_merged_paths_missing"
	BornMergedPathsUnbounded        = "born_merged_paths_unbounded"
	BornMergedClosureBindingMissing = "born_merged_closure_binding_missing"
	BornMergedTrailerMissing        = "born_merged_trailer_missing"
	BornMergedAtomicUnitInvalid     = "born_merged_atomic_unit_invalid"
)

// BornMerged is the issue-creation convergence and atomic leaf dispatchability readout.
type BornMerged struct {
	PathsBounded        bool     `json:"paths_bounded"`
	ClosureBindingValid bool     `json:"closure_binding_valid"`
	AtomicUnitValid     bool     `json:"atomic_unit_valid"`
	Flags               []string `json:"flags,omitempty"`
}

// bornMerged evaluates whether a candidate issue satisfies the strict-born-merged
// contract for atomic leaf dispatchability: bounded file paths, valid closure
// binding with conventional ship trailer, and atomic unit scaling.
func bornMerged(c Candidate) BornMerged {
	var r BornMerged

	// Paths: If len(c.Paths) == 0, flag BornMergedPathsMissing.
	// If any path contains bare wildcards or unbounded pattern (e.g. contains * or ...),
	// flag BornMergedPathsUnbounded. If neither, PathsBounded = true.
	if len(c.Paths) == 0 {
		r.Flags = append(r.Flags, BornMergedPathsMissing)
	} else {
		unbounded := false
		for _, p := range c.Paths {
			if strings.Contains(p, "*") || strings.Contains(p, "...") {
				unbounded = true
				break
			}
		}
		if unbounded {
			r.Flags = append(r.Flags, BornMergedPathsUnbounded)
		} else {
			r.PathsBounded = true
		}
	}

	// Closure Binding: Check c.ClosureBinding (trimmed).
	// If empty, flag BornMergedClosureBindingMissing.
	// Also must contain conventional trailer pattern like (fak  or (fak-private  or ( + fak (e.g. (fak <lane>)).
	// If trailer missing, flag BornMergedTrailerMissing. If neither, ClosureBindingValid = true.
	closure := strings.TrimSpace(c.ClosureBinding)
	if closure == "" {
		r.Flags = append(r.Flags, BornMergedClosureBindingMissing)
	} else {
		lower := strings.ToLower(closure)
		hasTrailer := strings.Contains(lower, "(fak ") || strings.Contains(lower, "(fak-private ") || strings.Contains(lower, "(fak")
		if !hasTrailer {
			r.Flags = append(r.Flags, BornMergedTrailerMissing)
		} else {
			r.ClosureBindingValid = true
		}
	}

	// Atomic Unit: Check unit scale: scale should be S0 or S1, or unit is "leaf", or expected steps <= 4.
	// If oversized or invalid (e.g. scale is S2/S3, or Unit is "epic", or ExpectedSteps > 8),
	// flag BornMergedAtomicUnitInvalid. If valid, AtomicUnitValid = true.
	scale, okScale := parseScale(c.Scale)
	unit := strings.ToLower(strings.TrimSpace(c.WorkUnit))
	unitScale, okUnitScale := scaleFromWorkUnit(unit)

	oversizedOrInvalid := false
	if strings.TrimSpace(c.Scale) != "" {
		if !okScale || (scale != ScaleStep && scale != ScaleLeaf) {
			oversizedOrInvalid = true
		}
	}
	if unit == "epic" || isNonDispatchWorkUnit(unit) || (okUnitScale && scaleRank(unitScale) >= scaleRank(ScaleFeature)) {
		oversizedOrInvalid = true
	}
	if c.ExpectedSteps > MaxDispatchExpectedSteps || c.ExpectedSteps < 0 {
		oversizedOrInvalid = true
	}

	atomicSignal := false
	if okScale && (scale == ScaleStep || scale == ScaleLeaf) {
		atomicSignal = true
	}
	if unit == "leaf" || isDispatchWorkUnit(unit) {
		atomicSignal = true
	}
	if c.ExpectedSteps > 0 && c.ExpectedSteps <= 4 {
		atomicSignal = true
	}

	if !oversizedOrInvalid && atomicSignal {
		r.AtomicUnitValid = true
	} else {
		r.Flags = append(r.Flags, BornMergedAtomicUnitInvalid)
	}

	return r
}
