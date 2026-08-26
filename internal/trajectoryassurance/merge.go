package trajectoryassurance

import "fmt"

// MergeInput joins adapters only when their declared identities agree. Empty
// layers are ignored so unavailable sources stay UNKNOWN in Assess.
func MergeInput(dst *Input, src Input) error {
	if err := mergeIdentity(&dst.ObjectiveID, src.ObjectiveID, "objective"); err != nil {
		return err
	}
	if err := mergeIdentity(&dst.TrajectoryID, src.TrajectoryID, "trajectory"); err != nil {
		return err
	}
	if err := mergeIdentity(&dst.SessionID, src.SessionID, "session"); err != nil {
		return err
	}
	if err := mergeIdentity(&dst.RunID, src.RunID, "run"); err != nil {
		return err
	}
	if src.ObservationWindow != "" {
		if dst.ObservationWindow != "" && dst.ObservationWindow != src.ObservationWindow {
			dst.ObservationWindow = dst.ObservationWindow + "," + src.ObservationWindow
		} else {
			dst.ObservationWindow = src.ObservationWindow
		}
	}
	if len(src.DeterministicFloor) > 0 {
		dst.DeterministicFloor = append(dst.DeterministicFloor, src.DeterministicFloor...)
	}
	if populatedObservation(src.ObjectiveProgress) {
		dst.ObjectiveProgress = src.ObjectiveProgress
	}
	if populatedEfficiency(src.Efficiency) {
		dst.Efficiency = src.Efficiency
	}
	if populatedObservation(src.DelegationIntegrity) {
		dst.DelegationIntegrity = src.DelegationIntegrity
	}
	if populatedObservation(src.SemanticReview) {
		dst.SemanticReview = src.SemanticReview
	}
	return nil
}
func mergeIdentity(dst *string, src, name string) error {
	if src == "" {
		return nil
	}
	if *dst != "" && *dst != src {
		return fmt.Errorf("trajectory assurance: %s identity mismatch: %q != %q", name, *dst, src)
	}
	*dst = src
	return nil
}
func populatedObservation(o Observation) bool { return o.State != "" || o.Evidence.Source != "" }
func populatedEfficiency(e EfficiencyInput) bool {
	return e.Outcome != nil || e.ConstraintsSatisfied != nil || e.ParentUnits != nil || e.Evidence.Source != ""
}
