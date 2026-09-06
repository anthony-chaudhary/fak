package selfinstall

import (
	"fmt"
	"path/filepath"
	"sort"
	"strings"
)

// ComponentAcquisition records how a desired artifact became available before activation.
type ComponentAcquisition string

const (
	ComponentReuse           ComponentAcquisition = "reuse"
	ComponentTransferOrBuild ComponentAcquisition = "transfer_or_build"
)

// ComponentActivation records whether the installed bytes need to move.
type ComponentActivation string

const (
	ComponentNoop     ComponentActivation = "no_op"
	ComponentActivate ComponentActivation = "activate"
)

// Component describes one independently converging target. Components with the same
// non-empty CompatibilityGroup are staged and activated by one transaction.
type Component struct {
	Name               string
	Source             string
	Target             string
	CompatibilityGroup string
	Acquisition        ComponentAcquisition
}

// ComponentPlan is the witnessed desired-vs-installed decision for one component.
type ComponentPlan struct {
	Name                    string               `json:"name"`
	Target                  string               `json:"target"`
	CompatibilityGroup      string               `json:"compatibility_group,omitempty"`
	DesiredArtifactDigest   string               `json:"desired_artifact_digest"`
	InstalledArtifactDigest string               `json:"installed_artifact_digest"`
	Acquisition             ComponentAcquisition `json:"acquisition"`
	Activation              ComponentActivation  `json:"activation"`
	Rollback                string               `json:"rollback"`
}

// PlanComponents derives an explicit, deterministic plan from artifact bytes. Missing
// installed targets have an empty installed digest and require activation.
func PlanComponents(components []Component) ([]ComponentPlan, error) {
	plans := make([]ComponentPlan, 0, len(components))
	seen := make(map[string]struct{}, len(components))
	for i, component := range components {
		if strings.TrimSpace(component.Name) == "" || strings.TrimSpace(component.Source) == "" || strings.TrimSpace(component.Target) == "" {
			return nil, fmt.Errorf("component %d: name, source, and target are required", i)
		}
		target, err := filepath.Abs(component.Target)
		if err != nil {
			return nil, fmt.Errorf("component %q target: %w", component.Name, err)
		}
		key := filepath.Clean(target)
		if filepath.Separator == '\\' {
			key = strings.ToLower(key)
		}
		if _, ok := seen[key]; ok {
			return nil, fmt.Errorf("component %q duplicates target %q", component.Name, target)
		}
		seen[key] = struct{}{}

		desired, _, err := fileSHA256(component.Source)
		if err != nil {
			return nil, fmt.Errorf("component %q desired artifact: %w", component.Name, err)
		}
		installed, _, err := fileSHA256(target)
		if err != nil {
			return nil, fmt.Errorf("component %q installed artifact: %w", component.Name, err)
		}
		activation := ComponentActivate
		rollback := "restore_installed_artifact"
		if desired == installed {
			needsRepair, err := executableModeNeedsRepair(component.Source, target)
			if err != nil {
				return nil, fmt.Errorf("component %q mode check: %w", component.Name, err)
			}
			if !needsRepair {
				activation = ComponentNoop
				rollback = "none"
			}
		}
		acquisition := component.Acquisition
		if acquisition == "" {
			acquisition = ComponentTransferOrBuild
		}
		plans = append(plans, ComponentPlan{
			Name: component.Name, Target: target, CompatibilityGroup: component.CompatibilityGroup,
			DesiredArtifactDigest: desired, InstalledArtifactDigest: installed,
			Acquisition: acquisition, Activation: activation, Rollback: rollback,
		})
	}
	sort.Slice(plans, func(i, j int) bool { return plans[i].Target < plans[j].Target })
	return plans, nil
}

// CopiesForActivation returns only stale components. The returned set remains one atomic
// transaction, preserving compatibility groups represented in the caller's selected set.
func CopiesForActivation(components []Component, plans []ComponentPlan) []Copy {
	byTarget := make(map[string]ComponentPlan, len(plans))
	for _, plan := range plans {
		byTarget[filepath.Clean(plan.Target)] = plan
	}
	copies := make([]Copy, 0, len(components))
	for _, component := range components {
		target, err := filepath.Abs(component.Target)
		if err != nil {
			continue
		}
		if plan, ok := byTarget[filepath.Clean(target)]; ok && plan.Activation == ComponentActivate {
			copies = append(copies, Copy{Source: component.Source, Target: target})
		}
	}
	return copies
}
